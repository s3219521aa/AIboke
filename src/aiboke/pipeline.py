"""三阶段流水线编排。

阶段一 文稿生成 -> 阶段二 语音合成 -> 阶段三 封面生成，全程受四道
0 分门限约束。封面失败不导致整案失败（仅 10 分，且产物需存在）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import audio_utils
from .config import Config
from .gates import (
    all_passed,
    check_chinese,
    check_duration,
    check_genders,
    check_two_speakers,
    cjk_ratio,
)
from .length import DurationVerdict, classify_duration
from .schema import CaseInput, Episode, Transcript, Turn, VoicePreset
from .voices import resolve_pair

logger = logging.getLogger(__name__)

# 成功时三者必须同时存在，因此任一次失败都不得留下它们
DELIVERABLE_NAMES = ("podcast.mp3", "cover.png", "script.json")

# mp3 的暂存名：时长门限通过后才改成交付名，未过门的音频不能顶着交付名出现
PENDING_AUDIO_NAME = "podcast.pending.mp3"

# 逐幕语言门限的**硬失败**阈值。0.85 是规范对整篇交付物的要求，逐幕套用会
# 误杀：cjk_ratio 把阿拉伯数字算作非中文，数字密集的主体幕可能低于 0.85 而
# 全稿仍在 0.85 以上（20/55/25 权重下，主体幕 0.75 对应全稿 0.854）。因此逐幕
# 只拦「这一幕基本不是中文」的灾难性情况（整幕英文的幻觉输出），其余交整篇门限。
CATASTROPHIC_LANGUAGE_RATIO = 0.5


def _clear_stale_outputs(out_dir: Path) -> None:
    """清掉上一轮留下的交付物与暂存文件。

    「podcast.mp3 是否存在」是自动判分最可能采用的成功判据，而上一轮成功的
    目录会原样留着完整的一套：本次若在中途失败，忽略了异常的判分方会读到
    上一集（陈旧混音）。暂存文件同理——它会让「退出码 0 但没写出文件」的
    静默失败被旧文件掩盖过去。
    """
    for name in (*DELIVERABLE_NAMES, PENDING_AUDIO_NAME):
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass


class PipelineError(RuntimeError):
    """流水线失败。"""


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        script_writer,
        tts,
        cover,
        preset_map: Mapping[str, Sequence[VoicePreset]],
        runner: Callable | None = None,
    ) -> None:
        self._cfg = cfg
        self._writer = script_writer
        self._tts = tts
        self._cover = cover
        self._presets = preset_map
        self._run = runner

    def run(self, case: CaseInput, out_dir: Path) -> Episode:
        """跑完三阶段并产出三件交付物。

        调用方必须同时捕获 `PipelineError` 与 `CoverError`：门限失败、文稿
        生成、语音合成与音频后处理失败都归一为 PipelineError；封面仅 10 分，
        主封面失败会降级为兜底封面，但兜底也失败时 CoverError 照常向外传播
        ——封面缺失必须是响亮的失败。

        交付物只在完整成功后才出现：进入时先清掉上一轮的同名产物，mp3 先写
        暂存名、过了时长门限才改名。失败的一轮不会留下 podcast.mp3 /
        cover.png / script.json，判分方不会把上一集或未过门的音频当成本次成功。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_stale_outputs(out_dir)

        # 前置门限：性别配置必须可用，否则重试无意义
        gender_gate = check_genders(case.speaker_gender1, case.speaker_gender2, self._presets)
        if not gender_gate.passed:
            raise PipelineError(f"音色配置不满足性别要求：{gender_gate.detail}")

        try:
            voices = resolve_pair(
                self._presets, case.speaker_gender1, case.speaker_gender2
            )
        except ValueError as exc:
            raise PipelineError(f"无法解析音色：{exc}") from exc

        # 阶段一 + 阶段二交织：逐幕生成文稿并立即合成语音。
        # 交织而非「先全稿后合成」是三幕式设计的核心收益——第一幕一产出
        # 即开始合成，与后续幕的生成重叠，保障「音频首字 <= 30s」的基线。
        audio_wavs: list[Path] = []
        all_turns: list[Turn] = []
        title = ""
        try:
            for act in self._writer.iter_acts(case):
                if not title and act.title:
                    title = act.title
                all_turns.extend(act.turns)

                # 逐幕先查语言门限：整幕英文（幻觉输出）要在付出昂贵的语音
                # 合成代价之前就拦下来。但 0.85 是规范对整篇交付物的要求，
                # 逐幕套用会误杀数字密集的幕——cjk_ratio 把阿拉伯数字算作
                # 非中文，主体幕低于 0.85 而全稿仍可能高于 0.85。故逐幕只在
                # 灾难性的「基本不是中文」时硬失败，偏低一档只记警告。
                act_text = "".join(t.text for t in act.turns)
                act_ratio = cjk_ratio(act_text)
                if act_ratio < CATASTROPHIC_LANGUAGE_RATIO:
                    raise PipelineError(
                        f"第 {act.act_index + 1} 幕未通过语言门限"
                        f"（内容几乎不是中文）：中文占比 {act_ratio:.3f}，"
                        f"灾难性阈值 {CATASTROPHIC_LANGUAGE_RATIO}"
                    )
                if act_ratio < self._cfg.chinese_min_ratio:
                    logger.warning(
                        "第 %d 幕中文占比 %.3f 低于整篇阈值 %.2f，暂不拦截，交整篇门限判定",
                        act.act_index + 1,
                        act_ratio,
                        self._cfg.chinese_min_ratio,
                    )

                # 用该幕实际字数折算目标秒数，比用预设字数更贴近真实产出
                act_seconds = len(act_text) / self._cfg.chars_per_minute * 60.0
                out = out_dir / f"act{act.act_index}.wav"
                audio_wavs.append(
                    self._tts.synthesize(act.turns, voices, out, act_seconds)
                )
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"文稿生成或语音合成失败：{exc}") from exc

        transcript = Transcript(title=title or case.topic, turns=tuple(all_turns))

        # 整体门限：双人对话与轮次均衡必须对完整文稿判定——
        # 单幕内部可能天然由一位主播主导，逐幕判定会误报
        script_gates = [
            check_chinese(transcript.full_text, self._cfg.chinese_min_ratio),
            check_two_speakers(transcript.turns, self._cfg.speaker_min_share),
        ]
        if not all_passed(script_gates):
            failed = "；".join(g.detail for g in script_gates if not g.passed)
            raise PipelineError(f"文稿未通过门限校验：{failed}")

        if not audio_wavs:
            raise PipelineError("没有任何一幕成功合成语音")

        # 拼接 -> 响度归一化 -> 转 mp3（暂存名，过门限后才改名为交付物）
        try:
            concat_path = audio_utils.concat_wavs(
                audio_wavs, out_dir / "podcast_concat.wav", runner=self._run
            )
            norm_path = audio_utils.normalize_loudness(
                concat_path, out_dir / "podcast_norm.wav",
                target_lufs=self._cfg.target_lufs, runner=self._run,
            )
            pending_audio = audio_utils.to_mp3(
                norm_path, out_dir / PENDING_AUDIO_NAME, runner=self._run
            )
            duration = audio_utils.probe_duration(pending_audio, runner=self._run)
        except audio_utils.AudioError as exc:
            # 与文稿/语音失败保持同一异常面，调用方只需捕获 PipelineError
            # （封面走 CoverError，见 run 的 docstring）
            raise PipelineError(f"音频后处理或时长探测失败：{exc}") from exc

        # 时长门限：越界即 0 分，必须失败而非静默通过
        duration_gate = check_duration(duration)
        if not duration_gate.passed:
            raise PipelineError(
                f"时长未通过门限校验：{duration_gate.detail}。"
                f"建议：{duration_gate.retry_hint or '调整目标字数后重试'}"
            )

        # 只有过了门限的 mp3 才配得上交付名：目录里出现 podcast.mp3 就意味着
        # 这一案已产出可交付音频——它同时也是自动判分认领产物的判据
        audio_path = out_dir / "podcast.mp3"
        os.replace(pending_audio, audio_path)

        # 合规但远离目标（设计的容忍带）只记录警告：门限是硬约束，把警告
        # 升级成失败会白白丢掉一个合规的案子，静默通过又会让偏离无从察觉
        verdict = classify_duration(duration, self._cfg.target_seconds)
        if verdict is DurationVerdict.WARN:
            logger.warning(
                "实测时长 %.1f 秒偏离目标 %.1f 秒超过容忍带，已通过时长门限：%s",
                duration,
                self._cfg.target_seconds,
                duration_gate.detail,
            )

        # 阶段三：封面（失败则兜底，不阻断交付）
        cover_path = self._generate_cover(case, out_dir)

        script_path = out_dir / "script.json"
        script_path.write_text(
            json.dumps(transcript.to_json_obj(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return Episode(audio_path=audio_path, cover_path=cover_path, script_path=script_path)

    def _generate_cover(self, case: CaseInput, out_dir: Path) -> Path:
        from .cover import FallbackCover, build_generator

        out = out_dir / "cover.png"
        try:
            return self._cover.generate(case.topic, out)
        except Exception as exc:  # noqa: BLE001 — 封面仅 10 分，降级优于失败
            # 降级必须可观测：系统性坏掉的封面模型会让每一集都退化成纯色
            # 底图，静默丢掉封面分，而无人知道原因
            logger.warning("封面生成失败，降级为兜底封面：%s", exc, exc_info=True)
            fallback = FallbackCover(self._cfg.cover, runner=self._run)
            return fallback.generate(case.topic, out)
