"""三阶段流水线编排。

阶段一 文稿生成 -> 阶段二 语音合成 -> 阶段三 封面生成，全程受四道
0 分门限约束。封面失败不导致整案失败（仅 10 分，且产物需存在）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import audio_utils
from .config import Config
from .gates import all_passed, check_chinese, check_duration, check_genders, check_two_speakers
from .length import DurationVerdict, classify_duration
from .schema import CaseInput, Episode, Transcript, Turn, VoicePreset
from .voices import resolve_pair

logger = logging.getLogger(__name__)


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
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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

                # 逐幕先查语言门限：命中「非中文则 0 分」的风险要在付出
                # 昂贵的语音合成代价之前就拦下来
                act_text = "".join(t.text for t in act.turns)
                lang_gate = check_chinese(act_text, self._cfg.chinese_min_ratio)
                if not lang_gate.passed:
                    raise PipelineError(
                        f"第 {act.act_index + 1} 幕未通过语言门限：{lang_gate.detail}"
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

        # 拼接 -> 响度归一化 -> 转 mp3
        concat_path = audio_utils.concat_wavs(
            audio_wavs, out_dir / "podcast_concat.wav", runner=self._run
        )
        norm_path = audio_utils.normalize_loudness(
            concat_path, out_dir / "podcast_norm.wav",
            target_lufs=self._cfg.target_lufs, runner=self._run,
        )
        audio_path = audio_utils.to_mp3(
            norm_path, out_dir / "podcast.mp3", runner=self._run
        )

        # 时长门限：越界即 0 分，必须失败而非静默通过
        duration = audio_utils.probe_duration(audio_path, runner=self._run)
        duration_gate = check_duration(duration)
        if not duration_gate.passed:
            raise PipelineError(
                f"时长未通过门限校验：{duration_gate.detail}。"
                f"建议：{duration_gate.retry_hint or '调整目标字数后重试'}"
            )

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
        except Exception:  # noqa: BLE001 — 封面仅 10 分，降级优于失败
            fallback = FallbackCover(self._cfg.cover, runner=self._run)
            return fallback.generate(case.topic, out)
