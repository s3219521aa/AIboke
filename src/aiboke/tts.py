"""MOSS-TTSD 语音合成适配器。

两个后端：
  - llamacpp（默认）：走 MOSS-TTS 的 llama.cpp 原生路径，权重为 GGUF，
    显存约 9GB，可与 LLM 同时常驻
  - transformers（备选）：走官方 inference.py，权重 bf16 约 19GB，
    与 LLM 无法同时常驻，但无需编译 fork

时长控制依赖 MOSS-TTSD 的官方换算：1 秒音频 ≈ 12.5 tokens，
因此 max_new_tokens = 目标秒数 * 12.5（三重保险中的第二重）。

音色性别（另一条 0 分门限）由参考音频条件保证：脚本里的 [S1]/[S2] 只标明
轮次归属、不含性别，所以 llama.cpp 后端必须拿到参考音频，拿不到就报错，
绝不静默合成出两个与 speaker_genderN 无关的音色。而 llama-moss-tts 只接受
**一个** --reference-audio，因此两位主播的参考音频会被拼接成一个临时文件
（S1 → S2，与提示文本 [S1]…[S2]… 的分段顺序一致）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence

from . import audio_utils
from .config import TtsConfig
from .length import seconds_to_max_tokens
from .schema import Turn, VoicePair


class TtsError(RuntimeError):
    """语音合成失败。"""


def format_tagged_script(turns: Sequence[Turn]) -> str:
    """拼接为 MOSS-TTSD 的 [S1]/[S2] 标签脚本。"""
    if not turns:
        raise TtsError("没有任何对话轮次可供合成")
    return "".join(f"[S{t.speaker}]{t.text}" for t in turns)


def format_input_jsonl(turns: Sequence[Turn], voices: VoicePair) -> str:
    """构造 transformers 后端的 JSONL 输入（单行）。"""
    if not turns:
        raise TtsError("没有任何对话轮次可供合成")
    if voices.speaker1.gender != "男" and voices.speaker1.gender != "女":
        raise TtsError(f"speaker1 性别非法：{voices.speaker1.gender}")
    if voices.speaker2.gender != "男" and voices.speaker2.gender != "女":
        raise TtsError(f"speaker2 性别非法：{voices.speaker2.gender}")

    payload = {
        "text": format_tagged_script(turns),
        "prompt_audio_speaker1": voices.speaker1.reference_audio or "",
        "prompt_text_speaker1": f"[S1]{voices.speaker1.description}",
        "prompt_audio_speaker2": voices.speaker2.reference_audio or "",
        "prompt_text_speaker2": f"[S2]{voices.speaker2.description}",
        "voice_description_speaker1": voices.speaker1.description,
        "voice_description_speaker2": voices.speaker2.description,
    }
    return json.dumps(payload, ensure_ascii=False)


class TtsBackend(Protocol):
    def synthesize(
        self,
        turns: Sequence[Turn],
        voices: VoicePair,
        out_path: Path,
        target_seconds: float,
    ) -> Path: ...


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _run_or_raise(run: Callable, cmd: list[str], label: str):
    """执行子进程，并把 OSError 归一为 TtsError。

    二进制/解释器不在 PATH 上时 subprocess 抛 FileNotFoundError（OSError 子类），
    它不是 TtsError，会穿透调用方的 `except TtsError` 重试与兜底逻辑。
    """
    try:
        return run(cmd)
    except OSError as exc:
        raise TtsError(f"无法启动{label}进程（{cmd[0]}）：{exc}") from exc


def _require_reference_audio(voices: VoicePair) -> None:
    """llama.cpp 后端的音色性别只能靠参考音频控制，缺了就报错。

    该后端的文本只有 [S1]/[S2] 轮次标签，标签不含性别——没有参考音频时，
    合成出的两个音色与请求的 speaker_gender1/2 无关，会直接踩中
    「性别必须符合要求」这条 0 分门限。宁可失败并给出两条出路。
    """
    missing = [
        name
        for name, preset in (("speaker1", voices.speaker1), ("speaker2", voices.speaker2))
        if not preset.reference_audio
    ]
    if missing:
        raise TtsError(
            f"llama.cpp 后端缺少音色条件：{'、'.join(missing)} 没有 reference_audio，"
            "无法保证音色性别符合 speaker_gender（0 分门限）。两条出路："
            "① 用 MOSS-VoiceGenerator 依据 description 预生成音色 wav，"
            "把路径填进 configs/voice_presets.json 的 reference_audio 字段；"
            "② 把 tts.backend 改为 transformers——它会把音色描述传给模型。"
        )


def _combined_reference_text(voices: VoicePair) -> str:
    """两段参考音频对应的提示文本：[S1]描述[S2]描述。

    顺序与拼接顺序（S1 → S2）一致，即 volcengine 写法的
    `prompt_audio: combined_speakers.wav` + `prompt_text: "[S1]…[S2]…"`。
    """
    return f"[S1]{voices.speaker1.description}[S2]{voices.speaker2.description}"


def _claim_output(out_path: Path, before: set[Path]) -> Path:
    """认领 transformers 后端的产物。

    官方 inference.py 不接受输出文件名，只在 --save_dir 下按自己的规则命名，
    所以父进程不能假设 out_path 一定存在。这里把「本次调用新产生」的 wav 移到
    out_path；只认运行前不存在的文件，避免某次静默失败把上一幕的产物搬过来
    冒充本幕结果——那比直接报错更糟。
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    produced = [
        p
        for p in out_path.parent.glob("*.wav")
        if p.resolve() not in before and p.stat().st_size > 0
    ]
    if not produced:
        raise TtsError(f"transformers 语音合成未产出音频文件：{out_path}")

    newest = max(produced, key=lambda p: p.stat().st_mtime)
    shutil.move(str(newest), str(out_path))
    return out_path


class LlamaCppTts:
    """MOSS-TTS 的 llama.cpp 原生路径（默认后端）——OpenMOSS fork 的 llama-moss-tts。

    命令行形状取自 fork 的首方文档与源码（`docs/moss-tts-firstclass-e2e_zh.md`
    与 `tools/tts/run-moss-tts-delay.cpp`）：`-m` 是 backbone GGUF **文件**，
    `--audio-encoder-model` / `--audio-decoder-model` 是两个附加 GGUF，
    `--text` 是 [S1]/[S2] 标签稿，`--reference-audio` 只接受**一个** wav，
    `--wav-out` 是输出文件。

    该二进制**没有** `--output`、`--temperature`、`--top-p`、`--top-k`、
    `--repetition-penalty`、`--reference-text`、`--text-normalize` 这些标志
    （它们属于官方 inference.py，属 transformers 后端）——给 CLI 发未知标志是
    失败而不是提示，所以一律不发。`-ngl` 同样不必发：该二进制的默认值就是
    -1（全部层上 GPU）。标志名全部可配置，上机核对后若要改拼写不必动代码。
    """

    def __init__(
        self,
        cfg: TtsConfig,
        runner: Callable | None = None,
        concat_runner: Callable | None = None,
    ) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner
        # 拼接参考音频用的 ffmpeg runner；None 时用 audio_utils 的默认实现
        self._concat_run = concat_runner

    def _build_cmd(
        self,
        voices: VoicePair,
        turns: Sequence[Turn],
        out_path: Path,
        target_seconds: float,
        reference: Path,
    ) -> list[str]:
        cfg = self._cfg
        cmd = [cfg.binary, cfg.model_flag, cfg.model_path]
        if cfg.audio_encoder_model:
            cmd += [cfg.audio_encoder_flag, cfg.audio_encoder_model]
        if cfg.audio_decoder_model:
            cmd += [cfg.audio_decoder_flag, cfg.audio_decoder_model]
        cmd += [
            "--text", format_tagged_script(turns),
            cfg.reference_audio_flag, str(reference),
            cfg.output_flag, str(out_path),
            "--max-new-tokens", str(seconds_to_max_tokens(target_seconds)),
        ]
        if cfg.reference_text_flag:
            # 该二进制默认没有这个标志；fork 变体确有时才由配置启用
            cmd += [cfg.reference_text_flag, _combined_reference_text(voices)]
        return cmd

    def synthesize(self, turns, voices, out_path: Path, target_seconds: float) -> Path:
        if not self._cfg.binary:
            raise TtsError("tts.binary 未配置，无法调用 llama.cpp 后端")
        if not self._cfg.model_path:
            raise TtsError("tts.model_path 未配置，找不到 MOSS-TTSD 权重")
        _require_reference_audio(voices)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 该二进制只接受一个 --reference-audio：把两位主播的参考音频按 S1→S2
        # 的顺序拼成一个临时文件（与提示文本 [S1]…[S2]… 的分段顺序一致），
        # 用完随临时目录一起删除。
        with tempfile.TemporaryDirectory(prefix="aiboke_tts_ref_") as tmp:
            try:
                reference = audio_utils.concat_wavs(
                    [
                        Path(voices.speaker1.reference_audio),
                        Path(voices.speaker2.reference_audio),
                    ],
                    Path(tmp) / "reference.wav",
                    gap_ms=200,
                    runner=self._concat_run,
                )
            except audio_utils.AudioError as exc:
                raise TtsError(f"拼接两位主播的参考音频失败：{exc}") from exc

            cmd = self._build_cmd(voices, turns, out_path, target_seconds, reference)
            proc = _run_or_raise(self._run, cmd, "llama.cpp 语音合成")
            if proc.returncode != 0:
                raise TtsError(
                    f"llama.cpp 语音合成失败（退出码 {proc.returncode}）：{proc.stderr.strip()[:500]}"
                )

        if not out_path.exists() or out_path.stat().st_size == 0:
            # 静默失败：进程退出码为 0 但没有产出文件
            raise TtsError(f"llama.cpp 语音合成未产出音频文件：{out_path}")
        return out_path


class TransformersTts:
    """官方 inference.py 后端（备选）。"""

    def __init__(self, cfg: TtsConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def synthesize(self, turns, voices, out_path: Path, target_seconds: float) -> Path:
        if not self._cfg.model_path:
            raise TtsError("tts.model_path 未配置，找不到 MOSS-TTSD 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(format_input_jsonl(turns, voices))
            jsonl_path = fh.name

        # 解释器用当前进程的解释器，脚本用绝对路径：两者都不该依赖 PATH 和 cwd
        script = Path(self._cfg.inference_script).expanduser().resolve()
        before = {p.resolve() for p in out_path.parent.glob("*.wav")}

        cmd = [
            sys.executable, str(script),
            "--model_path", self._cfg.model_path,
            "--input_jsonl", jsonl_path,
            "--save_dir", str(out_path.parent),
            "--mode", "voice_clone_and_continuation",
            "--max_new_tokens", str(seconds_to_max_tokens(target_seconds)),
            "--temperature", str(self._cfg.temperature),
            "--top_p", str(self._cfg.top_p),
            "--top_k", str(self._cfg.top_k),
            "--repetition_penalty", str(self._cfg.repetition_penalty),
            "--text_normalize",
            "--sample_rate_normalize",
        ]

        proc = _run_or_raise(self._run, cmd, "transformers 语音合成")
        if proc.returncode != 0:
            raise TtsError(
                f"transformers 语音合成失败（退出码 {proc.returncode}）：{proc.stderr.strip()[:500]}"
            )
        return _claim_output(out_path, before)


def build_backend(cfg: TtsConfig, runner: Callable | None = None) -> TtsBackend:
    if cfg.backend == "llamacpp":
        return LlamaCppTts(cfg, runner=runner)
    if cfg.backend == "transformers":
        return TransformersTts(cfg, runner=runner)
    raise TtsError(f"未知的 tts.backend：{cfg.backend!r}")
