"""MOSS-TTSD 语音合成适配器。

两个后端：
  - llamacpp（默认）：走 MOSS-TTS 的 llama.cpp 原生路径，权重为 GGUF，
    显存约 9GB，可与 LLM 同时常驻
  - transformers（备选）：走官方 inference.py，权重 bf16 约 19GB，
    与 LLM 无法同时常驻，但无需编译 fork

时长控制依赖 MOSS-TTSD 的官方换算：1 秒音频 ≈ 12.5 tokens，
因此 max_new_tokens = 目标秒数 * 12.5（三重保险中的第二重）。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .config import TtsConfig
from .length import seconds_to_max_tokens
from .schema import Turn, VoicePair

# transformers 后端的官方推理脚本名（位于 MOSS-TTSD 仓库内）
_INFERENCE_SCRIPT = "inference.py"


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


class LlamaCppTts:
    """MOSS-TTS 的 llama.cpp 原生路径（默认后端）。"""

    def __init__(self, cfg: TtsConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def synthesize(self, turns, voices, out_path: Path, target_seconds: float) -> Path:
        if not self._cfg.binary:
            raise TtsError("tts.binary 未配置，无法调用 llama.cpp 后端")
        if not self._cfg.model_path:
            raise TtsError("tts.model_path 未配置，找不到 MOSS-TTSD 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._cfg.binary,
            "--model", self._cfg.model_path,
            "--text", format_tagged_script(turns),
            "--output", str(out_path),
            "--max-new-tokens", str(seconds_to_max_tokens(target_seconds)),
            "--temperature", str(self._cfg.temperature),
            "--top-p", str(self._cfg.top_p),
            "--top-k", str(self._cfg.top_k),
            "--repetition-penalty", str(self._cfg.repetition_penalty),
            "--text-normalize",
            "--sample-rate-normalize",
        ]

        proc = self._run(cmd)
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

        cmd = [
            "python", _INFERENCE_SCRIPT,
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

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise TtsError(
                f"transformers 语音合成失败（退出码 {proc.returncode}）：{proc.stderr.strip()[:500]}"
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise TtsError(f"transformers 语音合成未产出音频文件：{out_path}")
        return out_path


def build_backend(cfg: TtsConfig, runner: Callable | None = None) -> TtsBackend:
    if cfg.backend == "llamacpp":
        return LlamaCppTts(cfg, runner=runner)
    if cfg.backend == "transformers":
        return TransformersTts(cfg, runner=runner)
    raise TtsError(f"未知的 tts.backend：{cfg.backend!r}")
