"""配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .length import MAX_SECONDS, MIN_SECONDS

VALID_BACKENDS = frozenset({"llamacpp", "transformers"})


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    timeout: float = 300.0
    temperature: float = 0.7
    top_p: float = 0.80
    presence_penalty: float = 1.5


@dataclass(frozen=True)
class TtsConfig:
    backend: str
    binary: str | None = None
    model_path: str | None = None
    # native 路径（OpenMOSS fork 的 llama-moss-tts）另需两个转换出来的 GGUF：
    # 没有 decoder 就出不了 wav；encoder 只在给了参考音频时才需要。因此
    # decoder 缺失会在起进程前直接报错（require_audio_decoder 可关掉这道校验，
    # 供自带 decoder 或改用别的参数的 fork 变体使用）。
    audio_encoder_model: str | None = None
    audio_decoder_model: str | None = None
    require_audio_decoder: bool = True
    # transformers 后端用：官方 inference.py 的 --codec_model_path（音频编解码器，
    # 与 --model_path 是两个仓库）。未配置时不发送，由脚本自己的默认值决定。
    codec_model_path: str | None = None
    codec_model_flag: str = "--codec_model_path"
    # 采样参数只对 transformers 后端生效：llama-moss-tts 没有 --temperature/
    # --top-p/--top-k/--repetition-penalty，只有 --text-temperature 这类分通道
    # 标志，通道映射无法离线核实——宁可不发，也不给 CLI 发未知标志。
    temperature: float = 1.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    # 以下标志名全部可配置：上游 fork 的确切拼写以上机 `--help` 为准，不一致
    # 时运维在 config.yaml 里改，不必动代码。
    model_flag: str = "-m"
    output_flag: str = "--wav-out"
    audio_encoder_flag: str = "--audio-encoder-model"
    audio_decoder_flag: str = "--audio-decoder-model"
    reference_audio_flag: str = "--reference-audio"
    # llama-moss-tts **没有** reference-text 标志（源码 tools/tts/
    # run-moss-tts-delay.cpp 的参数表里没有它），故默认为空 = 不发送。若某个
    # fork 变体确有该标志，填上标志名即可；发送的值是 [S1]描述[S2]描述。
    reference_text_flag: str = ""
    # transformers 后端用：官方 inference.py 的路径（按 cwd 解析，故部署时给绝对路径）
    inference_script: str = "inference.py"


@dataclass(frozen=True)
class CoverConfig:
    steps: int = 8
    size: int = 1024
    model_path: str | None = None


@dataclass(frozen=True)
class Config:
    target_seconds: float
    chars_per_minute: float
    models_root: str
    llm: LlmConfig
    tts: TtsConfig
    cover: CoverConfig
    max_retries: int = 2
    chinese_min_ratio: float = 0.85
    speaker_min_share: float = 0.25
    target_lufs: float = -16.0
    factcheck: bool = True


def _mapping(value, where: str) -> dict:
    """把一段配置当映射用；「存在但为空」（如 `tts:` 后无内容）算配置错误。

    YAML 里 `tts:` 这种写法解析出来是 None，直接取键会抛 TypeError——它不是
    调用方（CLI/HTTP）捕获的异常类型，于是操作员看到的是一段 traceback，
    而不是「配置错误：tts 必须是配置映射」。这里统一归一为 ValueError。
    """
    if value is None:
        raise ValueError(f"{where} 不能为空：该段必须给出字段（检查 YAML 缩进与内容）")
    if not isinstance(value, dict):
        raise ValueError(f"{where} 必须是配置映射，实际为 {type(value).__name__}")
    return value


def _require(d: dict, key: str, where: str):
    if key not in _mapping(d, where):
        raise ValueError(f"{where} 缺少必需字段 {key!r}")
    return d[key]


def _optional_mapping(d: dict, key: str, where: str) -> dict:
    """可选配置段：缺失按空映射处理，「存在但为空」与类型写错都报错。"""
    if key not in d or d[key] is None:
        if key in d:
            raise ValueError(f"{where} 不能为空：要么整段删掉，要么给出字段")
        return {}
    return _mapping(d[key], where)


def load_config(path: Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"配置文件为空: {path}")
    raw = _mapping(raw, "config")

    target_seconds = float(_require(raw, "target_seconds", "config"))
    if not (MIN_SECONDS <= target_seconds <= MAX_SECONDS):
        raise ValueError(
            f"target_seconds 必须落在 [{MIN_SECONDS}, {MAX_SECONDS}] 内，"
            f"否则必然违反时长门限，实际为 {target_seconds}"
        )

    llm_raw = _require(raw, "llm", "config")
    tts_raw = _require(raw, "tts", "config")
    cover_raw = _optional_mapping(raw, "cover", "cover")

    backend = _require(tts_raw, "backend", "tts")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"tts.backend 必须是 {sorted(VALID_BACKENDS)} 之一，实际为 {backend!r}"
        )

    return Config(
        target_seconds=target_seconds,
        chars_per_minute=float(_require(raw, "chars_per_minute", "config")),
        models_root=str(raw.get("models_root", "/models")),
        llm=LlmConfig(
            base_url=_require(llm_raw, "base_url", "llm"),
            model=_require(llm_raw, "model", "llm"),
            timeout=float(llm_raw.get("timeout", 300.0)),
            temperature=float(llm_raw.get("temperature", 0.7)),
            top_p=float(llm_raw.get("top_p", 0.80)),
            presence_penalty=float(llm_raw.get("presence_penalty", 1.5)),
        ),
        tts=TtsConfig(
            backend=backend,
            binary=tts_raw.get("binary"),
            model_path=tts_raw.get("model_path"),
            audio_encoder_model=tts_raw.get("audio_encoder_model"),
            audio_decoder_model=tts_raw.get("audio_decoder_model"),
            require_audio_decoder=bool(tts_raw.get("require_audio_decoder", True)),
            codec_model_path=tts_raw.get("codec_model_path"),
            codec_model_flag=str(
                tts_raw.get("codec_model_flag", "--codec_model_path")
            ),
            temperature=float(tts_raw.get("temperature", 1.1)),
            top_p=float(tts_raw.get("top_p", 0.9)),
            top_k=int(tts_raw.get("top_k", 50)),
            repetition_penalty=float(tts_raw.get("repetition_penalty", 1.1)),
            model_flag=str(tts_raw.get("model_flag", "-m")),
            output_flag=str(tts_raw.get("output_flag", "--wav-out")),
            audio_encoder_flag=str(
                tts_raw.get("audio_encoder_flag", "--audio-encoder-model")
            ),
            audio_decoder_flag=str(
                tts_raw.get("audio_decoder_flag", "--audio-decoder-model")
            ),
            reference_audio_flag=str(
                tts_raw.get("reference_audio_flag", "--reference-audio")
            ),
            # 默认空串：llama-moss-tts 没有这个标志，不发送（见 dataclass 注释）
            reference_text_flag=str(tts_raw.get("reference_text_flag", "")),
            inference_script=str(tts_raw.get("inference_script", "inference.py")),
        ),
        cover=CoverConfig(
            steps=int(cover_raw.get("steps", 8)),
            size=int(cover_raw.get("size", 1024)),
            model_path=cover_raw.get("model_path"),
        ),
        max_retries=int(raw.get("max_retries", 2)),
        chinese_min_ratio=float(raw.get("chinese_min_ratio", 0.85)),
        speaker_min_share=float(raw.get("speaker_min_share", 0.25)),
        target_lufs=float(raw.get("target_lufs", -16.0)),
        factcheck=bool(raw.get("factcheck", True)),
    )
