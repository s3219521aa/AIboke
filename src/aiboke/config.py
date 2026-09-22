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
    temperature: float = 1.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


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


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where} 缺少必需字段 {key!r}")
    return d[key]


def load_config(path: Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    target_seconds = float(_require(raw, "target_seconds", "config"))
    if not (MIN_SECONDS <= target_seconds <= MAX_SECONDS):
        raise ValueError(
            f"target_seconds 必须落在 [{MIN_SECONDS}, {MAX_SECONDS}] 内，"
            f"否则必然违反时长门限，实际为 {target_seconds}"
        )

    llm_raw = _require(raw, "llm", "config")
    tts_raw = _require(raw, "tts", "config")
    cover_raw = raw.get("cover", {})

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
            temperature=float(tts_raw.get("temperature", 1.1)),
            top_p=float(tts_raw.get("top_p", 0.9)),
            top_k=int(tts_raw.get("top_k", 50)),
            repetition_penalty=float(tts_raw.get("repetition_penalty", 1.1)),
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
