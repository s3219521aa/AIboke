"""性别到音色预设的解析。

性别门限由构造保证——按请求的性别挑选音色，而非事后检测音频。
同性别组合（男男 / 女女）时必须选中不同预设，否则违反
「音色应有区分度」的评分要求。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from .schema import VALID_GENDERS, VoicePair, VoicePreset


def load_presets(path: Path) -> dict[str, list[VoicePreset]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音色预设文件不存在: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("presets")
    if not entries:
        raise ValueError(f"presets 字段缺失或为空: {path}")

    grouped: dict[str, list[VoicePreset]] = {g: [] for g in VALID_GENDERS}
    for entry in entries:
        try:
            preset = VoicePreset(
                id=entry["id"],
                gender=entry["gender"],
                description=entry["description"],
                reference_audio=entry.get("reference_audio"),
            )
        except KeyError as exc:
            raise ValueError(f"音色预设缺少字段 {exc.args[0]!r}: {entry}") from exc
        grouped[preset.gender].append(preset)

    return {g: v for g, v in grouped.items() if v}


def resolve_pair(
    presets: Mapping[str, Sequence[VoicePreset]],
    gender1: str,
    gender2: str,
    seed: int | None = None,
) -> VoicePair:
    for label, gender in (("speaker_gender1", gender1), ("speaker_gender2", gender2)):
        if not presets.get(gender):
            raise ValueError(f"{label}={gender} 没有可用的音色预设")

    rng = random.Random(seed)
    first = rng.choice(list(presets[gender1]))

    if gender1 == gender2:
        candidates = [p for p in presets[gender2] if p.id != first.id]
        if not candidates:
            raise ValueError(
                f"两位主播均为 {gender2}，但该性别只有一条音色预设，"
                "需要至少两条才能保证音色有区分度"
            )
        second = rng.choice(candidates)
    else:
        second = rng.choice(list(presets[gender2]))

    return VoicePair(speaker1=first, speaker2=second)
