"""全部数据结构定义。纯数据，不含业务逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VALID_GENDERS = frozenset({"男", "女"})


@dataclass(frozen=True)
class CaseInput:
    topic: str
    speaker_gender1: str
    speaker_gender2: str

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("topic 不能为空")
        for field, value in (
            ("speaker_gender1", self.speaker_gender1),
            ("speaker_gender2", self.speaker_gender2),
        ):
            if value not in VALID_GENDERS:
                raise ValueError(f"{field} 必须是 男 或 女，实际为 {value!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "CaseInput":
        try:
            topic = d["topic"]
            g1 = d["speaker_gender1"]
            g2 = d["speaker_gender2"]
        except KeyError as exc:
            raise ValueError(f"缺少必需字段: {exc.args[0]}") from exc
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic 必须是非空字符串")
        return cls(topic=topic.strip(), speaker_gender1=g1, speaker_gender2=g2)


@dataclass(frozen=True)
class Turn:
    speaker: int
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in (1, 2):
            raise ValueError(f"speaker 必须是 1 或 2，实际为 {self.speaker!r}")


@dataclass(frozen=True)
class Transcript:
    title: str
    turns: tuple[Turn, ...]

    def to_json_obj(self) -> dict:
        return {
            "title": self.title,
            "content": [{"speaker": t.speaker, "text": t.text} for t in self.turns],
        }

    @property
    def full_text(self) -> str:
        return "".join(t.text for t in self.turns)


@dataclass(frozen=True)
class VoicePreset:
    id: str
    gender: str
    description: str
    reference_audio: str | None = None

    def __post_init__(self) -> None:
        if self.gender not in VALID_GENDERS:
            raise ValueError(f"VoicePreset.gender 必须是 男 或 女，实际为 {self.gender!r}")


@dataclass(frozen=True)
class VoicePair:
    speaker1: VoicePreset
    speaker2: VoicePreset


@dataclass(frozen=True)
class Episode:
    audio_path: Path
    cover_path: Path
    script_path: Path
