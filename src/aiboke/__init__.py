"""AI 中文双人播客生成系统。"""

from .config import Config, load_config
from .pipeline import Pipeline, PipelineError
from .schema import CaseInput, Episode, Transcript, Turn, VoicePair, VoicePreset

__all__ = [
    "CaseInput",
    "Config",
    "Episode",
    "Pipeline",
    "PipelineError",
    "Transcript",
    "Turn",
    "VoicePair",
    "VoicePreset",
    "load_config",
]

__version__ = "0.1.0"
