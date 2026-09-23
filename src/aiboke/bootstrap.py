"""依赖装配：把配置组装成可运行的 Pipeline。

把「如何构造各组件」从 CLI 与 HTTP 入口中抽出来，使两个入口共享
同一套装配逻辑，也便于在测试中替换假实现。
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .cover import build_generator
from .llm_client import LlmClient
from .pipeline import Pipeline
from .script_writer import ScriptWriter
from .tts import build_backend
from .voices import load_presets


def load_presets_from_config(path: Path):
    return load_presets(Path(path))


def build_pipeline(
    cfg: Config,
    models_root: Path,
    prefer_fallback_cover: bool = False,
) -> Pipeline:
    # 模型挂载点只做归一化：各组件从 cfg 里读自己的具体路径（tts.model_path、
    # cover.model_path），这里保留该参数供两个入口统一覆盖配置里的根目录
    models_root = Path(models_root)

    llm = LlmClient(cfg.llm)
    writer = ScriptWriter(
        llm,
        target_seconds=cfg.target_seconds,
        chars_per_minute=cfg.chars_per_minute,
        max_retries=cfg.max_retries,
        factcheck=cfg.factcheck,
        # 两道 0 分门限（语言门、双人门）的阈值同样来自配置。ScriptWriter 的
        # 同名参数默认值恰好等于 gates 里的常量，漏传不会改变默认行为，却会
        # 让 configs/default.yaml 里的修改静默失效——必须显式转发。
        chinese_min_ratio=cfg.chinese_min_ratio,
        speaker_min_share=cfg.speaker_min_share,
    )
    tts = build_backend(cfg.tts)
    cover = build_generator(cfg.cover, prefer_fallback=prefer_fallback_cover)

    # 预设随代码走，按模块位置解析绝对路径：容器的 cwd 不保证是仓库根目录
    preset_path = Path(__file__).resolve().parents[2] / "configs" / "voice_presets.json"
    presets = load_presets(preset_path)

    return Pipeline(cfg, writer, tts, cover, presets)
