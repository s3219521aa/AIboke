"""依赖装配：把配置组装成可运行的 Pipeline。

把「如何构造各组件」从 CLI 与 HTTP 入口中抽出来，使两个入口共享
同一套装配逻辑，也便于在测试中替换假实现。
"""

from __future__ import annotations

from dataclasses import replace
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


def _resolve_under(root: Path, value: str | None) -> str | None:
    """相对路径按 models_root 解析；已锚定的路径原样返回。

    绝对性判定用 `is_absolute() or anchor`：/models/X 在 POSIX 上是绝对路径，
    在 Windows 上是「有根无盘符」路径——后者的 is_absolute() 为假，但同样不该
    被拼到 models_root 下（拼接会把它改写成 <盘符>:\\models\\X，把配置里写死的
    挂载点改坏）。两者合起来正是 pathlib 的「已锚定」语义，等价于 is_anchored()
    ——该 API 在 Python 3.11/3.12（本项目最低支持版本与开发机）上并不存在，
    所以这里用 anchor 属性判定。

    已锚定的路径原样返回，而不是 str(Path(value))：Windows 上 str() 会顺手把
    /models/X 改成 \\models\\X，虽指向同一位置，配置回显与报错信息却会变味。
    只有 ~ 展开才真的需要改写写法。
    """
    if not value:
        return value
    p = Path(value).expanduser()
    if p.is_absolute() or p.anchor:
        return str(p) if value.startswith("~") else value
    return str(root / p)


def build_pipeline(
    cfg: Config,
    models_root: Path,
    prefer_fallback_cover: bool = False,
) -> Pipeline:
    # 模型挂载点在这里真正生效：配置里写相对路径时按 models_root 解析，
    # 写绝对路径时原样使用（configs/default.yaml 全为 /models/... 绝对路径，
    # 默认行为不变）。模型以挂载方式提供，操作员正是靠这个设置把系统指向
    # 挂载点——只传参不解析，这个开关就是空的。
    #
    # tts.binary 与 tts.inference_script 故意不参与解析：它们是可执行文件/
    # 脚本，不是挂载进来的模型数据，这个不对称是有意的。
    models_root = Path(models_root)
    cfg = replace(
        cfg,
        tts=replace(
            cfg.tts, model_path=_resolve_under(models_root, cfg.tts.model_path)
        ),
        cover=replace(
            cfg.cover, model_path=_resolve_under(models_root, cfg.cover.model_path)
        ),
    )

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
