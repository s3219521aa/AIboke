"""轻量 HTTP 封装。

评测方的容器调用方式未知，因此除 CLI 外额外提供常驻服务模式，
两种入口共用同一套 Pipeline 装配逻辑。

异常契约与 CLI 一致：流水线失败（PipelineError）与封面失败（CoverError，
主封面与兜底封面都挂了）都返回 500 并在 detail 里给出原因，不新造状态码；
仅请求体本身不合法时才用 422。

FastAPI 只在 create_app 内部导入：本模块属于可选依赖（pyproject 的 server
extra），未安装时不影响 CLI 与流水线。

本模块**不能**使用 `from __future__ import annotations`：FastAPI 在模块全局
命名空间里解析端点注解，而 GenerateRequest 定义在 create_app 内部；注解一旦
被字符串化就成了无法解析的 ForwardRef，FastAPI 会把 req 当成查询参数，于是
带着 JSON 体的 POST 恒返回 422（已实测）。返回注解因此写成字符串常量。
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .bootstrap import build_pipeline
from .config import load_config
from .cover import CoverError
from .schema import CaseInput

if TYPE_CHECKING:  # 仅在类型检查时可见，运行时不引入可选依赖
    from fastapi import FastAPI


def create_app(cfg_path: Path) -> "FastAPI":
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class GenerateRequest(BaseModel):
        topic: str
        speaker_gender1: str
        speaker_gender2: str

    cfg = load_config(Path(cfg_path))
    pipeline = build_pipeline(cfg, Path(cfg.models_root))
    app = FastAPI(title="AI 中文播客生成")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        try:
            case = CaseInput(
                topic=req.topic,
                speaker_gender1=req.speaker_gender1,
                speaker_gender2=req.speaker_gender2,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        out_dir = Path(tempfile.mkdtemp(prefix="aiboke_"))
        try:
            episode = pipeline.run(case, out_dir)
        except CoverError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"封面生成失败（主封面与兜底封面均失败）：{exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — 统一 500，细节进 detail
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "audio_path": str(episode.audio_path),
            "cover_path": str(episode.cover_path),
            "script_path": str(episode.script_path),
        }

    return app
