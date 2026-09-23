"""封面生成。1024x1024 PNG，与播客主题相关。

封面仅占 10 分，但产物缺失可能被判定为输出不达标而拉高失败率
（基线失败率 <= 10%）。因此提供 FallbackCover：即使图像模型不可用，
也用纯色底图保证产物存在。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Protocol

from .config import CoverConfig
from .prompts import build_cover_prompt


class CoverError(RuntimeError):
    """封面生成失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _require_output(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise CoverError(f"封面生成未产出文件：{path}")
    return path


class CoverGenerator(Protocol):
    def generate(self, topic: str, out_path: Path) -> Path: ...


class ZImageCover:
    """Z-Image-Turbo：6B / 8 步 / 1024x1024 / Apache-2.0。"""

    def __init__(self, cfg: CoverConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def generate(self, topic: str, out_path: Path) -> Path:
        if not self._cfg.model_path:
            raise CoverError("cover.model_path 未配置，找不到 Z-Image-Turbo 权重")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 提示词由 prompts 模块统一构造：明确禁止画面出现文字，
        # 因为文生图模型的文字渲染不可靠，乱码会严重拉低观感
        prompt = build_cover_prompt(topic)

        cmd = [
            "python", "-m", "aiboke.cover_runner",
            "--model", self._cfg.model_path,
            "--prompt", prompt,
            "--size", str(self._cfg.size),
            "--steps", str(self._cfg.steps),
            "--output", str(out_path),
        ]
        proc = self._run(cmd)
        if proc.returncode != 0:
            raise CoverError(f"Z-Image 封面生成失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)


class FallbackCover:
    """纯色底图兜底，保证产物存在，不因 10 分项拉高失败率。"""

    def __init__(self, cfg: CoverConfig, runner: Callable | None = None) -> None:
        self._cfg = cfg
        self._run = runner or _default_runner

    def generate(self, topic: str, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        size = self._cfg.size

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x1F3A93:s={size}x{size}",
            "-frames:v", "1",
            str(out_path),
        ]
        proc = self._run(cmd)
        if proc.returncode != 0:
            raise CoverError(f"生成兜底封面失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)


def build_generator(cfg: CoverConfig, runner: Callable | None = None, prefer_fallback: bool = False) -> CoverGenerator:
    if prefer_fallback:
        return FallbackCover(cfg, runner=runner)
    return ZImageCover(cfg, runner=runner)
