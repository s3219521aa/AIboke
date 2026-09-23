"""封面生成。1024x1024 PNG，与播客主题相关。

封面仅占 10 分，但产物缺失可能被判定为输出不达标而拉高失败率
（基线失败率 <= 10%）。因此提供 FallbackCover：即使图像模型不可用，
也用纯色底图保证产物存在。
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable, Protocol

from .config import CoverConfig
from .prompts import build_cover_image_prompt


class CoverError(RuntimeError):
    """封面生成失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _run_or_raise(run: Callable, cmd: list[str], label: str):
    """执行子进程，并把 OSError 归一为 CoverError。

    解释器不在 PATH 上时 subprocess 抛 FileNotFoundError（OSError 子类），
    它不是 CoverError，会穿透编排层的 `except CoverError`——那会把兜底封面
    也一起跳过，让本该存在的产物彻底消失。
    """
    try:
        return run(cmd)
    except OSError as exc:
        raise CoverError(f"无法启动{label}进程（{cmd[0]}）：{exc}") from exc


def _require_output(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise CoverError(f"封面生成未产出文件：{path}")
    return path


def _require_png_size(path: Path, size: int) -> Path:
    """按 PNG 里的真实像素校验尺寸——IHDR 的宽高位于字节 16-24。"""
    head = path.read_bytes()[:24]
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        raise CoverError(f"封面不是合法 PNG：{path}")
    width, height = struct.unpack(">II", head[16:24])
    if (width, height) != (size, size):
        raise CoverError(f"封面尺寸必须是 {size}x{size}，实际为 {width}x{height}：{path}")
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

        # 提示词是直接给图像模型的英文画面描述（禁文字），不是给 LLM 的指令模板
        prompt = build_cover_image_prompt(topic)

        cmd = [
            sys.executable, "-m", "aiboke.cover_runner",
            "--model", self._cfg.model_path,
            "--prompt", prompt,
            "--size", str(self._cfg.size),
            "--steps", str(self._cfg.steps),
            "--output", str(out_path),
        ]
        proc = _run_or_raise(self._run, cmd, "Z-Image 封面生成")
        if proc.returncode != 0:
            raise CoverError(f"Z-Image 封面生成失败：{proc.stderr.strip()[:300]}")
        _require_output(out_path)
        return _require_png_size(out_path, self._cfg.size)


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
        proc = _run_or_raise(self._run, cmd, "ffmpeg")
        if proc.returncode != 0:
            raise CoverError(f"生成兜底封面失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)


def build_generator(cfg: CoverConfig, runner: Callable | None = None, prefer_fallback: bool = False) -> CoverGenerator:
    if prefer_fallback:
        return FallbackCover(cfg, runner=runner)
    return ZImageCover(cfg, runner=runner)
