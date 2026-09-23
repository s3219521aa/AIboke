"""封面生成与兜底的测试。

全部测试通过注入 FakeRunner 替代 subprocess，不启动任何真实进程。
兜底的 FallbackCover 不依赖 GPU，是本模块「产物必须存在」的最后一道保障。

封面尺寸校验只读 PNG 的 IHDR（字节 16-24），因此夹具用 `_png_bytes()`
构造「真实签名 + 带正确 CRC 的 IHDR」，不必真的编码一张 1024x1024 的图。
"""

import struct
import sys
import zlib
from pathlib import Path

import pytest

from aiboke.config import CoverConfig
from aiboke.cover import (
    CoverError,
    FallbackCover,
    ZImageCover,
    build_generator,
)


def _png_bytes(size: int, height: int | None = None) -> bytes:
    """构造 PNG 签名 + IHDR 块（尺寸校验所需的最小合法头部）。"""
    width = size
    height = size if height is None else height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    return b"\x89PNG\r\n\x1a\n" + chunk + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


class FakeRunner:
    def __init__(self, *, returncode=0, stderr="", writes=None, raises=None):
        self.calls = []
        self._rc = returncode
        self._stderr = stderr
        self._writes = writes
        self._raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._raises:
            raise self._raises
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._rc, "stdout": "", "stderr": self._stderr})()


def _cfg():
    return CoverConfig(steps=8, size=1024, model_path="/models/Z-Image-Turbo")


def _prompt_arg(cmd):
    return cmd[cmd.index("--prompt") + 1]


def test_zimage_invokes_runner_with_size_and_steps(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(_png_bytes(1024)))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    joined = " ".join(runner.calls[0])
    assert "1024" in joined
    assert "8" in joined
    assert "/models/Z-Image-Turbo" in joined


def test_zimage_uses_current_interpreter_and_cover_runner(tmp_path):
    """解释器用 sys.executable，避免 PATH 上没有 python 时抛 FileNotFoundError。"""
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(_png_bytes(1024)))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)

    cmd = runner.calls[0]
    assert cmd[0] == sys.executable
    assert "aiboke.cover_runner" in cmd


def test_zimage_prompt_is_a_visual_description_that_forbids_text(tmp_path):
    """提示词必须是直接给图像模型的视觉描述，且明确禁止画面出现文字。

    不能把给 LLM 的指令模板（「请只输出英文的图像生成提示词」）原样喂给
    文生图模型——那等于让它输出一段提示词文字，而不是画一张封面。
    """
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(_png_bytes(1024)))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)

    prompt = _prompt_arg(runner.calls[0])
    assert "星巴克国内运营转移" in prompt          # 主题必须进入画面描述
    assert "no text" in prompt.lower()             # 明确禁止文字
    assert "请只输出" not in prompt                # 不得再传 LLM 指令模板
    assert "要求：" not in prompt


def test_zimage_rejects_png_of_wrong_size(tmp_path):
    """规格要求 1024x1024，必须按 PNG 里的真实尺寸校验，不能只看配置值。"""
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(_png_bytes(512)))
    with pytest.raises(CoverError, match="1024x1024"):
        ZImageCover(_cfg(), runner=runner).generate("主题", out)


def test_zimage_rejects_non_png_output(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"not a png at all"))
    with pytest.raises(CoverError, match="PNG"):
        ZImageCover(_cfg(), runner=runner).generate("主题", out)


def test_zimage_raises_when_no_output(tmp_path):
    with pytest.raises(CoverError, match="未产出"):
        ZImageCover(_cfg(), runner=FakeRunner()).generate("主题", tmp_path / "c.png")


def test_zimage_raises_on_nonzero_exit(tmp_path):
    runner = FakeRunner(returncode=1, stderr="CUDA out of memory")
    with pytest.raises(CoverError, match="out of memory"):
        ZImageCover(_cfg(), runner=runner).generate("主题", tmp_path / "c.png")


def test_zimage_wraps_missing_interpreter_into_covererror(tmp_path):
    """解释器缺失时抛的是 FileNotFoundError，必须归一为 CoverError。

    否则它会穿透编排层的 `except CoverError`，直接把兜底封面也一起跳过。
    """
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(CoverError, match="无法启动"):
        ZImageCover(_cfg(), runner=runner).generate("主题", tmp_path / "c.png")


def test_zimage_raises_when_model_path_missing():
    with pytest.raises(CoverError, match="model_path"):
        ZImageCover(CoverConfig(model_path=None), runner=FakeRunner()).generate(
            "主题", Path("c.png")
        )


def test_fallback_cover_produces_file(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    FallbackCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    assert out.exists()
    assert "ffmpeg" in " ".join(runner.calls[0])


def test_fallback_cover_wraps_missing_ffmpeg_into_covererror(tmp_path):
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(CoverError, match="无法启动"):
        FallbackCover(_cfg(), runner=runner).generate("主题", tmp_path / "c.png")


def test_build_generator_returns_zimage_by_default():
    assert isinstance(build_generator(_cfg()), ZImageCover)


def test_build_generator_returns_fallback_when_requested():
    assert isinstance(build_generator(_cfg(), prefer_fallback=True), FallbackCover)
