"""封面生成与兜底的测试。

全部测试通过注入 FakeRunner 替代 subprocess，不启动任何真实进程。
兜底的 FallbackCover 不依赖 GPU，是本模块「产物必须存在」的最后一道保障。
"""

from pathlib import Path

import pytest

from aiboke.config import CoverConfig
from aiboke.cover import (
    CoverError,
    FallbackCover,
    ZImageCover,
    build_generator,
)


class FakeRunner:
    def __init__(self, *, returncode=0, stderr="", writes=None):
        self.calls = []
        self._rc = returncode
        self._stderr = stderr
        self._writes = writes

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._rc, "stdout": "", "stderr": self._stderr})()


def _cfg():
    return CoverConfig(steps=8, size=1024, model_path="/models/Z-Image-Turbo")


def test_zimage_invokes_runner_with_size_and_steps(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    joined = " ".join(runner.calls[0])
    assert "1024" in joined
    assert "8" in joined
    assert "/models/Z-Image-Turbo" in joined


def test_zimage_prompt_forbids_text_in_image(tmp_path):
    out = tmp_path / "c.png"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"\x89PNG"))
    ZImageCover(_cfg(), runner=runner).generate("星巴克国内运营转移", out)
    assert "文字" in " ".join(runner.calls[0])


def test_zimage_raises_when_no_output(tmp_path):
    with pytest.raises(CoverError, match="未产出"):
        ZImageCover(_cfg(), runner=FakeRunner()).generate("主题", tmp_path / "c.png")


def test_zimage_raises_on_nonzero_exit(tmp_path):
    runner = FakeRunner(returncode=1, stderr="CUDA out of memory")
    with pytest.raises(CoverError, match="out of memory"):
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


def test_build_generator_returns_zimage_by_default():
    assert isinstance(build_generator(_cfg()), ZImageCover)


def test_build_generator_returns_fallback_when_requested():
    assert isinstance(build_generator(_cfg(), prefer_fallback=True), FallbackCover)
