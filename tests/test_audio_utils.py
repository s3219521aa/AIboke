"""ffmpeg / ffprobe 封装的测试。

全部测试通过注入 FakeRunner 替代 subprocess，不启动任何真实进程；
「退出码 0 但没产出文件」这一静默失败由不写文件的 FakeRunner 覆盖。
"""

from pathlib import Path

import pytest

from aiboke.audio_utils import (
    AudioError,
    concat_wavs,
    normalize_loudness,
    probe_duration,
    to_mp3,
)


class FakeRunner:
    def __init__(self, *, stdout="", returncode=0, stderr="", writes=None):
        self.calls = []
        self._stdout = stdout
        self._rc = returncode
        self._stderr = stderr
        self._writes = writes

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if self._writes:
            self._writes(cmd)
        return type("R", (), {"returncode": self._rc, "stdout": self._stdout, "stderr": self._stderr})()


def test_probe_duration_parses_ffprobe_output():
    runner = FakeRunner(stdout="512.345\n")
    assert probe_duration(Path("a.mp3"), runner=runner) == pytest.approx(512.345)


def test_probe_duration_uses_ffprobe_show_entries():
    runner = FakeRunner(stdout="1.0\n")
    probe_duration(Path("a.mp3"), runner=runner)
    joined = " ".join(runner.calls[0])
    assert "ffprobe" in joined
    assert "format=duration" in joined


def test_probe_duration_raises_on_unparseable_output():
    with pytest.raises(AudioError, match="时长"):
        probe_duration(Path("a.mp3"), runner=FakeRunner(stdout="N/A\n"))


def test_probe_duration_raises_on_nonzero_exit():
    with pytest.raises(AudioError, match="ffprobe"):
        probe_duration(Path("a.mp3"), runner=FakeRunner(returncode=1, stderr="No such file"))


def test_concat_wavs_requires_at_least_one_input(tmp_path):
    with pytest.raises(AudioError, match="至少"):
        concat_wavs([], tmp_path / "o.wav", runner=FakeRunner())


def test_concat_wavs_invokes_ffmpeg_with_all_inputs(tmp_path):
    out = tmp_path / "o.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], out, runner=runner)
    joined = " ".join(runner.calls[0])
    assert "ffmpeg" in joined
    assert "a.wav" in joined and "b.wav" in joined


def test_concat_wavs_applies_gap(tmp_path):
    out = tmp_path / "o.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], out, gap_ms=400, runner=runner)
    assert "0.4" in " ".join(runner.calls[0])


def test_concat_wavs_raises_when_no_output(tmp_path):
    with pytest.raises(AudioError, match="未产出"):
        concat_wavs([tmp_path / "a.wav"], tmp_path / "o.wav", runner=FakeRunner())


def test_normalize_loudness_targets_configured_lufs(tmp_path):
    out = tmp_path / "n.wav"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"RIFF"))
    normalize_loudness(tmp_path / "i.wav", out, target_lufs=-16.0, runner=runner)
    assert "loudnorm" in " ".join(runner.calls[0])
    assert "-16.0" in " ".join(runner.calls[0])


def test_to_mp3_sets_bitrate(tmp_path):
    out = tmp_path / "o.mp3"
    runner = FakeRunner(writes=lambda cmd: out.write_bytes(b"ID3"))
    to_mp3(tmp_path / "i.wav", out, bitrate="192k", runner=runner)
    joined = " ".join(runner.calls[0])
    assert "libmp3lame" in joined
    assert "192k" in joined


def test_to_mp3_raises_when_no_output(tmp_path):
    with pytest.raises(AudioError, match="未产出"):
        to_mp3(tmp_path / "i.wav", tmp_path / "o.mp3", runner=FakeRunner())
