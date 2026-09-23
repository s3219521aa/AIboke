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
    def __init__(self, *, stdout="", returncode=0, stderr="", writes=None, raises=None):
        self.calls = []
        self._stdout = stdout
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
        return type("R", (), {"returncode": self._rc, "stdout": self._stdout, "stderr": self._stderr})()


def _concat(tmp_path, runner, **kwargs):
    out = tmp_path / "o.wav"
    concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], out, runner=runner, **kwargs)
    return out, " ".join(runner.calls[0])


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


def test_probe_duration_rejects_non_finite_output():
    """NaN / inf 必须被拒绝，不能作为「实测时长」流出。

    float("nan") 是合法解析，坏容器会让 ffprobe 打印 nan/inf。两个消费者
    会因此分道扬镳：check_duration(nan) 报不通过却在校验重试提示时崩溃，
    classify_duration(nan) 则返回 OK——那是失败开口。probe 是唯一产出实测
    时长的地方，在这里拦一次即可同时保护两者。
    """
    for bad in ("nan\n", "inf\n", "-inf\n"):
        with pytest.raises(AudioError, match="时长"):
            probe_duration(Path("a.mp3"), runner=FakeRunner(stdout=bad))


def test_probe_duration_raises_on_nonzero_exit():
    with pytest.raises(AudioError, match="ffprobe"):
        probe_duration(Path("a.mp3"), runner=FakeRunner(returncode=1, stderr="No such file"))


def test_missing_ffmpeg_binary_surfaces_as_audioerror():
    """ffmpeg/ffprobe 不在 PATH 上时抛的是 FileNotFoundError（OSError），

    它必须被归一为 AudioError，否则会穿透调用方的 `except AudioError`。
    与 F5 在 tts/cover 里修的是同一类缺陷，本模块一并拉齐。
    """
    runner = FakeRunner(raises=FileNotFoundError("[WinError 2] 系统找不到指定的文件"))
    with pytest.raises(AudioError, match="无法启动"):
        probe_duration(Path("a.mp3"), runner=runner)


def test_concat_wavs_requires_at_least_one_input(tmp_path):
    with pytest.raises(AudioError, match="至少"):
        concat_wavs([], tmp_path / "o.wav", runner=FakeRunner())


def test_concat_wavs_invokes_ffmpeg_with_all_inputs(tmp_path):
    runner = FakeRunner(writes=lambda cmd: (tmp_path / "o.wav").write_bytes(b"RIFF"))
    _, joined = _concat(tmp_path, runner)
    assert "ffmpeg" in joined
    assert "a.wav" in joined and "b.wav" in joined


def test_concat_wavs_applies_gap(tmp_path):
    runner = FakeRunner(writes=lambda cmd: (tmp_path / "o.wav").write_bytes(b"RIFF"))
    _, joined = _concat(tmp_path, runner, gap_ms=400)
    assert "0.4" in joined


def test_concat_wavs_folds_fade_into_the_filter_graph(tmp_path):
    """淡入必须写在 -filter_complex 图里，不能再出现独立的 -af。

    同时给 -filter_complex + -map 和 -af 时，简单的 -af 图没有输入流可绑定，
    ffmpeg 会拒绝整条命令——那会打断每一次多幕拼接（关键路径）。
    """
    runner = FakeRunner(writes=lambda cmd: (tmp_path / "o.wav").write_bytes(b"RIFF"))
    _, joined = _concat(tmp_path, runner)

    cmd = runner.calls[0]
    assert "-af" not in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.endswith("[cat]afade=t=in:d=0.05[out]")
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "[out]"
    assert "adelay" not in graph  # 零延迟的 adelay 是无作用的死滤镜


def test_concat_wavs_rejects_negative_gap(tmp_path):
    with pytest.raises(AudioError, match="gap_ms"):
        concat_wavs([tmp_path / "a.wav"], tmp_path / "o.wav", gap_ms=-1, runner=FakeRunner())


def test_concat_wavs_zero_gap_uses_anull_not_apad(tmp_path):
    """gap_ms=0 必须退化为 anull。

    apad 把 pad_dur=0 解释为「无限补静音」，会让 ffmpeg 永不退出——
    这是个挂死，不是慢。
    """
    runner = FakeRunner(writes=lambda cmd: (tmp_path / "o.wav").write_bytes(b"RIFF"))
    _, joined = _concat(tmp_path, runner, gap_ms=0)
    assert "anull" in joined
    assert "apad" not in joined
    assert "pad_dur" not in joined


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
