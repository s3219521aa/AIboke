"""ffmpeg / ffprobe 封装：拼接、响度归一化、转码、时长探测。

评分要求「无突然爆音或断裂」，因此幕间拼接插入自然停顿，并给整段开头做
50ms 淡入；响度归一化到 -16 LUFS 防止音量忽大忽小。这里不做交叉淡化——
交错两段的尾部与头部会切掉真实语音。

不混入背景音乐——赛题仅要求语音，BGM 会压低清晰度并增加失分风险。
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Callable, Sequence


class AudioError(RuntimeError):
    """音频处理失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _run_or_raise(run: Callable, cmd: list[str], label: str):
    """执行子进程，并把 OSError 归一为 AudioError。

    ffmpeg/ffprobe 不在 PATH 上时 subprocess 抛 FileNotFoundError（OSError 子类），
    它不是 AudioError，会穿透调用方的 `except AudioError`。
    """
    try:
        return run(cmd)
    except OSError as exc:
        raise AudioError(f"无法启动{label}进程（{cmd[0]}）：{exc}") from exc


def _require_output(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise AudioError(f"ffmpeg 未产出音频文件：{path}")
    return path


def probe_duration(path: Path, runner: Callable | None = None) -> float:
    """读取音频时长（秒）。时长门限的第三重保险依赖此函数。"""
    run = runner or _default_runner
    path = Path(path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = _run_or_raise(run, cmd, "ffprobe")
    if proc.returncode != 0:
        raise AudioError(f"ffprobe 读取时长失败：{proc.stderr.strip()[:300]}")

    try:
        seconds = float(proc.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise AudioError(f"无法从 ffprobe 输出解析时长：{proc.stdout!r}") from exc

    # float("nan") / float("inf") 都是合法解析，一旦流出就会让两个消费者分道
    # 扬镳：check_duration(nan) 报不通过却在校验重试提示时崩溃，
    # classify_duration(nan) 则返回 OK——那是失败开口。probe 是唯一产出实测
    # 时长的地方，在此拦截一次即可同时保护两者。
    if not math.isfinite(seconds):
        raise AudioError(f"ffprobe 返回的时长不是有限值：{proc.stdout!r}")
    return seconds


def concat_wavs(
    paths: Sequence[Path],
    out_path: Path,
    gap_ms: int = 400,
    runner: Callable | None = None,
) -> Path:
    """按顺序拼接多段音频：段间插入 gap_ms 毫秒静音，整段开头淡入 50ms。

    淡入写在 -filter_complex 图内而不是单独的 -af：与 -map [out] 同时使用时，
    简单的 -af 图没有输入流可绑定，ffmpeg 会拒绝整条命令。
    """
    if not paths:
        raise AudioError("至少需要一段音频才能拼接")
    if gap_ms < 0:
        raise AudioError(f"gap_ms 不能为负数，实际为 {gap_ms}")

    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        cmd = ["ffmpeg", "-y", "-i", str(paths[0]), "-c", "copy", str(out_path)]
        proc = _run_or_raise(run, cmd, "ffmpeg")
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg 复制音频失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)

    gap_s = gap_ms / 1000.0
    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", str(p)]

    # 段间补静音（末段除外），再整体 concat，最后对拼接结果做淡入
    n = len(paths)
    filter_parts = []
    for i in range(n - 1):
        if gap_ms:
            filter_parts.append(f"[{i}:a]apad=pad_dur={gap_s}[a{i}]")
        else:
            # apad 把 pad_dur=0 解释为「无限补静音」，会让 ffmpeg 永不退出
            filter_parts.append(f"[{i}:a]anull[a{i}]")
    filter_parts.append(f"[{n - 1}:a]anull[a{n - 1}]")
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[cat]")
    filter_parts.append("[cat]afade=t=in:d=0.05[out]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(out_path),
    ]
    proc = _run_or_raise(run, cmd, "ffmpeg")
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 拼接音频失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)


def normalize_loudness(
    in_path: Path,
    out_path: Path,
    target_lufs: float = -16.0,
    runner: Callable | None = None,
) -> Path:
    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        str(out_path),
    ]
    proc = _run_or_raise(run, cmd, "ffmpeg")
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 响度归一化失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)


def to_mp3(
    in_path: Path,
    out_path: Path,
    bitrate: str = "192k",
    runner: Callable | None = None,
) -> Path:
    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_path),
    ]
    proc = _run_or_raise(run, cmd, "ffmpeg")
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 转码 mp3 失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)

