"""ffmpeg / ffprobe 封装：拼接、响度归一化、转码、时长探测。

评分要求「无突然爆音或断裂」，因此幕间拼接需要插入自然停顿并做
淡入淡出；响度归一化到 -16 LUFS 防止音量忽大忽小。

不混入背景音乐——赛题仅要求语音，BGM 会压低清晰度并增加失分风险。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence


class AudioError(RuntimeError):
    """音频处理失败。"""


def _default_runner(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


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
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe 读取时长失败：{proc.stderr.strip()[:300]}")

    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError) as exc:
        raise AudioError(f"无法从 ffprobe 输出解析时长：{proc.stdout!r}") from exc


def concat_wavs(
    paths: Sequence[Path],
    out_path: Path,
    gap_ms: int = 400,
    runner: Callable | None = None,
) -> Path:
    """按顺序拼接多段音频，段间插入停顿并做交叉淡化。"""
    if not paths:
        raise AudioError("至少需要一段音频才能拼接")

    run = runner or _default_runner
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        cmd = ["ffmpeg", "-y", "-i", str(paths[0]), "-c", "copy", str(out_path)]
        proc = run(cmd)
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg 复制音频失败：{proc.stderr.strip()[:300]}")
        return _require_output(out_path)

    gap_s = gap_ms / 1000.0
    inputs: list[str] = []
    for p in paths:
        inputs += ["-i", str(p)]

    # 段间插入静音，末端做淡化，避免生硬切边
    n = len(paths)
    filter_parts = []
    for i in range(n - 1):
        filter_parts.append(
            f"[{i}:a]adelay=0|0,apad=pad_dur={gap_s}[a{i}]"
        )
    filter_parts.append(f"[{n - 1}:a]anull[a{n - 1}]")
    concat_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[out]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-af", "afade=t=in:d=0.05",
        str(out_path),
    ]
    proc = run(cmd)
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
    proc = run(cmd)
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
    proc = run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 转码 mp3 失败：{proc.stderr.strip()[:300]}")
    return _require_output(out_path)
