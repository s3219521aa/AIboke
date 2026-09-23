#!/usr/bin/env python
"""语速标定。

configs/default.yaml 里的 chars_per_minute=200 是从已有的播客长度
标定表折算的移植值，不同音色与语速设置会显著影响实际语速。本脚本
用实际音色合成一段已知字数的中文，测出真实时长后反算语速。

用法：
    python scripts/calibrate_length.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke import audio_utils  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import build_backend  # noqa: E402

# 一段字数已知、内容中立的中文，避免数字与英文影响语速
SAMPLE_TEXT = (
    "今天我们聊一个很有意思的话题，关于一家公司如何在最困难的时候做出改变，"
    "并且最终重新赢得了市场的认可。这个故事里有决策，有取舍，也有运气。"
    "我们先从它的起点说起，那时候没人看好它。"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标定实际语速")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--repeat", type=int, default=3, help="重复次数，取中位数以降低抖动")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tts = build_backend(cfg.tts)

    turns = (Turn(speaker=1, text=SAMPLE_TEXT),)
    voices = VoicePair(
        speaker1=VoicePreset(id="calib1", gender="男", description="标准的男声，语速自然"),
        speaker2=VoicePreset(id="calib2", gender="女", description="标准的女声，语速自然"),
    )
    char_count = len(SAMPLE_TEXT)
    measured: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.repeat):
            out = Path(tmp) / f"calib{i}.wav"
            tts.synthesize(turns, voices, out, target_seconds=30.0)
            seconds = audio_utils.probe_duration(out)
            measured.append(seconds)
            print(f"第 {i + 1} 次：{char_count} 字 -> {seconds:.2f} 秒")

    measured.sort()
    median = measured[len(measured) // 2]
    chars_per_minute = char_count / median * 60.0

    print(f"\n实测语速：{chars_per_minute:.1f} 字/分钟")
    print(f"当前配置：{cfg.chars_per_minute} 字/分钟")
    print("\n建议把 configs/default.yaml 的 chars_per_minute 改为：")
    print(f"  chars_per_minute: {chars_per_minute:.1f}")
    print(
        f"\n按该语速，{cfg.target_seconds:.0f} 秒目标时长对应"
        f" {int(cfg.target_seconds / 60 * chars_per_minute)} 字。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
