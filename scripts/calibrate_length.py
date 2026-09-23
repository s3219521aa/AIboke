#!/usr/bin/env python
"""语速标定。

configs/default.yaml 里的 chars_per_minute=200 是从已有的播客长度
标定表折算的移植值，不同音色与语速设置会显著影响实际语速。本脚本
用实际音色合成一段已知字数的中文，测出真实时长后反算语速。

音色走**生产路径**：读仓库里真实的 configs/voice_presets.json，再按性别取
音色（与 pipeline 同一条链路）。语速是音色相关的，用假预设标定出来的数字
与生产用的音色无关，标了等于没标。音色尚未生成时才用 --no-reference-audio
走内置合成预设，且那条路只在 tts.backend 为 transformers 时跑得通
（llamacpp 后端的音色条件就是参考音频本身）。

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
from aiboke.bootstrap import load_presets_from_config  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import TtsError, build_backend  # noqa: E402
from aiboke.voices import resolve_pair  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS_PATH = _REPO_ROOT / "configs" / "voice_presets.json"

SAMPLE_TEXT = (
    "今天我们聊一个很有意思的话题，关于一家公司如何在最困难的时候做出改变，"
    "并且最终重新赢得了市场的认可。这个故事里有决策，有取舍，也有运气。"
    "我们先从它的起点说起，那时候没人看好它。"
)


def resolve_calibration_voices(presets_path: Path, allow_synthetic: bool) -> VoicePair:
    """标定用的音色对，默认与生成时同一条解析链路。

    失败一律以可读信息浮出，不吞成「用默认音色继续」：预设文件缺失/格式错，
    或某一性别没有可用音色，都属于配置错误（退出码 2）。用无参考音频的假
    预设标定出来的语速，填回 configs/default.yaml 只会让时长门限的估算
    系统性偏掉。
    """
    if allow_synthetic:
        return VoicePair(
            speaker1=VoicePreset(id="calib1", gender="男", description="标准的男声，语速自然"),
            speaker2=VoicePreset(id="calib2", gender="女", description="标准的女声，语速自然"),
        )

    presets_path = Path(presets_path)
    if not presets_path.exists():
        raise FileNotFoundError(
            f"音色预设文件不存在：{presets_path}（用 --voice-presets 指定，"
            "或在音色尚未生成时用 --no-reference-audio）"
        )
    presets = load_presets_from_config(presets_path)
    voices = resolve_pair(presets, "男", "女")
    print(
        f"音色：{voices.speaker1.id}（{voices.speaker1.gender}）"
        f" + {voices.speaker2.id}（{voices.speaker2.gender}），来自 {presets_path}"
    )
    return voices


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="标定实际语速")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--repeat", type=int, default=3, help="重复次数，取中位数以降低抖动")
    ap.add_argument(
        "--voice-presets",
        type=Path,
        default=DEFAULT_PRESETS_PATH,
        help="音色预设文件（默认仓库的 configs/voice_presets.json）",
    )
    ap.add_argument(
        "--no-reference-audio",
        action="store_true",
        help="改用内置合成预设（无参考音频）；只在 transformers 后端下跑得通，"
             "llamacpp 后端的音色条件就是参考音频",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    try:
        voices = resolve_calibration_voices(args.voice_presets, args.no_reference_audio)
    except (FileNotFoundError, ValueError) as exc:
        print(f"音色解析失败：{exc}", file=sys.stderr)
        return 2

    try:
        tts = build_backend(cfg.tts)
    except TtsError as exc:
        print(f"语音后端不可用：{exc}", file=sys.stderr)
        return 1

    turns = (Turn(speaker=1, text=SAMPLE_TEXT),)
    char_count = len(SAMPLE_TEXT)
    measured: list[float] = []

    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.repeat):
            out = Path(tmp) / f"calib{i}.wav"
            try:
                tts.synthesize(turns, voices, out, target_seconds=30.0)
                seconds = audio_utils.probe_duration(out)
            except (TtsError, audio_utils.AudioError) as exc:
                print(f"语音合成失败：{exc}", file=sys.stderr)
                return 1
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
    if args.no_reference_audio:
        print(
            "\n注意：本次用的是内置合成预设（无参考音频），这个语速只代表该假预设。"
            "回填真实音色的参考音频后必须重新标定。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
