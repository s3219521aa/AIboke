#!/usr/bin/env python
"""可跑性验证。在目标 GPU 机器上运行，退出码非 0 表示环境不可用。

最重要的一步是第 2 步：llama.cpp 旧版本在 Qwen3.8 的 DeltaNet 层
CUDA 路径上有 bug，症状是模型正常加载、显存正常、速度正常、零报错，
但输出全是乱码。因此必须校验输出内容，而不能只看加载是否成功。

音色走**生产路径**：读仓库里真实的 configs/voice_presets.json，再按性别取
音色（bootstrap.load_presets_from_config + voices.resolve_pair）。手搓一套只有
描述、没有参考音频的假预设去冒烟，等于绕开「性别由参考音频保证」这条 0 分
门限的唯一真实来源——冒烟通过也不代表生产能跑。只有在音色尚未生成的过渡期
才用 --no-reference-audio 走内置合成预设；注意那条路只在 tts.backend 为
transformers 时才真的跑得通（llamacpp 后端的音色条件就是参考音频本身）。

前置条件（默认 llamacpp 后端）：configs/voice_presets.json 的每个预设都必须
填好 reference_audio，否则第 4 步会以 TtsError 明确失败——详见 README 的
「部署前必办」一节。TTS 那一半环境还没就绪时，用 --skip-tts 只跑其余步骤
（跳过的步骤**不算通过**，结尾会显式说明）。

用法：
    python scripts/smoke_test.py --config configs/default.yaml
    python scripts/smoke_test.py --skip-tts          # 只跑 LLM 与封面
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke import audio_utils  # noqa: E402
from aiboke.bootstrap import load_presets_from_config  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.cover import CoverError, build_generator  # noqa: E402
from aiboke.gates import cjk_ratio  # noqa: E402
from aiboke.llm_client import LlmClient, LlmError  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import TtsError, build_backend  # noqa: E402
from aiboke.voices import resolve_pair  # noqa: E402

MIN_LLAMACPP_BUILD = 10450

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESETS_PATH = _REPO_ROOT / "configs" / "voice_presets.json"

# 该二进制对参考音频的要求（fork 文档：必须是 24 kHz）
REQUIRED_REFERENCE_RATE = 24000

# 这些是各组件自带的领域异常。冒烟测试最常见的两种失败（缺 reference_audio
# 的 TtsError、连不上 llama-server 的 LlmError）都在其中——只捕 SmokeFailure
# 的话，操作员看到的是一段 traceback 而不是可读的「不通过：……」。
DOMAIN_ERRORS = (TtsError, LlmError, CoverError, audio_utils.AudioError)

# --no-reference-audio 时才用的内置预设：没有参考音频，只服务于「音色尚未
# 生成、但想先确认其余环境」的过渡期。它不是生产路径。
SYNTHETIC_VOICE_SEEDS = (("smoke_m", "男", "清晰的男声"), ("smoke_f", "女", "清晰的女声"))


class SmokeFailure(RuntimeError):
    pass


def step_no(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")


def resolve_smoke_voices(presets_path: Path, allow_synthetic: bool) -> VoicePair:
    """冒烟测试用的音色对，默认走生产路径。

    生产路径 = 读真实预设文件 + 按性别解析（bootstrap -> voices.resolve_pair），
    与 pipeline 用的是同一条链路：预设文件坏掉、缺某一性别、同性别只有一条，
    都会在这里以可读的错误浮出来，而不是等到整集生成时才发现。
    """
    if allow_synthetic:
        return VoicePair(
            speaker1=VoicePreset(
                id=SYNTHETIC_VOICE_SEEDS[0][0],
                gender=SYNTHETIC_VOICE_SEEDS[0][1],
                description=SYNTHETIC_VOICE_SEEDS[0][2],
            ),
            speaker2=VoicePreset(
                id=SYNTHETIC_VOICE_SEEDS[1][0],
                gender=SYNTHETIC_VOICE_SEEDS[1][1],
                description=SYNTHETIC_VOICE_SEEDS[1][2],
            ),
        )

    presets_path = Path(presets_path)
    try:
        presets = load_presets_from_config(presets_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SmokeFailure(f"无法读取音色预设 {presets_path}：{exc}") from exc
    try:
        return resolve_pair(presets, "男", "女")
    except ValueError as exc:
        raise SmokeFailure(f"从 {presets_path} 解析 男+女 音色失败：{exc}") from exc


def find_llama_server(cfg) -> Path | None:
    """定位 upstream 的 llama-server。

    MOSS-TTSD 的二进制来自 OpenMOSS fork、llama-server 来自 upstream，两者不在
    同一个构建树里，所以先查 PATH（Dockerfile 已把两个 bin 目录都加进 PATH），
    找不到再退回 tts.binary 的同级目录。两处都没有则返回 None，由调用方告警跳过。
    """
    found = shutil.which("llama-server")
    if found:
        return Path(found)
    if cfg.tts.binary:
        candidate = Path(cfg.tts.binary).parent / "llama-server"
        if candidate.exists():
            return candidate
    return None


def check_llamacpp_version(cfg) -> None:
    server = find_llama_server(cfg)
    if server is None:
        print(
            "  警告：PATH 与 tts.binary 同级目录都找不到 llama-server，"
            "跳过版本检查。请自行确认构建号 >= b10450。"
        )
        return
    try:
        out = subprocess.run([str(server), "--version"], capture_output=True, text=True).stdout
    except OSError as exc:
        raise SmokeFailure(f"无法执行 {server}：{exc}") from exc

    print(f"  {server}")
    print(f"  {out.strip().splitlines()[0] if out.strip() else '(无版本信息)'}")
    if str(MIN_LLAMACPP_BUILD) not in out:
        print(
            f"  警告：未能确认构建号 >= b{MIN_LLAMACPP_BUILD}。"
            "低于该版本在 Qwen3.8 上可能静默输出乱码，请务必核实。"
        )


def check_llm(cfg) -> None:
    """最关键的一步：必须校验输出内容为连贯中文，而不只是调用成功。"""
    client = LlmClient(cfg.llm)
    prompt = "请只回答一句话，用简体中文介绍你自己是一位播客主播。不要有任何其他内容。"
    text = client.complete("你是一位中文播客主播。", prompt)
    ratio = cjk_ratio(text)

    print(f"  输出：{text[:120]}")
    print(f"  中文占比：{ratio:.3f}")

    if ratio < 0.5:
        raise SmokeFailure(
            f"LLM 输出疑似乱码或非中文（中文占比 {ratio:.3f}）。"
            "这极可能是 llama.cpp 版本低于 b10450 导致的 DeltaNet CUDA bug——"
            "该 bug 不会报错，只会静默产出乱码。请升级 llama.cpp 并确认 "
            "libggml-cuda.so 与二进制同版本。"
        )


def probe_sample_rate(path: Path) -> int | None:
    """尽力读出音频的采样率；ffprobe 不可用或读不出时返回 None。"""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def check_reference_rates(voices: VoicePair) -> None:
    """核对参考音频的采样率。

    采样率不一致**不会**被拦下：拼接走 ffmpeg 的滤镜图，它会自动插入重采样
    把两段对齐，另一段的音色被静默改变——一个听不出成因、也无人报警的退化。
    该二进制要求 24kHz，所以这里先查明（读不出来就只提醒，不阻断）。
    """
    for preset in (voices.speaker1, voices.speaker2):
        if not preset.reference_audio:
            print(f"  提示：{preset.id} 没有 reference_audio，跳过采样率核对")
            continue
        rate = probe_sample_rate(Path(preset.reference_audio))
        if rate is None:
            print(
                f"  提示：无法用 ffprobe 读取 {preset.id} 的参考音频，"
                f"请人工确认它是 {REQUIRED_REFERENCE_RATE} Hz"
            )
        elif rate != REQUIRED_REFERENCE_RATE:
            raise SmokeFailure(
                f"{preset.id} 的参考音频是 {rate} Hz，该二进制要求 "
                f"{REQUIRED_REFERENCE_RATE} Hz。采样率不一致不会被拦下——"
                "ffmpeg 会在拼接时静默重采样，两位主播的音色被无声地改变。"
                "先转换再继续："
                f"ffmpeg -i {preset.reference_audio} -ar {REQUIRED_REFERENCE_RATE} "
                f"-ac 1 {preset.reference_audio}.24k.wav"
            )
        else:
            print(f"  {preset.id} 参考音频：{rate} Hz")


def check_tts(cfg, voices: VoicePair) -> None:
    tts = build_backend(cfg.tts)
    turns = (
        Turn(speaker=1, text="大家好，欢迎收听本期商业故事。"),
        Turn(speaker=2, text="今天我们聊一个关于品牌转型的话题。"),
        Turn(speaker=1, text="没错，这个故事挺有意思的。"),
        Turn(speaker=2, text="那我们就从头说起吧。"),
    )
    print(
        f"  音色：{voices.speaker1.id}（{voices.speaker1.gender}）"
        f" + {voices.speaker2.id}（{voices.speaker2.gender}）"
    )
    check_reference_rates(voices)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.wav"
        tts.synthesize(turns, voices, out, target_seconds=30.0)
        seconds = audio_utils.probe_duration(out)
        size_kb = out.stat().st_size / 1024
    print(f"  产出 {seconds:.1f} 秒音频（{size_kb:.0f} KB）")
    if seconds < 3.0:
        raise SmokeFailure(f"TTS 仅产出 {seconds:.1f} 秒音频，明显偏短，疑似失败")
    print("  请人工试听，确认两位主播音色可区分，且性别与预设标注一致")


def check_cover(cfg) -> None:
    gen = build_generator(cfg.cover)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.png"
        gen.generate("一家公司如何完成品牌转型", out)
        size_kb = out.stat().st_size / 1024
    print(f"  产出封面（{size_kb:.0f} KB），期望 {cfg.cover.size}x{cfg.cover.size}")
    if size_kb < 5:
        raise SmokeFailure("封面文件过小，疑似生成失败")


def check_roundtrip(cfg) -> None:
    client = LlmClient(cfg.llm)
    raw = client.complete(
        "你只输出 JSON，不要任何其他内容。",
        '请输出 {"title": "测试", "content": [{"speaker": 1, "text": "你好"}]} 这个 JSON。',
    )
    from aiboke.script_writer import parse_act

    try:
        turns = parse_act(raw)
    except Exception as exc:  # noqa: BLE001
        raise SmokeFailure(f"LLM 输出的 JSON 无法解析：{exc}\n原始输出：{raw[:200]}") from exc
    print(f"  JSON 解析成功，得到 {len(turns)} 轮对话")


def build_steps(cfg, args) -> list[tuple[str, Callable[[], None]]]:
    """按开关组装要跑的步骤（顺序即编号顺序）。"""
    steps: list[tuple[str, Callable[[], None]]] = [
        ("llama.cpp 版本与库一致性", lambda: check_llamacpp_version(cfg)),
        ("LLM 中文输出正确性（防静默乱码）", lambda: check_llm(cfg)),
        ("LLM JSON 输出可解析", lambda: check_roundtrip(cfg)),
    ]
    if not args.skip_tts:
        # 音色在步骤真正执行时才解析：预设文件坏掉属于「这一步不通过」，
        # 应当被步骤循环捕获成一行可读信息，而不是 main 里的 traceback
        steps.append(
            (
                "TTS 双人中文语音",
                lambda: check_tts(
                    cfg,
                    resolve_smoke_voices(args.voice_presets, args.no_reference_audio),
                ),
            )
        )
    if not args.skip_cover:
        steps.append(("封面生成", lambda: check_cover(cfg)))
    return steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="可跑性验证")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--skip-cover", action="store_true", help="跳过封面检查")
    ap.add_argument(
        "--skip-tts",
        action="store_true",
        help="跳过 TTS 检查（TTS 那一半环境未就绪时，仍可跑其余步骤）",
    )
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
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    steps = build_steps(cfg, args)
    total = len(steps)
    skipped = args.skip_tts or args.skip_cover

    for i, (title, fn) in enumerate(steps, start=1):
        step_no(i, total, title)
        try:
            fn()
        except (SmokeFailure, *DOMAIN_ERRORS) as exc:
            print(f"\n\033[1;31m不通过：{exc}\033[0m", file=sys.stderr)
            return 1

    if skipped:
        # 跳过的步骤不算通过：说「环境可用于生成播客」会让操作员据此进入
        # 批处理，而那正是最贵的一种误判
        print(
            "\n\033[1;33m已运行的步骤全部通过。跳过的步骤未验证，"
            "环境尚不能判定为可用。\033[0m"
        )
        return 0

    print("\n\033[1;32m全部通过。环境可用于生成播客。\033[0m")
    print("提醒：请人工确认音色预设的性别标注与试听结果一致（见设计文档 7.2），")
    print(f"      并用 ffprobe 逐一核对参考音频的采样率为 {REQUIRED_REFERENCE_RATE} Hz。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
