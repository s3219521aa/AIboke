#!/usr/bin/env python
"""可跑性验证。在目标 GPU 机器上运行，退出码非 0 表示环境不可用。

最重要的一步是第 2 步：llama.cpp 旧版本在 Qwen3.8 的 DeltaNet 层
CUDA 路径上有 bug，症状是模型正常加载、显存正常、速度正常、零报错，
但输出全是乱码。因此必须校验输出内容，而不能只看加载是否成功。

前置条件（默认 llamacpp 后端）：configs/voice_presets.json 的每个预设都必须
填好 reference_audio，否则第 4 步会以 TtsError 明确失败——详见 README 的
「部署前必办」一节。

用法：
    python scripts/smoke_test.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke import audio_utils  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.cover import CoverError, build_generator  # noqa: E402
from aiboke.gates import cjk_ratio  # noqa: E402
from aiboke.llm_client import LlmClient, LlmError  # noqa: E402
from aiboke.schema import Turn, VoicePair, VoicePreset  # noqa: E402
from aiboke.tts import TtsError, build_backend  # noqa: E402

MIN_LLAMACPP_BUILD = 10450

# 这些是各组件自带的领域异常。冒烟测试最常见的两种失败（缺 reference_audio
# 的 TtsError、连不上 llama-server 的 LlmError）都在其中——只捕 SmokeFailure
# 的话，操作员看到的是一段 traceback 而不是可读的「不通过：……」。
DOMAIN_ERRORS = (TtsError, LlmError, CoverError, audio_utils.AudioError)


class SmokeFailure(RuntimeError):
    pass


def step_no(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title}")


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


def check_tts(cfg) -> None:
    tts = build_backend(cfg.tts)
    turns = (
        Turn(speaker=1, text="大家好，欢迎收听本期商业故事。"),
        Turn(speaker=2, text="今天我们聊一个关于品牌转型的话题。"),
        Turn(speaker=1, text="没错，这个故事挺有意思的。"),
        Turn(speaker=2, text="那我们就从头说起吧。"),
    )
    voices = VoicePair(
        speaker1=VoicePreset(id="smoke_m", gender="男", description="清晰的男声"),
        speaker2=VoicePreset(id="smoke_f", gender="女", description="清晰的女声"),
    )
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="可跑性验证")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--skip-cover", action="store_true", help="跳过封面检查")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    steps = [
        ("llama.cpp 版本与库一致性", lambda: check_llamacpp_version(cfg)),
        ("LLM 中文输出正确性（防静默乱码）", lambda: check_llm(cfg)),
        ("LLM JSON 输出可解析", lambda: check_roundtrip(cfg)),
        ("TTS 双人中文语音", lambda: check_tts(cfg)),
    ]
    if not args.skip_cover:
        steps.append(("封面生成", lambda: check_cover(cfg)))

    for i, (title, fn) in enumerate(steps, start=1):
        step_no(i, title)
        try:
            fn()
        except (SmokeFailure, *DOMAIN_ERRORS) as exc:
            print(f"\n\033[1;31m不通过：{exc}\033[0m", file=sys.stderr)
            return 1

    print("\n\033[1;32m全部通过。环境可用于生成播客。\033[0m")
    print("提醒：请人工确认音色预设的性别标注与试听结果一致（见设计文档 7.2）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
