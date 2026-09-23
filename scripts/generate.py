#!/usr/bin/env python
"""CLI 入口。

用法：
    python scripts/generate.py --input case.json --output-dir out/
    python scripts/generate.py --topic "星巴克国内运营转移" --g1 男 --g2 女 --output-dir out/

输入 case.json：
    {"topic": "...", "speaker_gender1": "男", "speaker_gender2": "女"}

输出（在 --output-dir 下）：
    podcast.mp3  cover.png  script.json

错误处理遵循「操作员入口」的约定：任何失败都只打印一行干净的信息并返回
非零退出码，绝不把 traceback 抛给调用方。输入与配置错误是用法问题（退出码
2），必须在任何模型工作开始之前拦下；生成阶段的失败（退出码 1）区分
「生成失败」与「封面失败且兜底也失败」两类。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiboke.bootstrap import build_pipeline  # noqa: E402
from aiboke.config import load_config  # noqa: E402
from aiboke.cover import CoverError  # noqa: E402
from aiboke.pipeline import PipelineError  # noqa: E402
from aiboke.schema import CaseInput  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成一期 AI 中文双人播客")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="输入 JSON 文件")
    src.add_argument("--topic", type=str, help="直接指定主题")
    p.add_argument("--g1", default="男", help="主播1性别（男/女）")
    p.add_argument("--g2", default="女", help="主播2性别（男/女）")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--config", type=Path, default=_REPO_ROOT / "configs" / "default.yaml")
    p.add_argument("--models-root", type=Path, default=None, help="模型挂载点，覆盖配置")
    p.add_argument("--fallback-cover", action="store_true", help="跳过大模型封面，用纯色兜底")
    return p.parse_args(argv)


def load_case(args: argparse.Namespace) -> CaseInput:
    if args.input is not None:
        path: Path = args.input
        if not path.exists():
            raise FileNotFoundError(f"输入文件不存在: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # 目录、权限、非 UTF-8 编码：都归为「输入文件不可用」，不冒泡成 traceback
            raise ValueError(f"输入文件无法作为 UTF-8 文本读取: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"输入文件的 JSON 无法解析: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"输入文件的顶层必须是 JSON 对象，实际为 {type(data).__name__}"
            )
        try:
            return CaseInput.from_dict(data)
        except TypeError as exc:
            # 例如 gender 写成数组：frozenset 的成员判断会抛 TypeError，
            # 它不是 main 捕获的类型，归一为输入错误
            raise ValueError(f"输入文件字段类型非法: {exc}") from exc
    return CaseInput(topic=args.topic, speaker_gender1=args.g1, speaker_gender2=args.g2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        case = load_case(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"输入错误：{exc}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    models_root = args.models_root or Path(cfg.models_root)

    try:
        pipeline = build_pipeline(cfg, models_root, prefer_fallback_cover=args.fallback_cover)
        episode = pipeline.run(case, args.output_dir)
    except CoverError as exc:
        # 封面仅 10 分，但「主封面与兜底封面都失败」意味着产物缺失。
        # 它与生成阶段失败是两回事，报错必须让操作员一眼分得清。
        print(f"封面生成失败（主封面与兜底封面均失败）：{exc}", file=sys.stderr)
        return 1
    except PipelineError as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI 需要给出干净的失败信息
        print(f"生成失败（未预期错误 {type(exc).__name__}）：{exc}", file=sys.stderr)
        return 1

    print("生成完成：")
    print(f"  音频：{episode.audio_path}")
    print(f"  封面：{episode.cover_path}")
    print(f"  文稿：{episode.script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
