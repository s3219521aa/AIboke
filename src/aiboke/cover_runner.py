"""Z-Image-Turbo 推理入口。

被 cover.ZImageCover 以子进程方式调用，使 diffusers/torch 的重型依赖
不污染主流程，也便于在无 GPU 环境下跳过。

用法：
    python -m aiboke.cover_runner --model <路径> --prompt <提示词> \
        --size 1024 --steps 8 --output <输出png>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo 封面生成")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--prompt", required=True, help="图像生成提示词")
    parser.add_argument("--size", type=int, default=1024, help="输出边长（正方形）")
    parser.add_argument("--steps", type=int, default=8, help="推理步数，Z-Image-Turbo 推荐 8")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    args = parser.parse_args(argv)

    try:
        import torch
        from diffusers import ZImagePipeline
    except ImportError as exc:
        print(f"缺少依赖：{exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    image = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        # Z-Image-Turbo 为蒸馏模型，CFG 固定为 1.0（不可调）
        guidance_scale=1.0,
    ).images[0]

    image.save(output)
    print(f"已保存封面：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
