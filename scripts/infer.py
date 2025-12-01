from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import CaptionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单张图片描述生成脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的模型权重路径")
    parser.add_argument("--image", type=str, required=True, help="待描述的图片路径")
    parser.add_argument("--hf-name", type=str, default="bert-base-uncased", help="HuggingFace tokenizer 名称")
    parser.add_argument("--spm-model", type=str, default=None, help="SentencePiece 模型路径（可选）")
    parser.add_argument("--max-len", type=int, default=32, help="生成的最大长度")
    parser.add_argument("--image-size", type=int, default=384, help="推理时的图像尺寸")
    parser.add_argument("--device", type=str, default=None, help="强制指定设备，如 cpu / cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度,越高越随机,0表示贪婪解码")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k 采样,0表示不限制")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = CaptionGenerator(
        checkpoint_path=args.checkpoint,
        hf_name=args.hf_name,
        spm_model_path=args.spm_model,
        max_len=args.max_len,
        image_size=args.image_size,
        device=args.device,
    )
    caption = generator.generate(
        args.image, 
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    print(f"生成描述：{caption}")


if __name__ == "__main__":
    main()
