import json
import argparse
from pathlib import Path
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from inference.caption_generator import CaptionGenerator
from evaluation.bleu import BLEUMetric


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="图像描述模型评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data", type=str, required=True, help="processed/test.jsonl 位置")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--hf-name", type=str, default="bert-base-uncased")
    parser.add_argument("--spm-model", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/results/bleu_test.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # caption 模型
    generator = CaptionGenerator(
        checkpoint_path=args.checkpoint,
        hf_name=args.hf_name,
        spm_model_path=args.spm_model,
        max_len=args.max_len,
        image_size=args.image_size,
        device=args.device,
    )

    dataset = load_jsonl(args.data)

    refs = []
    hyps = []

    print(f"🔍 加载数据 {len(dataset)} 条")

    for idx, item in enumerate(dataset):
        img_path = item["image_path"]
        reference = item["caption"]

        hypothesis = generator.generate('dataset/'+img_path)

        refs.append(reference)
        hyps.append(hypothesis)

        if (idx + 1) % 50 == 0:
            print(f"🖼 处理进度 {idx+1}/{len(dataset)}")

    # BLEU
    bleu = BLEUMetric()
    scores = bleu(refs, hyps)

    print("\n✨ BLEU 结果：")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    # 保存结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"\n📁 结果保存于 → {out_path}")


if __name__ == "__main__":
    main()
