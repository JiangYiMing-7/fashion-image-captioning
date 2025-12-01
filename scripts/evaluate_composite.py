import json
import argparse
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from inference.caption_generator import CaptionGenerator
from evaluation.composite_metric import CompositeMetric

os.environ['JAVA_TOOL_OPTIONS'] = '--add-opens java.base/java.lang=ALL-UNNAMED'


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
    parser.add_argument("--output", type=str, default="outputs/results/full_evaluation.json")
    parser.add_argument("--metrics", type=str, nargs="+", 
                       choices=["BLEU", "METEOR", "ROUGE", "CIDEr", "SPICE"],
                       default=["BLEU", "METEOR", "ROUGE", "CIDEr", "SPICE"],
                       help="选择要计算的评估指标")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度,越高越随机,0表示贪婪解码")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k 采样,0表示不限制")
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化生成器
    generator = CaptionGenerator(
        checkpoint_path=args.checkpoint,
        hf_name=args.hf_name,
        spm_model_path=args.spm_model,
        max_len=args.max_len,
        image_size=args.image_size,
        device=args.device,
    )

    # 加载数据
    dataset = load_jsonl(args.data)
    print(f"🔍 加载数据 {len(dataset)} 条")

    refs = []
    hyps = []

    # 生成描述
    for idx, item in enumerate(dataset):
        img_path = item["image_path"]
        reference = item["caption"]

        # 注意：这里可能需要根据实际路径调整
        full_img_path = f"dataset/{img_path}" if not img_path.startswith("dataset/") else img_path
        hypothesis = generator.generate(
            full_img_path,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

        refs.append(reference)
        hyps.append(hypothesis)

        if (idx + 1) % 50 == 0:
            print(f"🖼 处理进度 {idx+1}/{len(dataset)}")

    print("📊 开始计算评估指标...")

    # 使用组合评估器计算所有指标
    evaluator = CompositeMetric(metrics=args.metrics)
    scores = evaluator.compute(refs, hyps)

    # 打印结果
    print("\n✨ 评估结果：")
    print("=" * 50)
    
    # 按类别分组显示
    metric_groups = {
        "N-gram匹配指标": ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
        "语义相似度指标": ["METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "高级语义指标": ["CIDEr", "CIDEr-D", "SPICE", "SPICE-F1"]
    }
    
    for group_name, metric_names in metric_groups.items():
        print(f"\n{group_name}:")
        print("-" * 30)
        for metric in metric_names:
            if metric in scores:
                print(f"  {metric}: {scores[metric]:.4f}")

    # 保存结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\n📁 完整评估结果保存于 → {out_path}")


if __name__ == "__main__":
    main()