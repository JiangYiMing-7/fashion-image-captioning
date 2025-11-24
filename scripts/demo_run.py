from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from configs import BaseConfig, CaptioningModelConfig
from data.collate import deepfashion_collate_fn
from data.deepfashion_mm_dataset import DeepFashionMMDataset
from data.tokenizer import CaptionTokenizer
from losses import CaptionCrossEntropyLoss
from models.captioning_model import build_captioning_model
from training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小样本端到端验证 Demo")
    parser.add_argument("--root", type=str, default="./dataset")
    parser.add_argument("--hf-name", type=str, default="bert-base-uncased")
    parser.add_argument("--spm-model", type=str, default=None)
    parser.add_argument("--num-train", type=int, default=32, help="参与训练的样本数")
    parser.add_argument("--num-val", type=int, default=8, help="验证样本数")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--train-backbone", action="store_true")
    return parser.parse_args()


def build_tokenizer(args: argparse.Namespace) -> CaptionTokenizer:
    if args.spm_model:
        return CaptionTokenizer(spm_model_path=args.spm_model, max_len=args.max_len)
    return CaptionTokenizer(hf_name=args.hf_name, max_len=args.max_len)


def subset_dataset(dataset: DeepFashionMMDataset, limit: int) -> torch.utils.data.Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    indices = list(range(limit))
    return Subset(dataset, indices)


def main() -> None:
    args = parse_args()
    base_cfg = BaseConfig()
    random.seed(base_cfg.seed)
    torch.manual_seed(base_cfg.seed)

    tokenizer = build_tokenizer(args)
    root = Path(args.root).resolve()
    train_dataset = DeepFashionMMDataset(
        root=str(root),
        ann_file=str((root / "processed" / "train.jsonl")),
        tokenizer=tokenizer,
        max_len=args.max_len,
        augment=True,
    )
    val_dataset = DeepFashionMMDataset(
        root=str(root),
        ann_file=str((root / "processed" / "val.jsonl")),
        tokenizer=tokenizer,
        max_len=args.max_len,
        augment=False,
    )

    train_dataset = subset_dataset(train_dataset, args.num_train)
    val_dataset = subset_dataset(val_dataset, args.num_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=deepfashion_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=deepfashion_collate_fn,
        num_workers=0,
    )

    model_cfg = CaptioningModelConfig()
    model_cfg.encoder.trainable_backbone = args.train_backbone
    model = build_captioning_model(model_cfg, tokenizer.vocab_size)
    device = torch.device(base_cfg.device)
    model.to(device)

    criterion = CaptionCrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=base_cfg.outputs_dir / "checkpoints",
    )

    trainer.fit(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()

