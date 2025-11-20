from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from configs import BaseConfig, CaptioningModelConfig, TrainingConfig
from data import FashionCaptionDataset, Vocabulary, caption_collate_fn
from losses import CaptionCrossEntropyLoss
from models.captioning_model import build_captioning_model
from training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练基础图像描述模型（方案1）。")
    parser.add_argument("--epochs", type=int, default=TrainingConfig().max_epochs)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig().lr)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=TrainingConfig().max_caption_len)
    parser.add_argument("--val_split", type=float, default=TrainingConfig().val_split)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_captions(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values())


def main() -> None:
    args = parse_args()
    base_cfg = BaseConfig()
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        val_split=args.val_split,
        max_caption_len=args.max_len,
    )
    set_seed(base_cfg.seed)

    captions_path = base_cfg.data_dir / "textual_descriptions" / "captions.json"
    images_dir = base_cfg.data_dir / "images"

    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(load_captions(captions_path))
    vocab_path = base_cfg.outputs_dir / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)

    dataset = FashionCaptionDataset(
        image_dir=images_dir,
        captions_file=captions_path,
        vocabulary=vocab,
        max_length=train_cfg.max_caption_len,
    )

    val_size = max(1, int(len(dataset) * train_cfg.val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collate = lambda batch: caption_collate_fn(batch, vocab.pad_idx)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)

    model_config = CaptioningModelConfig()
    model_config.decoder.vocab_size = len(vocab)
    model = build_captioning_model(len(vocab), model_config)
    device = torch.device(base_cfg.device)
    model.to(device)

    criterion = CaptionCrossEntropyLoss(ignore_index=vocab.pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        grad_clip=train_cfg.grad_clip,
        output_dir=base_cfg.outputs_dir / "checkpoints",
    )

    trainer.fit(train_loader, val_loader, epochs=train_cfg.max_epochs)


if __name__ == "__main__":
    main()
