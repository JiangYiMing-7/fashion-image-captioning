from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import optim

from configs import BaseConfig, CaptioningModelConfig, TrainingConfig
from data.datamodule import DeepFashionMMDataModule
from data.tokenizer import CaptionTokenizer
from losses import CaptionCrossEntropyLoss
from models.captioning_model import build_captioning_model
from training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练方案1：CNN + RNN 基线模型")
    parser.add_argument("--root", type=str, default="./dataset", help="数据根目录")
    parser.add_argument("--hf-name", type=str, default="bert-base-uncased", help="HuggingFace tokenizer 名称")
    parser.add_argument("--spm-model", type=str, default=None, help="SentencePiece 模型路径")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().max_epochs)
    parser.add_argument("--lr", type=float, default=TrainingConfig().lr)
    parser.add_argument("--max-len", type=int, default=TrainingConfig().max_len)
    parser.add_argument("--train-backbone", action="store_true", help="是否训练 CNN backbone")
    return parser.parse_args()


def build_tokenizer(args: argparse.Namespace) -> CaptionTokenizer:
    if args.spm_model:
        return CaptionTokenizer(spm_model_path=args.spm_model, max_len=args.max_len)
    return CaptionTokenizer(hf_name=args.hf_name, max_len=args.max_len)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    base_cfg = BaseConfig()
    train_cfg = TrainingConfig(batch_size=args.batch_size, num_workers=args.num_workers, max_epochs=args.epochs, lr=args.lr)
    base_cfg.ensure_dirs()
    set_seed(base_cfg.seed)

    tokenizer = build_tokenizer(args)
    datamodule = DeepFashionMMDataModule(
        root=args.root,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        max_len=args.max_len,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model_cfg = CaptioningModelConfig()
    model_cfg.encoder.trainable_backbone = args.train_backbone
    model = build_captioning_model(model_cfg, tokenizer.vocab_size)
    device = torch.device(base_cfg.device)
    model.to(device)

    criterion = CaptionCrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        grad_clip=train_cfg.grad_clip,
        checkpoint_dir=base_cfg.outputs_dir / "checkpoints",
    )

    trainer.fit(train_loader, val_loader, epochs=train_cfg.max_epochs)


if __name__ == "__main__":
    main()

