"""Smoke-test script for DeepFashion-MM DataModule."""

from __future__ import annotations

import argparse
from itertools import islice
from typing import Optional

from datasets.datamodule import DeepFashionMMDataModule
from datasets.tokenizer import CaptionTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./data/DeepFashion-MultiModal")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-len", type=int, default=28)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--hf-tokenizer-name", type=str, default=None)
    return parser.parse_args()


def build_tokenizer(args: argparse.Namespace) -> Optional[CaptionTokenizer]:
    if args.tokenizer_path:
        return CaptionTokenizer(spm_model_path=args.tokenizer_path, max_len=args.max_len)
    if args.hf_tokenizer_name:
        return CaptionTokenizer(hf_name=args.hf_tokenizer_name, max_len=args.max_len)
    return None


def describe_batch(name: str, batch: dict) -> None:
    print(f"\n=== {name.upper()} BATCH ===")
    print("image:", tuple(batch["image"].shape))
    if batch["input_ids"] is not None:
        print("input_ids:", tuple(batch["input_ids"].shape))
    else:
        print("input_ids: None")
    print("captions:", list(islice(batch["caption"], 3)))
    print("image_ids:", list(islice(batch["image_id"], 3)))


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args)
    datamodule = DeepFashionMMDataModule(
        root=args.root,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_len=args.max_len,
    )

    loaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    for name, loader in loaders.items():
        try:
            batch = next(iter(loader))
        except StopIteration:
            print(f"No samples available for {name}.")
            continue
        describe_batch(name, batch)


if __name__ == "__main__":
    main()

