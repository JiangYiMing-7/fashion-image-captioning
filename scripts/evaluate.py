from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs import BaseConfig, CaptioningModelConfig, TrainingConfig
from data import FashionCaptionDataset, Vocabulary, caption_collate_fn
from evaluation import BLEUMetric
from inference.greedy_search import greedy_decode
from models.captioning_model import build_captioning_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估基础模型。")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=TrainingConfig().max_caption_len)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = BaseConfig()
    captions_path = base_cfg.data_dir / "textual_descriptions" / "captions.json"
    images_dir = base_cfg.data_dir / "images"

    vocab = Vocabulary.load(Path(args.vocab))

    dataset = FashionCaptionDataset(
        image_dir=images_dir,
        captions_file=captions_path,
        vocabulary=vocab,
        max_length=args.max_len,
    )

    subset_size = min(len(dataset), args.samples)
    subset_indices = list(range(subset_size))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    collate = lambda batch: caption_collate_fn(batch, vocab.pad_idx)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate)

    model_config = CaptioningModelConfig()
    model_config.decoder.vocab_size = len(vocab)
    model = build_captioning_model(len(vocab), model_config)
    state = torch.load(args.checkpoint, map_location=base_cfg.device)
    model.load_state_dict(state["model_state"])
    model.to(base_cfg.device)

    metric = BLEUMetric()

    for batch in dataloader:
        image = batch["images"].squeeze(0)
        reference_ids = batch["captions"].squeeze(0)
        reference = vocab.decode(reference_ids.tolist())
        prediction = greedy_decode(model, image, vocab, args.max_len)
        metric.update(prediction, reference)

    print(json.dumps(metric.compute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
