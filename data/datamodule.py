"""DataModule-style helper for DeepFashion-MultiModal datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from data.collate import deepfashion_collate_fn
from data.deepfashion_mm_dataset import DeepFashionMMDataset
from data.tokenizer import CaptionTokenizer


class DeepFashionMMDataModule:
    """Construct train/val/test DataLoaders for DeepFashion-MultiModal."""

    def __init__(
        self,
        root: str,
        tokenizer: Optional[CaptionTokenizer],
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 384,
        max_len: int = 28,
        pin_memory: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_len = max_len
        self.pin_memory = pin_memory

    def _ann_path(self, split: str) -> Path:
        path = self.root / "processed" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed annotations: {path}")
        return path

    def _build_dataset(self, split: str, augment: bool) -> DeepFashionMMDataset:
        return DeepFashionMMDataset(
            root=str(self.root),
            ann_file=str(self._ann_path(split)),
            tokenizer=self.tokenizer,
            image_size=self.image_size,
            max_len=self.max_len,
            augment=augment,
        )

    def _build_dataloader(self, split: str, shuffle: bool, drop_last: bool, augment: bool) -> DataLoader:
        dataset = self._build_dataset(split=split, augment=augment)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=deepfashion_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(split="train", shuffle=True, drop_last=True, augment=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(split="val", shuffle=False, drop_last=False, augment=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(split="test", shuffle=False, drop_last=False, augment=False)


__all__ = ["DeepFashionMMDataModule"]

