"""Dataset definition for DeepFashion-MultiModal captions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.tokenizer import CaptionTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepFashionMMDataset(Dataset):
    """Unified dataset returning normalized images and tokenized captions."""

    def __init__(
        self,
        root: str,
        ann_file: str,
        tokenizer: Optional[CaptionTokenizer] = None,
        image_size: int = 384,
        max_len: int = 28,
        augment: bool = False,
        lowercase: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.ann_file = Path(ann_file)
        if not self.ann_file.is_absolute():
            self.ann_file = self.root / self.ann_file
        if not self.ann_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.tokenizer = tokenizer
        if self.tokenizer:
            self.tokenizer.max_len = max_len
            self.tokenizer.lowercase = lowercase

        self.annotations: List[Dict[str, Any]] = self._load_annotations()
        self.transform = self._build_transform(image_size=image_size, augment=augment)

    def _load_annotations(self) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        with self.ann_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                annotations.append(json.loads(line))
        if not annotations:
            raise RuntimeError(f"No annotations loaded from {self.ann_file}")
        return annotations

    def _build_transform(self, image_size: int, augment: bool) -> transforms.Compose:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        if augment:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.annotations[idx]
        image_rel = Path(record["image_path"])
        image_path = image_rel if image_rel.is_absolute() else self.root / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_tensor = self.transform(image)

        caption = record.get("caption", "")
        tokenized = self.tokenizer.encode(caption) if self.tokenizer else None
        input_ids = tokenized["input_ids"] if tokenized else None
        attention_mask = tokenized["attention_mask"] if tokenized else None
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else None

        sample = {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "image_id": str(record.get("image_id", "")),
            "path": str(image_path),
            "pad_token_id": pad_token_id,
        }
        return sample

