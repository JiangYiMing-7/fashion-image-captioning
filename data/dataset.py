from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from data.vocabulary import Vocabulary


class FashionCaptionDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        captions_file: str | Path,
        vocabulary: Vocabulary,
        max_length: int,
        transform: Callable | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.transform = transform or self._default_transform()

        with open(captions_file, "r", encoding="utf-8") as f:
            self.captions: Dict[str, str] = json.load(f)

        self.items: List[Tuple[str, str]] = sorted(self.captions.items())

    def _default_transform(self) -> Callable:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name, caption = self.items[idx]
        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"未找到图像：{image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        ids = self.vocabulary.numericalize(caption, self.max_length)
        caption_tensor = torch.tensor(ids, dtype=torch.long)

        return {"image": image, "caption": caption_tensor}


def caption_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    lengths = torch.tensor([sample["caption"].shape[0] for sample in batch], dtype=torch.long)
    max_len = lengths.max().item()
    captions = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)

    for i, sample in enumerate(batch):
        captions[i, : sample["caption"].shape[0]] = sample["caption"]

    return {"images": images, "captions": captions, "lengths": lengths}

