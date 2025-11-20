from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T

from configs.model_configs import CaptioningModelConfig
from data.vocabulary import Vocabulary
from inference.greedy_search import greedy_decode
from models.captioning_model import build_captioning_model


class CaptionGenerator:
    def __init__(
        self,
        checkpoint_path: Path,
        vocabulary_path: Path,
        config: Optional[CaptioningModelConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab = Vocabulary.load(vocabulary_path)
        self.model = build_captioning_model(len(self.vocab), config)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)["model_state"])
        self.model.to(self.device)
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def generate(self, image_path: Path, max_length: int = 40) -> str:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        caption = greedy_decode(self.model, tensor, self.vocab, max_length)
        return caption

