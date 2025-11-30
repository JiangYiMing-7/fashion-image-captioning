from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms as T

from configs.model_configs import CaptioningModelConfig
from data.tokenizer import CaptionTokenizer
from models.captioning_model import build_captioning_model


class CaptionGenerator:
    def __init__(
        self,
        checkpoint_path: str | Path,
        hf_name: Optional[str] = "bert-base-uncased",
        spm_model_path: Optional[str] = None,
        max_len: int = 32,
        image_size: int = 384,
        device: Optional[str] = None,
    ) -> None:
        if not (hf_name or spm_model_path):
            raise ValueError("必须提供 hf_name 或 spm_model_path 之一。")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = CaptionTokenizer(hf_name=hf_name, spm_model_path=spm_model_path, max_len=max_len)

        config = CaptioningModelConfig()
        self.model = build_captioning_model(config, self.tokenizer.vocab_size)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model") or checkpoint.get("model_state") or checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.start_token = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        self.end_token = self.tokenizer.eos_token_id
        self.max_len = max_len

    @torch.no_grad()
    def generate(
        self, 
        image_path: str | Path, 
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> str:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        length = max_length or self.max_len
        token_ids = self.model.generate(
            tensor, 
            self.start_token, 
            self.end_token, 
            length,
            temperature=temperature,
            top_k=top_k,
        )
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())

