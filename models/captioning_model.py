from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from configs.model_configs import CaptioningModelConfig, RNNDecoderConfig
from models.decoders import RNNDecoder
from models.encoders import CNNEncoder


class CaptioningModel(nn.Module):
    def __init__(self, config: CaptioningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CNNEncoder(config.encoder)
        self.decoder = RNNDecoder(config.decoder)

        self.init_linear = nn.Linear(config.encoder.embed_dim, config.decoder.hidden_dim)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        hidden = self._init_hidden(features)
        logits, _ = self.decoder(captions, hidden)
        return logits

    def predict(self, images: torch.Tensor, max_length: int, start_idx: int, end_idx: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            features = self.encoder(images)
            hidden = self._init_hidden(features)
            inputs = torch.full((images.size(0), 1), start_idx, dtype=torch.long, device=images.device)
            generated = []
            for _ in range(max_length):
                logits, hidden = self.decoder(inputs, hidden)
                next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                generated.append(next_token)
                inputs = next_token
            return torch.cat(generated, dim=1)

    def _init_hidden(self, features: torch.Tensor):
        hidden_state = torch.tanh(self.init_linear(features)).unsqueeze(0)
        cell_state = torch.zeros_like(hidden_state)
        return hidden_state, cell_state


def build_captioning_model(
    vocab_size: int,
    encoder_cfg: Optional[CaptioningModelConfig] = None,
) -> CaptioningModel:
    config = encoder_cfg or CaptioningModelConfig()
    decoder_cfg: RNNDecoderConfig = config.decoder
    decoder_cfg.vocab_size = vocab_size
    return CaptioningModel(config)

