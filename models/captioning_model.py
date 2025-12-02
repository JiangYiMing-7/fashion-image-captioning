from __future__ import annotations

from typing import Optional, Tuple

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
        self.hidden_proj = nn.Linear(config.encoder.embed_dim, config.decoder.hidden_dim)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        hidden = self._init_hidden(features)
        logits, _ = self.decoder(captions, hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        start_token: int,
        end_token: Optional[int],
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        features = self.encoder(images)
        hidden = self._init_hidden(features)
        inputs = torch.full((images.size(0), 1), start_token, dtype=torch.long, device=images.device)
        sequences = []
        for _ in range(max_length):
            logits, hidden = self.decoder(inputs, hidden)
            next_logits = logits[:, -1] / temperature
            
            # Top-k 采样
            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_val)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # 采样或贪婪解码
            if temperature > 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
            sequences.append(next_token)
            inputs = next_token
            if end_token is not None and torch.all(next_token.squeeze(-1) == end_token):
                break
        return torch.cat(sequences, dim=1)

    def _init_hidden(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.hidden_proj(features).unsqueeze(0)
        hidden_state = torch.tanh(projected)
        cell_state = torch.tanh(projected)  # 使用图像特征初始化 cell_state
        return hidden_state, cell_state


def build_captioning_model(config: CaptioningModelConfig, vocab_size: int) -> CaptioningModel:
    decoder_cfg: RNNDecoderConfig = config.decoder
    decoder_cfg.vocab_size = vocab_size
    return CaptioningModel(config)
