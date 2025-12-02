from __future__ import annotations

import torch
from torch import nn


class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        targets = targets.reshape(-1)
        return self.loss_fn(logits, targets)

