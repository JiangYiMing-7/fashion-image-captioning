from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 5
    grad_clip: float = 5.0
    val_split: float = 0.1
    max_caption_len: int = 40

