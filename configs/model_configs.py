from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CNNEncoderConfig:
    embed_dim: int = 512
    pretrained: bool = True
    trainable_backbone: bool = False


@dataclass
class RNNDecoderConfig:
    vocab_size: int = 1  # 运行时更新
    embed_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.1


@dataclass
class CaptioningModelConfig:
    encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    decoder: RNNDecoderConfig = field(default_factory=RNNDecoderConfig)

