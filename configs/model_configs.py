from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CNNEncoderConfig:
    image_size: int = 224
    embed_dim: int = 256
    pretrained: bool = True
    trainable_backbone: bool = False


@dataclass
class RNNDecoderConfig:
    vocab_size: int = 1  # 训练脚本动态覆盖
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.1


@dataclass
class CaptioningModelConfig:
    encoder: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    decoder: RNNDecoderConfig = field(default_factory=RNNDecoderConfig)

