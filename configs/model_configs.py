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



@dataclass
class ViTEncoderConfig:
    """Vision Transformer 编码器配置"""
    model_name: str = "vit_b_16"
    embed_dim: int = 512
    pretrained: bool = True
    trainable_backbone: bool = False


@dataclass
class TransformerDecoderConfig:
    """Transformer 解码器配置"""
    vocab_size: int = 30522
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 128


@dataclass
class ViTTransformerModelConfig:
    """ViT + Transformer 完整模型配置"""
    encoder: ViTEncoderConfig = field(default_factory=ViTEncoderConfig)
    decoder: TransformerDecoderConfig = field(default_factory=TransformerDecoderConfig)