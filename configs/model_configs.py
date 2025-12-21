from __future__ import annotations

from dataclasses import dataclass, field


# ==================== 模型一：CNN + RNN ====================

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


# ==================== 模型三：局部表示+自注意力编码器 → RNN+注意力解码 ====================

@dataclass
class AttentionCNNEncoderConfig:
    """带自注意力的CNN编码器配置"""
    backbone: str = "resnet50"  # resnet18, resnet50, resnet101
    embed_dim: int = 512
    pretrained: bool = True
    trainable_backbone: bool = False
    num_heads: int = 8
    num_attention_layers: int = 2
    ff_dim: int = 2048
    dropout: float = 0.1
    max_grid_size: int = 196  # 14x14 for 224x224 input


@dataclass
class AttentionRNNDecoderConfig:
    """带注意力的RNN解码器配置"""
    vocab_size: int = 1  # 运行时更新
    embed_dim: int = 512
    hidden_dim: int = 512
    encoder_dim: int = 512  # 编码器输出维度
    attention_dim: int = 512  # 注意力层维度
    dropout: float = 0.1


@dataclass
class AttentionCaptioningModelConfig:
    """模型三完整配置"""
    encoder: AttentionCNNEncoderConfig = field(default_factory=AttentionCNNEncoderConfig)
    decoder: AttentionRNNDecoderConfig = field(default_factory=AttentionRNNDecoderConfig)



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
