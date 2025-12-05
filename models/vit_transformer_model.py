from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from configs.model_configs import ViTTransformerModelConfig
from models.encoders import ViTEncoder
from models.decoders import TransformerDecoder  # 常 实现


class ViTTransformerModel(nn.Module):
    """
    ViT编码器 + Transformer解码器的完整图像描述模型。
    """
    
    def __init__(self, config: ViTTransformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder(config.encoder)
        self.decoder = TransformerDecoder(config.decoder)
    
    def forward(
        self, 
        images: torch.Tensor, 
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        训练时的前向传播（Teacher-Forcing）。
        
        Args:
            images: [B, 3, H, W] 来自DataLoader的batch["image"]
            captions: [B, seq_len] 来自DataLoader的batch["input_ids"]
        
        Returns:
            logits: [B, seq_len-1, vocab_size]
        """
        # 1. 编码图像
        encoder_output = self.encoder(images)
        
        # 2. 准备Decoder输入（去掉最后一个token）
        tgt_input = captions[:, :-1]
        
        # 3. Decoder前向
        logits = self.decoder(tgt_input, encoder_output)
        
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
        """
        推理时的自回归生成。
        
        Args:
            images: [B, 3, H, W]
            start_token: BOS token ID
            end_token: EOS token ID
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-k采样参数
        
        Returns:
            sequences: [B, generated_len]
        """
        self.eval()
        encoder_output = self.encoder(images)
        sequences = self.decoder.generate(
            encoder_output=encoder_output,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
        return sequences


def build_vit_transformer_model(
    config: ViTTransformerModelConfig, 
    vocab_size: int
) -> ViTTransformerModel:
    """构建ViT-Transformer模型的工厂函数。"""
    config.decoder.vocab_size = vocab_size
    return ViTTransformerModel(config)


__all__ = ["ViTTransformerModel", "build_vit_transformer_model"]
