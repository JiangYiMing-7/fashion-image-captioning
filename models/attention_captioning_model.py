"""模型三：局部表示+自注意力编码器 → RNN+注意力解码器."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from configs.model_configs import AttentionCaptioningModelConfig
from models.encoders.attention_cnn_encoder import AttentionCNNEncoder
from models.decoders.attention_rnn_decoder import AttentionRNNDecoder


class AttentionCaptioningModel(nn.Module):
    """
    模型三：局部表示+自注意力编码器 → RNN+注意力解码器
    
    架构：
    - 编码器：CNN提取局部网格特征 + 自注意力增强
    - 解码器：LSTM + Bahdanau注意力机制
    """

    def __init__(self, config: AttentionCaptioningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AttentionCNNEncoder(config.encoder)
        self.decoder = AttentionRNNDecoder(config.decoder)

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练时的前向传播（Teacher-Forcing）。
        
        Args:
            images: [B, 3, H, W] 输入图像
            captions: [B, seq_len] 输入token序列（包含BOS和EOS）
        
        Returns:
            logits: [B, seq_len-1, vocab_size] 预测的logits
        """
        # 1. 编码图像
        encoder_output = self.encoder(images)

        # 2. 准备解码器输入（去掉最后一个token，作为输入）
        decoder_input = captions[:, :-1]

        # 3. 解码器前向传播
        logits, _ = self.decoder(decoder_input, encoder_output)

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
            images: [B, 3, H, W] 输入图像
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

    def get_attention_weights(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）。
        
        Args:
            images: [B, 3, H, W]
            captions: [B, seq_len]
        
        Returns:
            attention_weights: [B, seq_len-1, num_grids]
        """
        encoder_output = self.encoder(images)
        decoder_input = captions[:, :-1]
        _, attention_weights = self.decoder(decoder_input, encoder_output)
        return attention_weights


def build_attention_captioning_model(
    config: AttentionCaptioningModelConfig,
    vocab_size: int,
) -> AttentionCaptioningModel:
    """
    构建模型三的工厂函数。
    
    Args:
        config: 模型配置
        vocab_size: 词表大小
    
    Returns:
        AttentionCaptioningModel实例
    """
    config.decoder.vocab_size = vocab_size
    return AttentionCaptioningModel(config)


__all__ = ["AttentionCaptioningModel", "build_attention_captioning_model"]
