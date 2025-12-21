"""带注意力机制的RNN解码器."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from configs.model_configs import AttentionRNNDecoderConfig


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) 注意力机制."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重和上下文向量。
        
        Args:
            encoder_features: [B, num_grids, encoder_dim] 编码器输出
            decoder_hidden: [B, decoder_dim] 解码器隐状态
            mask: [B, num_grids] 可选的注意力掩码
        
        Returns:
            context: [B, encoder_dim] 上下文向量
            attn_weights: [B, num_grids] 注意力权重
        """
        # 投影编码器特征和解码器隐状态
        enc_proj = self.encoder_proj(encoder_features)  # [B, num_grids, attention_dim]
        dec_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # [B, 1, attention_dim]

        # 计算注意力分数
        energy = torch.tanh(enc_proj + dec_proj)  # [B, num_grids, attention_dim]
        attn_scores = self.v(energy).squeeze(-1)  # [B, num_grids]

        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_grids]

        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_features).squeeze(1)  # [B, encoder_dim]

        return context, attn_weights


class AttentionRNNDecoder(nn.Module):
    """
    带注意力机制的RNN解码器。
    
    结构：
    - 词嵌入层
    - LSTM单元
    - Bahdanau注意力机制
    - 输出投影层
    """

    def __init__(self, config: AttentionRNNDecoderConfig) -> None:
        super().__init__()
        self.config = config

        # 词嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # 注意力机制
        self.attention = BahdanauAttention(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.hidden_dim,
            attention_dim=config.attention_dim,
        )

        # LSTM单元（输入 = 词嵌入 + 上下文向量）
        self.lstm = nn.LSTMCell(
            input_size=config.embed_dim + config.encoder_dim,
            hidden_size=config.hidden_dim,
        )

        # 用于初始化隐状态的投影层
        self.init_h = nn.Linear(config.encoder_dim, config.hidden_dim)
        self.init_c = nn.Linear(config.encoder_dim, config.hidden_dim)

        # 输出层
        self.fc = nn.Linear(config.hidden_dim + config.encoder_dim + config.embed_dim, config.vocab_size)

    def init_hidden_state(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用编码器特征的均值初始化LSTM隐状态。
        
        Args:
            encoder_features: [B, num_grids, encoder_dim]
        
        Returns:
            (h0, c0): 每个都是 [B, hidden_dim]
        """
        mean_features = encoder_features.mean(dim=1)  # [B, encoder_dim]
        h0 = torch.tanh(self.init_h(mean_features))
        c0 = torch.tanh(self.init_c(mean_features))
        return h0, c0

    def forward(
        self,
        captions: torch.Tensor,
        encoder_output: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播（Teacher-Forcing）。
        
        Args:
            captions: [B, seq_len] 输入token序列
            encoder_output: 编码器输出字典，包含 'grid_features' 和 'attention_mask'
        
        Returns:
            logits: [B, seq_len, vocab_size]
            all_attn_weights: [B, seq_len, num_grids] 所有时间步的注意力权重
        """
        encoder_features = encoder_output["grid_features"]  # [B, num_grids, encoder_dim]
        attention_mask = encoder_output.get("attention_mask", None)

        batch_size, seq_len = captions.shape

        # 初始化隐状态
        h, c = self.init_hidden_state(encoder_features)

        # 词嵌入
        embeddings = self.dropout(self.embedding(captions))  # [B, seq_len, embed_dim]

        # 存储输出
        outputs = []
        all_attn_weights = []

        for t in range(seq_len):
            # 当前时间步的词嵌入
            embed_t = embeddings[:, t, :]  # [B, embed_dim]

            # 计算注意力
            context, attn_weights = self.attention(encoder_features, h, attention_mask)
            all_attn_weights.append(attn_weights)

            # LSTM输入 = 词嵌入 + 上下文
            lstm_input = torch.cat([embed_t, context], dim=1)  # [B, embed_dim + encoder_dim]
            h, c = self.lstm(lstm_input, (h, c))

            # 输出层输入 = 隐状态 + 上下文 + 词嵌入
            output_input = torch.cat([h, context, embed_t], dim=1)
            logits_t = self.fc(self.dropout(output_input))  # [B, vocab_size]
            outputs.append(logits_t)

        # 堆叠所有时间步
        logits = torch.stack(outputs, dim=1)  # [B, seq_len, vocab_size]
        all_attn_weights = torch.stack(all_attn_weights, dim=1)  # [B, seq_len, num_grids]

        return logits, all_attn_weights

    @torch.no_grad()
    def generate(
        self,
        encoder_output: Dict[str, torch.Tensor],
        start_token: int,
        end_token: Optional[int],
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        推理时的自回归生成。
        
        Args:
            encoder_output: 编码器输出字典
            start_token: BOS token ID
            end_token: EOS token ID
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-k采样参数
        
        Returns:
            sequences: [B, generated_len]
        """
        encoder_features = encoder_output["grid_features"]
        attention_mask = encoder_output.get("attention_mask", None)
        batch_size = encoder_features.shape[0]
        device = encoder_features.device

        # 初始化
        h, c = self.init_hidden_state(encoder_features)
        current_token = torch.full((batch_size,), start_token, dtype=torch.long, device=device)
        sequences = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # 词嵌入
            embed_t = self.embedding(current_token)  # [B, embed_dim]

            # 计算注意力
            context, _ = self.attention(encoder_features, h, attention_mask)

            # LSTM
            lstm_input = torch.cat([embed_t, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            # 输出
            output_input = torch.cat([h, context, embed_t], dim=1)
            logits = self.fc(output_input)  # [B, vocab_size]

            # 采样
            next_logits = logits / temperature

            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_val)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            if temperature > 0:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            sequences.append(next_token)
            current_token = next_token

            # 检查是否结束
            if end_token is not None:
                finished = finished | (next_token == end_token)
                if finished.all():
                    break

        return torch.stack(sequences, dim=1)  # [B, generated_len]


__all__ = ["AttentionRNNDecoder", "BahdanauAttention"]
