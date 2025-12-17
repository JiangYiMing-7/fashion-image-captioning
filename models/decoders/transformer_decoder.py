from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from configs.model_configs import TransformerDecoderConfig


def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Causal mask for self-attention.
    Returns [sz, sz] with -inf in upper triangle (future positions), 0 elsewhere.
    """
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for image captioning.

    Expected encoder_output format:
      - encoder_output["grid_features"]: [B, N, E]
      - encoder_output["attention_mask"]: [B, N] where 1 = valid, 0 = pad (your ViT returns all-ones)

    Forward (teacher forcing):
      input_ids: [B, T] (already shifted right; e.g., BOS + tokens, without last token)
      returns logits: [B, T, vocab_size]
    """

    def __init__(self, config: TransformerDecoderConfig, pad_token_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = int(pad_token_id)

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.ln = nn.LayerNorm(config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.vocab_size)

        # A small init helps stabilize early training
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, encoder_output: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]
            encoder_output: dict from ViTEncoder

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = input_ids.shape
        device = input_ids.device

        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.config.max_seq_len}")

        # ---- embeddings ----
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
        x = self.token_embed(input_ids) + self.pos_embed(pos)           # [B, T, E]
        x = self.dropout(x)

        # ---- masks ----
        tgt_mask = _generate_square_subsequent_mask(T, device=device)   # [T, T]
        tgt_key_padding_mask = (input_ids == self.pad_token_id)         # [B, T] True=PAD

        memory = encoder_output["grid_features"]                         # [B, N, E]
        enc_attn = encoder_output.get("attention_mask", None)            # [B, N]
        memory_key_padding_mask = None
        if enc_attn is not None:
            # nn.Transformer expects True at PAD positions
            memory_key_padding_mask = (enc_attn == 0)

        # ---- decoder ----
        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        h = self.ln(h)
        logits = self.out_proj(h)  # [B, T, vocab]
        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: dict[str, torch.Tensor],
        start_token: int,
        end_token: Optional[int],
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive decoding.

        Args:
            encoder_output: output dict from ViTEncoder
            start_token: BOS id
            end_token: EOS id (can be None if not used)
            max_length: maximum generated length (including BOS)
            temperature: sampling temperature
            top_k: top-k sampling; if None, greedy decoding when temperature<=0 or top_k=None+temperature==1? (see below)

        Returns:
            sequences: [B, L] with L<=max_length
        """
        device = encoder_output["grid_features"].device
        B = encoder_output["grid_features"].shape[0]

        if max_length > self.config.max_seq_len:
            raise ValueError(f"max_length {max_length} exceeds max_seq_len={self.config.max_seq_len}")

        # [B, 1]
        seq = torch.full((B, 1), int(start_token), dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.forward(seq, encoder_output)          # [B, T, V]
            next_logits = logits[:, -1, :]                      # [B, V]

            # temperature handling
            if temperature is None or temperature <= 0:
                temperature = 1.0
            next_logits = next_logits / float(temperature)

            # top-k filtering
            if top_k is not None and top_k > 0 and top_k < next_logits.shape[-1]:
                topk_vals, topk_idx = torch.topk(next_logits, k=top_k, dim=-1)
                filtered = torch.full_like(next_logits, float("-inf"))
                filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                next_logits = filtered

            # choose next token
            if top_k is None and (temperature == 1.0):
                next_token = torch.argmax(next_logits, dim=-1)  # greedy
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # if EOS reached, keep generating PAD/EOS? we will freeze tokens after EOS
            if end_token is not None:
                next_token = torch.where(finished, torch.full_like(next_token, int(end_token)), next_token)
                finished = finished | (next_token == int(end_token))

            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)

            if end_token is not None and bool(torch.all(finished)):
                break

        return seq
