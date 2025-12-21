"""自注意力CNN编码器 - 提取局部网格特征并加入自注意力机制."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
from torchvision import models

from configs.model_configs import AttentionCNNEncoderConfig


class SelfAttention(nn.Module):
    """多头自注意力模块."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, embed_dim]
            mask: [B, seq_len] 可选的注意力掩码
        Returns:
            [B, seq_len, embed_dim]
        """
        B, L, _ = x.shape

        # 计算Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, heads, L, L]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 加权求和
        out = torch.matmul(attn_probs, V)  # [B, heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        return self.out_proj(out)


class SelfAttentionBlock(nn.Module):
    """自注意力块：自注意力 + FFN + 残差连接 + LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力 + 残差
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)

        # FFN + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class AttentionCNNEncoder(nn.Module):
    """
    带自注意力的CNN编码器。
    
    使用ResNet提取局部网格特征（去掉全局池化层），
    然后通过自注意力层增强特征表示。
    
    输出格式：
    - grid_features: [batch_size, num_grids, embed_dim] - 局部网格特征序列
    - global_feature: [batch_size, embed_dim] - 全局特征（平均池化）
    """

    def __init__(self, config: AttentionCNNEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # 加载预训练ResNet（去掉最后的avgpool和fc层）
        if config.backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet18(weights=weights)
            backbone_dim = 512
        elif config.backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet50(weights=weights)
            backbone_dim = 2048
        elif config.backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet101(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        # 提取特征层（去掉avgpool和fc）
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # 投影层：将backbone特征映射到embed_dim
        self.proj = nn.Linear(backbone_dim, config.embed_dim)

        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_attention_layers)
        ])

        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_grid_size, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 冻结backbone
        if not config.trainable_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        Args:
            images: [batch_size, 3, H, W] 输入图像
        
        Returns:
            dict包含:
            - 'grid_features': [batch_size, num_grids, embed_dim] 局部网格特征
            - 'global_feature': [batch_size, embed_dim] 全局特征
            - 'attention_mask': [batch_size, num_grids] 注意力掩码（全1）
        """
        batch_size = images.shape[0]

        # Step 1: CNN特征提取 [B, 3, H, W] -> [B, C, h, w]
        features = self.feature_extractor(images)  # [B, backbone_dim, h, w]

        # Step 2: 展平空间维度 -> [B, h*w, backbone_dim]
        B, C, h, w = features.shape
        num_grids = h * w
        features = features.flatten(2).transpose(1, 2)  # [B, num_grids, backbone_dim]

        # Step 3: 投影到embed_dim
        features = self.proj(features)  # [B, num_grids, embed_dim]

        # Step 4: 添加位置编码（截断或插值）
        if num_grids <= self.config.max_grid_size:
            pos_embed = self.pos_embed[:, :num_grids, :]
        else:
            # 插值位置编码
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=num_grids,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        features = features + pos_embed

        # Step 5: 自注意力层
        for attn_layer in self.self_attention_layers:
            features = attn_layer(features)

        # Step 6: 计算全局特征（平均池化）
        global_feature = features.mean(dim=1)  # [B, embed_dim]

        # Step 7: 创建注意力掩码
        attention_mask = torch.ones(batch_size, num_grids, dtype=torch.long, device=images.device)

        return {
            "grid_features": features,       # 用于解码器的交叉注意力
            "global_feature": global_feature,  # 可用于初始化解码器隐状态
            "attention_mask": attention_mask,
        }


__all__ = ["AttentionCNNEncoder", "SelfAttention", "SelfAttentionBlock"]
