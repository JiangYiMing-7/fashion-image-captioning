from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from configs.model_configs import ViTEncoderConfig


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) 编码器，用于提取图像特征。
    
    输出格式：
    - grid_features: [batch_size, num_patches, embed_dim] - 所有patch的特征序列
    - cls_token: [batch_size, embed_dim] - CLS token的全局特征
    
    适配Transformer解码器的注意力机制。
    """

    def __init__(self, config: ViTEncoderConfig) -> None:
        super().__init__()
        self.config = config
        
        # 加载预训练的ViT模型
        if config.model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if config.pretrained else None
            vit_model = models.vit_b_16(weights=weights)
            hidden_dim = 768
        elif config.model_name == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if config.pretrained else None
            vit_model = models.vit_b_32(weights=weights)
            hidden_dim = 768
        elif config.model_name == "vit_l_16":
            weights = models.ViT_L_16_Weights.DEFAULT if config.pretrained else None
            vit_model = models.vit_l_16(weights=weights)
            hidden_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {config.model_name}")
        
        # 提取ViT的核心组件
        self.patch_embed = vit_model.conv_proj
        self.cls_token = vit_model.class_token
        self.pos_embed = vit_model.encoder.pos_embedding
        self.encoder_blocks = vit_model.encoder.layers
        self.ln = vit_model.encoder.ln
        
        # 投影层：将ViT的hidden_dim映射到目标embed_dim
        self.feature_proj = nn.Linear(hidden_dim, config.embed_dim)
        
        # 是否冻结backbone
        if not config.trainable_backbone:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.encoder_blocks.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
    
    def _interpolate_pos_encoding(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        对位置编码进行插值以支持不同分辨率的图像。
        
        Args:
            x: [B, 1+num_patches, hidden_dim] 输入序列（包含CLS token）
            pos_embed: [1, 1+num_patches_pretrain, hidden_dim] 预训练的位置编码
        
        Returns:
            插值后的位置编码
        """
        num_patches = x.shape[1] - 1  # 减去CLS token
        num_pos_tokens = pos_embed.shape[1] - 1  # 减去CLS token的位置编码
        
        if num_patches == num_pos_tokens:
            return pos_embed
        
        # 分离CLS token的位置编码和patch的位置编码
        class_pos_embed = pos_embed[:, 0:1, :]  # [1, 1, hidden_dim]
        patch_pos_embed = pos_embed[:, 1:, :]   # [1, num_pos_tokens, hidden_dim]
        
        # 计算原始和目标的网格大小
        dim = x.shape[-1]
        h0 = w0 = int(num_pos_tokens ** 0.5)
        h = w = int(num_patches ** 0.5)
        
        # 使用双线性插值调整patch位置编码的大小
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        
        # 重新组合CLS token和patch的位置编码
        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)
    
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        前向传播。
        
        Args:
            images: [batch_size, 3, H, W] 输入图像
        
        Returns:
            dict包含:
            - 'grid_features': [batch_size, num_patches, embed_dim] patch序列特征
            - 'cls_token': [batch_size, embed_dim] 全局CLS特征
            - 'attention_mask': [batch_size, num_patches] 注意力掩码（全1）
        """
        batch_size = images.shape[0]
        
        # Step 1: Patch Embedding [B, 3, H, W] -> [B, hidden_dim, num_h, num_w]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # Step 2: 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+num_patches, hidden_dim]
        
        # Step 3: 添加位置编码（支持不同分辨率）
        pos_embed = self._interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        
        # Step 4: Transformer Encoder Blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Step 5: Layer Normalization
        x = self.ln(x)
        
        # Step 6: 分离CLS token和patch features
        cls_token_features = x[:, 0]  # [B, hidden_dim]
        grid_features = x[:, 1:]      # [B, num_patches, hidden_dim]
        
        # Step 7: 投影到目标维度
        cls_token_proj = self.feature_proj(cls_token_features)  # [B, embed_dim]
        grid_features_proj = self.feature_proj(grid_features)   # [B, num_patches, embed_dim]
        
        # Step 8: 创建注意力掩码（全1，表示所有patch都有效）
        num_patches = grid_features_proj.shape[1]
        attention_mask = torch.ones(
            batch_size, num_patches, 
            dtype=torch.long, 
            device=images.device
        )
        
        return {
            "grid_features": grid_features_proj,    # 用于Transformer解码器的交叉注意力
            "cls_token": cls_token_proj,            # 可选：用于初始化解码器
            "attention_mask": attention_mask,       # 用于掩码padding（这里全1）
        }


__all__ = ["ViTEncoder"]
