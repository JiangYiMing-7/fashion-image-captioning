from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision import models

from configs.model_configs import CNNEncoderConfig


class CNNEncoder(nn.Module):
    def __init__(self, config: CNNEncoderConfig) -> None:
        super().__init__()
        self.config = config
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        backbone = models.resnet18(weights=weights)

        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        feature_dim = backbone.fc.in_features
        self.proj = nn.Linear(feature_dim, config.embed_dim)

        if not config.trainable_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = features.flatten(1)
        projections = self.proj(features)
        return projections

