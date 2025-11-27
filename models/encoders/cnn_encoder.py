from __future__ import annotations

from torch import nn
from torchvision import models

from configs.model_configs import CNNEncoderConfig


class CNNEncoder(nn.Module):
    """使用 ResNet18 提取全局图像特征."""

    def __init__(self, config: CNNEncoderConfig) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(backbone.fc.in_features, config.embed_dim)

        if not config.trainable_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.flatten(1)
        return self.proj(features)

