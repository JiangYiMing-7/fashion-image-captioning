"""快速测试ViT编码器"""

import torch
from models.encoders import ViTEncoder
from configs.model_configs import ViTEncoderConfig

print("="*60)
print("🧪 ViT 编码器快速测试")
print("="*60)

# 创建配置（不使用预训练权重，更快）
config = ViTEncoderConfig(
    model_name="vit_b_16",
    embed_dim=512,
    pretrained=False,
    trainable_backbone=True
)

print(f"\n✅ 配置: {config.model_name}, embed_dim={config.embed_dim}")

# 创建编码器
encoder = ViTEncoder(config)
print(f"✅ 编码器创建成功")

# 测试前向传播
x = torch.randn(2, 3, 384, 384)
encoder.eval()
with torch.no_grad():
    out = encoder(x)

print(f"\n✅ 输出:")
print(f"   - grid_features: {out['grid_features'].shape}")
print(f"   - cls_token: {out['cls_token'].shape}")
print(f"   - attention_mask: {out['attention_mask'].shape}")

print(f"\n🎉 测试通过！ViT编码器工作正常！")
print("="*60)
