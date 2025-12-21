"""测试模型三：局部表示+自注意力编码器 → RNN+注意力解码器."""

import sys
sys.path.insert(0, ".")

import torch

from configs.model_configs import (
    AttentionCaptioningModelConfig,
    AttentionCNNEncoderConfig,
    AttentionRNNDecoderConfig,
)
from models.attention_captioning_model import (
    AttentionCaptioningModel,
    build_attention_captioning_model,
)


def test_model3():
    """测试模型三的前向传播和生成功能。"""
    print("=" * 60)
    print("测试模型三：局部表示+自注意力编码器 → RNN+注意力解码器")
    print("=" * 60)

    # 配置
    vocab_size = 8000
    batch_size = 2
    seq_len = 20
    image_size = 224

    # 创建配置
    encoder_config = AttentionCNNEncoderConfig(
        backbone="resnet18",  # 使用较小的backbone进行测试
        embed_dim=512,
        pretrained=True,
        trainable_backbone=False,
        num_heads=8,
        num_attention_layers=2,
        ff_dim=2048,
        dropout=0.1,
    )

    decoder_config = AttentionRNNDecoderConfig(
        vocab_size=vocab_size,
        embed_dim=512,
        hidden_dim=512,
        encoder_dim=512,
        attention_dim=512,
        dropout=0.1,
    )

    config = AttentionCaptioningModelConfig(
        encoder=encoder_config,
        decoder=decoder_config,
    )

    # 构建模型
    print("\n1. 构建模型...")
    model = build_attention_captioning_model(config, vocab_size)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    print("\n2. 测试前向传播（Teacher-Forcing）...")
    images = torch.randn(batch_size, 3, image_size, image_size)
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    model.train()
    logits = model(images, captions)
    print(f"   输入图像形状: {images.shape}")
    print(f"   输入caption形状: {captions.shape}")
    print(f"   输出logits形状: {logits.shape}")
    assert logits.shape == (batch_size, seq_len - 1, vocab_size), "Logits shape mismatch!"
    print("   ✓ 前向传播测试通过")

    # 测试编码器输出
    print("\n3. 测试编码器输出...")
    encoder_output = model.encoder(images)
    print(f"   grid_features形状: {encoder_output['grid_features'].shape}")
    print(f"   global_feature形状: {encoder_output['global_feature'].shape}")
    print(f"   attention_mask形状: {encoder_output['attention_mask'].shape}")
    print("   ✓ 编码器输出测试通过")

    # 测试注意力权重获取
    print("\n4. 测试注意力权重获取...")
    attn_weights = model.get_attention_weights(images, captions)
    num_grids = encoder_output['grid_features'].shape[1]
    print(f"   注意力权重形状: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, seq_len - 1, num_grids), "Attention weights shape mismatch!"
    print("   ✓ 注意力权重测试通过")

    # 测试生成
    print("\n5. 测试自回归生成...")
    model.eval()
    start_token = 1  # BOS
    end_token = 2    # EOS
    max_length = 15

    generated = model.generate(
        images=images,
        start_token=start_token,
        end_token=end_token,
        max_length=max_length,
        temperature=1.0,
        top_k=50,
    )
    print(f"   生成序列形状: {generated.shape}")
    print(f"   生成序列示例: {generated[0].tolist()[:10]}...")
    print("   ✓ 自回归生成测试通过")

    # 测试损失计算
    print("\n6. 测试损失计算...")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    logits = model(images, captions)
    targets = captions[:, 1:]  # 去掉BOS
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"   损失值: {loss.item():.4f}")
    print("   ✓ 损失计算测试通过")

    # 测试反向传播
    print("\n7. 测试反向传播...")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   梯度范数总和: {grad_norm:.4f}")
    print("   ✓ 反向传播测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！模型三实现正确。")
    print("=" * 60)

    return model


if __name__ == "__main__":
    test_model3()
