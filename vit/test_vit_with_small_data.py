"""使用小规模真实数据测试ViT编码器"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.encoders import ViTEncoder
from configs.model_configs import ViTEncoderConfig
from data.tokenizer import CaptionTokenizer
from data.datamodule import DeepFashionMMDataModule


def test_vit_with_small_data():
    print("=" * 70)
    print("🧪 ViT 编码器 + 小规模真实数据测试")
    print("=" * 70)
    
    # 1. 初始化Tokenizer
    print(f"\n📝 步骤1: 初始化Tokenizer...")
    try:
        tokenizer = CaptionTokenizer(
            hf_name="bert-base-uncased",
            max_len=32,
            lowercase=True
        )
        print(f"✅ Tokenizer加载成功 (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return
    
    # 2. 初始化DataModule
    print(f"\n📦 步骤2: 初始化DataModule...")
    try:
        datamodule = DeepFashionMMDataModule(
            root="./dataset",
            tokenizer=tokenizer,
            batch_size=4,          # 小batch size
            num_workers=0,         # Windows下设为0
            image_size=384,        # ViT需要384
            max_len=32
        )
        train_loader = datamodule.train_dataloader()
        print(f"✅ DataLoader加载成功")
        print(f"   - Total batches: {len(train_loader)}")
        print(f"   - Batch size: 4")
        print(f"   - Image size: 384x384")
    except FileNotFoundError as e:
        print(f"⚠️  数据集未找到: {e}")
        print(f"\n💡 提示: 请先运行数据预处理:")
        print(f"   python data/prepare_deepfashion_mm.py --root ./dataset")
        return
    except Exception as e:
        print(f"❌ DataLoader初始化失败: {e}")
        return
    
    # 3. 创建ViT编码器
    print(f"\n🔧 步骤3: 创建ViT编码器...")
    config = ViTEncoderConfig(
        model_name="vit_b_16",
        embed_dim=512,
        pretrained=False,      # 不下载预训练权重（更快）
        trainable_backbone=True
    )
    encoder = ViTEncoder(config)
    encoder.eval()
    
    # 统计参数
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"✅ ViT编码器创建成功")
    print(f"   - 模型: {config.model_name}")
    print(f"   - 输出维度: {config.embed_dim}")
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 4. 测试前3个batch
    print(f"\n🚀 步骤4: 测试真实数据...")
    print("-" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(f"📍 使用设备: {device}")
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # 只测试前3个batch
            break
        
        print(f"\n📦 Batch {batch_idx + 1}:")
        
        # 获取数据
        images = batch["image"].to(device)
        input_ids = batch["input_ids"]
        captions = batch["caption"]
        
        print(f"   输入:")
        print(f"   - 图像形状: {images.shape}")
        print(f"   - 文本形状: {input_ids.shape}")
        print(f"   - 第一个caption: '{captions[0][:60]}...'")
        
        # ViT编码器前向传播
        with torch.no_grad():
            encoder_output = encoder(images)
        
        print(f"   ViT编码器输出:")
        print(f"   - grid_features: {encoder_output['grid_features'].shape}")
        print(f"   - cls_token: {encoder_output['cls_token'].shape}")
        print(f"   - attention_mask: {encoder_output['attention_mask'].shape}")
        
        # 验证维度
        batch_size = images.shape[0]
        expected_patches = (384 // 16) ** 2  # 576
        
        assert encoder_output['grid_features'].shape == (batch_size, expected_patches, config.embed_dim)
        assert encoder_output['cls_token'].shape == (batch_size, config.embed_dim)
        assert encoder_output['attention_mask'].shape == (batch_size, expected_patches)
        
        print(f"   ✅ 维度验证通过")
        
        # 检查输出值的范围
        grid_mean = encoder_output['grid_features'].mean().item()
        grid_std = encoder_output['grid_features'].std().item()
        print(f"   📊 grid_features统计: mean={grid_mean:.4f}, std={grid_std:.4f}")
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！ViT编码器与真实数据集成成功！")
    print("=" * 70)
    
    # 5. 总结
    print(f"\n📋 测试总结:")
    print(f"   ✅ Tokenizer: 正常工作")
    print(f"   ✅ DataLoader: 正常加载数据")
    print(f"   ✅ ViT编码器: 正常处理图像")
    print(f"   ✅ 输出格式: 符合Transformer解码器要求")
    print(f"\n💡 下一步:")
    print(f"   1. 等待队友完成Transformer解码器")
    print(f"   2. 集成完整的ViT-Transformer模型")
    print(f"   3. 开始端到端训练")


if __name__ == "__main__":
    test_vit_with_small_data()
