"""使用真实图像测试ViT编码器（不需要tokenizer）"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision import transforms
from models.encoders import ViTEncoder
from configs.model_configs import ViTEncoderConfig


def load_and_preprocess_image(image_path, image_size=384):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"❌ 无法加载图像 {image_path}: {e}")
        return None


def test_vit_with_real_images():
    print("=" * 70)
    print("🧪 ViT 编码器 + 真实图像测试")
    print("=" * 70)
    
    # 1. 查找数据集中的图像
    print(f"\n📁 步骤1: 查找数据集图像...")
    dataset_root = Path("./dataset")
    image_dir = dataset_root / "images"
    
    if not image_dir.exists():
        print(f"⚠️  图像目录不存在: {image_dir}")
        print(f"💡 使用随机数据进行测试...")
        use_random_data = True
        image_files = []
    else:
        # 查找前5张图像
        image_files = list(image_dir.glob("*.jpg"))[:5]
        if not image_files:
            image_files = list(image_dir.glob("*.png"))[:5]
        
        if image_files:
            print(f"✅ 找到 {len(image_files)} 张图像")
            for i, img_file in enumerate(image_files, 1):
                print(f"   {i}. {img_file.name}")
            use_random_data = False
        else:
            print(f"⚠️  未找到图像文件")
            print(f"💡 使用随机数据进行测试...")
            use_random_data = True
    
    # 2. 创建ViT编码器
    print(f"\n🔧 步骤2: 创建ViT编码器...")
    config = ViTEncoderConfig(
        model_name="vit_b_16",
        embed_dim=512,
        pretrained=False,      # 不下载预训练权重
        trainable_backbone=True
    )
    encoder = ViTEncoder(config)
    encoder.eval()
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"✅ ViT编码器创建成功")
    print(f"   - 模型: {config.model_name}")
    print(f"   - 输出维度: {config.embed_dim}")
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 3. 准备测试数据
    print(f"\n📦 步骤3: 准备测试数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(f"📍 使用设备: {device}")
    
    if use_random_data:
        # 使用随机数据
        print(f"   使用随机生成的图像数据")
        batch_size = 4
        images = torch.randn(batch_size, 3, 384, 384).to(device)
        image_names = [f"random_image_{i+1}" for i in range(batch_size)]
    else:
        # 使用真实图像
        print(f"   加载真实图像...")
        image_tensors = []
        image_names = []
        
        for img_file in image_files:
            img_tensor = load_and_preprocess_image(img_file)
            if img_tensor is not None:
                image_tensors.append(img_tensor)
                image_names.append(img_file.name)
        
        if not image_tensors:
            print(f"❌ 无法加载任何图像，退出测试")
            return
        
        images = torch.stack(image_tensors).to(device)
        batch_size = len(image_tensors)
        print(f"   ✅ 成功加载 {batch_size} 张图像")
    
    # 4. ViT编码器前向传播
    print(f"\n🚀 步骤4: ViT编码器前向传播...")
    print("-" * 70)
    
    with torch.no_grad():
        encoder_output = encoder(images)
    
    print(f"\n📊 输入:")
    print(f"   - 图像形状: {images.shape}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - 图像列表:")
    for i, name in enumerate(image_names, 1):
        print(f"     {i}. {name}")
    
    print(f"\n📤 ViT编码器输出:")
    print(f"   - grid_features: {encoder_output['grid_features'].shape}")
    print(f"   - cls_token: {encoder_output['cls_token'].shape}")
    print(f"   - attention_mask: {encoder_output['attention_mask'].shape}")
    
    # 5. 验证输出
    print(f"\n✅ 步骤5: 验证输出...")
    expected_patches = (384 // 16) ** 2  # 576
    
    assert encoder_output['grid_features'].shape == (batch_size, expected_patches, config.embed_dim)
    assert encoder_output['cls_token'].shape == (batch_size, config.embed_dim)
    assert encoder_output['attention_mask'].shape == (batch_size, expected_patches)
    print(f"   ✅ 所有维度验证通过")
    
    # 6. 统计分析
    print(f"\n📈 步骤6: 输出统计分析...")
    grid_features = encoder_output['grid_features']
    cls_token = encoder_output['cls_token']
    
    print(f"   grid_features:")
    print(f"   - mean: {grid_features.mean().item():.4f}")
    print(f"   - std: {grid_features.std().item():.4f}")
    print(f"   - min: {grid_features.min().item():.4f}")
    print(f"   - max: {grid_features.max().item():.4f}")
    
    print(f"   cls_token:")
    print(f"   - mean: {cls_token.mean().item():.4f}")
    print(f"   - std: {cls_token.std().item():.4f}")
    print(f"   - min: {cls_token.min().item():.4f}")
    print(f"   - max: {cls_token.max().item():.4f}")
    
    # 7. 逐张图像分析
    print(f"\n🔍 步骤7: 逐张图像特征分析...")
    for i in range(batch_size):
        img_features = grid_features[i]  # [576, 512]
        img_cls = cls_token[i]           # [512]
        
        print(f"\n   图像 {i+1} ({image_names[i]}):")
        print(f"   - grid_features mean: {img_features.mean().item():.4f}")
        print(f"   - cls_token norm: {img_cls.norm().item():.4f}")
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！ViT编码器工作正常！")
    print("=" * 70)
    
    # 8. 总结
    print(f"\n📋 测试总结:")
    print(f"   ✅ ViT编码器: 成功创建并加载")
    print(f"   ✅ 图像处理: 正常处理 {batch_size} 张图像")
    print(f"   ✅ 输出格式: 符合要求")
    print(f"   ✅ 特征提取: 生成 {expected_patches} 个patch特征")
    print(f"\n💡 输出说明:")
    print(f"   - grid_features: 用于Transformer解码器的交叉注意力")
    print(f"   - cls_token: 全局图像特征，可用于初始化解码器")
    print(f"   - attention_mask: 注意力掩码（全1表示所有patch有效）")
    print(f"\n🎯 下一步:")
    print(f"   1. 队友完成Transformer解码器")
    print(f"   2. 集成完整模型进行端到端训练")


if __name__ == "__main__":
    test_vit_with_real_images()
