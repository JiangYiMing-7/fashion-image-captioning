"""
完整示例：展示ViT编码器如何与data目录的数据处理配合使用

数据流程：
1. data/prepare_deepfashion_mm.py → 预处理原始数据 → processed/*.jsonl
2. data/tokenizer.py → 文本分词
3. data/deepfashion_mm_dataset.py → 加载图像+文本，应用transforms
4. data/datamodule.py → 封装DataLoader
5. models/encoders/vit_encoder.py → 处理预处理后的图像
6. models/vit_transformer_model.py → 完整模型
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from data.tokenizer import CaptionTokenizer
from data.datamodule import DeepFashionMMDataModule
from models.encoders import ViTEncoder
# from models.vit_transformer_model import ViTTransformerModel  # 等待TransformerDecoder实现
from configs.model_configs import ViTEncoderConfig


def show_data_flow():
    """展示完整的数据流程"""
    
    print("=" * 80)
    print("📊 ViT编码器与data目录数据处理的完整流程")
    print("=" * 80)
    
    # ========== 步骤1: 数据预处理（离线完成） ==========
    print("\n" + "="*80)
    print("步骤1: 数据预处理（离线完成）")
    print("="*80)
    print("""
命令：python data/prepare_deepfashion_mm.py --root ./dataset

功能：
- 读取原始图像和caption
- 清洗文本数据
- 划分train/val/test集
- 生成 processed/train.jsonl, val.jsonl, test.jsonl

输出格式（每行一个JSON）：
{
    "image_path": "images/xxx.jpg",
    "caption": "a woman wears a blue dress...",
    "image_id": "xxx"
}
    """)
    
    # ========== 步骤2: 初始化Tokenizer ==========
    print("\n" + "="*80)
    print("步骤2: 初始化Tokenizer")
    print("="*80)
    
    try:
        tokenizer = CaptionTokenizer(
            hf_name="bert-base-uncased",
            max_len=32,
            lowercase=True
        )
        print(f"✅ Tokenizer初始化成功")
        print(f"   - vocab_size: {tokenizer.vocab_size}")
        print(f"   - pad_token_id: {tokenizer.pad_token_id}")
        print(f"   - max_len: {tokenizer.max_len}")
        
        # 示例：文本编码
        example_text = "a woman wears a blue dress"
        encoded = tokenizer.encode(example_text)
        print(f"\n   示例编码：")
        print(f"   - 原文: '{example_text}'")
        print(f"   - input_ids shape: {encoded['input_ids'].shape}")
        print(f"   - input_ids: {encoded['input_ids'][:10]}...")
    except Exception as e:
        print(f"❌ Tokenizer初始化失败: {e}")
        tokenizer = None
    
    # ========== 步骤3: 初始化DataModule ==========
    print("\n" + "="*80)
    print("步骤3: 初始化DataModule（封装DataLoader）")
    print("="*80)
    
    if tokenizer:
        try:
            datamodule = DeepFashionMMDataModule(
                root="./dataset",
                tokenizer=tokenizer,
                batch_size=4,
                num_workers=0,
                image_size=384,  # ViT需要384x384
                max_len=32
            )
            print(f"✅ DataModule初始化成功")
            print(f"   - batch_size: 4")
            print(f"   - image_size: 384x384")
            print(f"   - max_len: 32")
            
            # 内部使用的Dataset和transforms
            print(f"\n   内部流程：")
            print(f"   1. DeepFashionMMDataset 加载 processed/train.jsonl")
            print(f"   2. 对每个样本：")
            print(f"      a) 读取图像 → PIL Image")
            print(f"      b) 应用transforms：")
            print(f"         - Resize(384)")
            print(f"         - CenterCrop(384)")
            print(f"         - ToTensor()")
            print(f"         - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
            print(f"      c) 文本用Tokenizer编码")
            print(f"   3. collate_fn 将多个样本组成batch")
            
            train_loader = datamodule.train_dataloader()
            has_data = True
        except FileNotFoundError as e:
            print(f"⚠️  数据集未找到: {e}")
            print(f"   需要先运行: python data/prepare_deepfashion_mm.py --root ./dataset")
            has_data = False
        except Exception as e:
            print(f"❌ DataModule初始化失败: {e}")
            has_data = False
    else:
        has_data = False
    
    # ========== 步骤4: DataLoader输出格式 ==========
    print("\n" + "="*80)
    print("步骤4: DataLoader输出的batch格式")
    print("="*80)
    
    if has_data:
        print(f"✅ 从DataLoader获取一个batch：")
        for batch in train_loader:
            print(f"\nbatch = {{")
            print(f"    'image': torch.Tensor,        # shape: {batch['image'].shape}")
            print(f"    'input_ids': torch.Tensor,    # shape: {batch['input_ids'].shape}")
            print(f"    'attention_mask': torch.Tensor, # shape: {batch['attention_mask'].shape}")
            print(f"    'caption': List[str],         # length: {len(batch['caption'])}")
            print(f"    'image_id': List[str],        # length: {len(batch['image_id'])}")
            print(f"    'path': List[str],            # length: {len(batch['path'])}")
            print(f"}}")
            print(f"\n第一个样本：")
            print(f"   - caption: '{batch['caption'][0][:60]}...'")  
            print(f"   - image_id: {batch['image_id'][0]}")
            break
    else:
        print(f"⚠️  无法获取真实数据，使用示例说明：")
        print(f"""
batch = {{
    'image': torch.Tensor([4, 3, 384, 384]),      # 预处理后的图像
    'input_ids': torch.Tensor([4, 32]),           # tokenized文本
    'attention_mask': torch.Tensor([4, 32]),      # 文本掩码
    'caption': ['caption1', 'caption2', ...],     # 原始文本
    'image_id': ['id1', 'id2', ...],              # 图像ID
    'path': ['path1', 'path2', ...],              # 图像路径
}}
}
        """)
    
    # ========== 步骤5: ViT编码器处理 ==========
    print("\n" + "="*80)
    print("步骤5: ViT编码器处理预处理后的图像")
    print("="*80)
    
    config = ViTEncoderConfig(
        model_name="vit_b_16",
        embed_dim=512,
        pretrained=False,
        trainable_backbone=True
    )
    vit_encoder = ViTEncoder(config)
    vit_encoder.eval()
    
    print(f"✅ ViT编码器初始化成功")
    print(f"   - 输入: batch['image'] → [B, 3, 384, 384]")
    print(f"   - 处理: ViT-B/16 提取patch特征")
    print(f"   - 输出: encoder_output dict")
    
    # 使用真实或模拟数据
    if has_data:
        images = batch['image']
        print(f"\n   使用真实数据测试：")
    else:
        images = torch.randn(4, 3, 384, 384)
        print(f"\n   使用模拟数据测试：")
    
    with torch.no_grad():
        encoder_output = vit_encoder(images)
    
    print(f"\n   encoder_output = {{")
    print(f"       'grid_features': {encoder_output['grid_features'].shape},  # 576个patch特征")
    print(f"       'cls_token': {encoder_output['cls_token'].shape},          # 全局特征")
    print(f"       'attention_mask': {encoder_output['attention_mask'].shape}, # 掩码")
    print(f"   }}")
    
    # ========== 步骤6: 完整模型使用 ==========
    print("\n" + "="*80)
    print("步骤6: 完整模型（ViT + Transformer）的使用")
    print("="*80)
    
    print(f"""
训练循环示例代码：

```python
# 初始化
tokenizer = CaptionTokenizer(hf_name="bert-base-uncased", max_len=32)
datamodule = DeepFashionMMDataModule(
    root="./dataset",
    tokenizer=tokenizer,
    batch_size=16,
    image_size=384,
    max_len=32
)
train_loader = datamodule.train_dataloader()

# 创建模型
model_config = ViTTransformerModelConfig()
model = ViTTransformerModel(model_config)
model.to(device)

# 训练循环
for epoch in range(epochs):
    for batch in train_loader:
        # 1. 获取预处理后的数据
        images = batch["image"].to(device)      # [B, 3, 384, 384] ← data/处理过
        input_ids = batch["input_ids"].to(device)  # [B, seq_len] ← tokenizer处理过
        
        # 2. 模型前向传播
        # 内部流程：
        #   a) images → ViT编码器 → encoder_output
        #   b) input_ids + encoder_output → Transformer解码器 → logits
        logits = model(images, input_ids)  # [B, seq_len-1, vocab_size]
        
        # 3. 计算损失
        targets = input_ids[:, 1:]  # 目标序列
        loss = criterion(logits, targets)
        
        # 4. 反向传播
        loss.backward()
        optimizer.step()
```
    """)
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("📋 完整数据流程总结")
    print("="*80)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 原始数据                                                          │
│    - images/xxx.jpg                                                 │
│    - captions.json                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. data/prepare_deepfashion_mm.py                                   │
│    - 清洗数据                                                        │
│    - 划分数据集                                                      │
│    → processed/train.jsonl, val.jsonl, test.jsonl                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. data/deepfashion_mm_dataset.py (Dataset)                         │
│    - 读取jsonl                                                       │
│    - 加载图像 → transforms（Resize, Crop, Normalize）               │
│    - 文本 → tokenizer.encode()                                      │
│    → 返回: {{'image': tensor, 'input_ids': tensor, ...}}              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. data/datamodule.py (DataLoader)                                  │
│    - 批处理（collate_fn）                                            │
│    - 多线程加载                                                      │
│    → batch: {{'image': [B,3,384,384], 'input_ids': [B,seq_len]}}     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. models/encoders/vit_encoder.py                                   │
│    - 接收: batch["image"] [B, 3, 384, 384]                          │
│    - 处理: ViT-B/16 提取特征                                         │
│    → encoder_output: {{'grid_features': [B,576,512], ...}}           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 6. models/vit_transformer_model.py                                  │
│    - forward(images, captions)                                      │
│    - 内部调用: vit_encoder(images) + transformer_decoder()          │
│    → logits: [B, seq_len-1, vocab_size]                             │
└─────────────────────────────────────────────────────────────────────┘

关键点：
✅ ViT编码器的输入 = DataLoader输出的 batch["image"]
✅ batch["image"] 已经过 data/ 目录的完整预处理
✅ 预处理包括：Resize→CenterCrop→ToTensor→Normalize
✅ 这些预处理与ViT预训练时使用的完全一致
    """)
    
    print("\n" + "="*80)
    print("🎯 回答你的问题")
    print("="*80)
    print(f"""
Q: vit_transformer_model.py 的 forward() 函数会使用 data/ 的预处理结果吗？

A: 是的！完整流程是：

1. 训练时，你会这样调用：
   ```python
   for batch in train_loader:  # train_loader 来自 data/datamodule.py
       images = batch["image"]  # 这已经是 data/ 预处理过的！
       captions = batch["input_ids"]
       
       logits = model(images, captions)  # 传入预处理后的数据
   ```

2. model.forward() 内部：
   ```python
   def forward(self, images, captions):
       # images 参数就是 data/ 预处理的结果
       encoder_output = self.encoder(images)  # ViT处理预处理后的图像
       ...
   ```

3. 所以：
   ✅ ViT编码器 100% 使用 data/ 的预处理结果
   ✅ 不需要在 ViT 内部再做预处理
   ✅ data/ 的预处理与 ViT 的要求完全匹配
    """)


if __name__ == "__main__":
    show_data_flow()
