# 模型三：局部表示+自注意力编码器 → RNN+注意力解码器

## 快速开始

### 1. 确保数据集已准备好
```
dataset/
├── processed/
│   ├── train.jsonl
│   └── val.jsonl
└── images/
```

### 2. 开始训练

**最简单的方式（使用HuggingFace tokenizer，无需额外准备）：**
```bash
python scripts/train_model3.py --hf_tokenizer bert-base-uncased --data_root ./dataset
```

**或者先训练tokenizer再训练：**
```bash
# 训练tokenizer
python -m data.tokenizer train_sentencepiece --root ./dataset --output-dir ./dataset/tokenizer

# 训练模型
python scripts/train_model3.py --data_root ./dataset
```

---

## 常用调参命令

```bash
# 默认配置
python scripts/train_model3.py --hf_tokenizer bert-base-uncased

# 使用ResNet18 (更快，显存更少)
python scripts/train_model3.py --hf_tokenizer bert-base-uncased --backbone resnet18 --batch_size 64

# 使用ResNet101 (更强)
python scripts/train_model3.py --hf_tokenizer bert-base-uncased --backbone resnet101 --batch_size 16

# 微调backbone
python scripts/train_model3.py --hf_tokenizer bert-base-uncased --trainable_backbone --lr 5e-5

# 增加自注意力层数
python scripts/train_model3.py --hf_tokenizer bert-base-uncased --num_attention_layers 4
```

---

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `resnet50` | resnet18/50/101 |
| `--num_attention_layers` | `2` | 自注意力层数 (1-4) |
| `--trainable_backbone` | `False` | 是否微调backbone |
| `--epochs` | `30` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `1e-4` | 学习率 |

查看所有参数：
```bash
python scripts/train_model3.py --help
```

---

## 输出文件

训练完成后在 `outputs/model3/checkpoints/` 目录：
- `best_cider.pt` - 最佳CIDEr模型 ⭐
- `best_loss.pt` - 最佳损失模型
- `final.pt` - 最终模型

---

## 模型架构

```
图像 → ResNet(局部特征) → 自注意力×N → LSTM+Bahdanau注意力 → 输出
```

| Backbone | 总参数量 | 可训练参数 |
|----------|----------|-----------|
| ResNet18 | ~38M | ~27M |
| ResNet50 | ~57M | ~32M |
| ResNet101 | ~79M | ~35M |

---

## 验证模型

```bash
python test_model3.py
```
