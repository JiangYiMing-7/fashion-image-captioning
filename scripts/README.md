# 图像描述生成模型 — 多指标测评框架说明

本文档介绍图像描述生成（Image Captioning）模型的多维度自动化评估框架，支持 BLEU、METEOR、ROUGE、CIDEr、SPICE 等主流评估指标，提供全面的模型性能分析。

---

## 1️⃣ 目录结构概览

```
image_captioning_project/
├── scripts/
│   ├── evaluate.py              # 基础评估脚本（仅BLEU）
│   └── evaluate_composite.py    # 多指标综合评估脚本
├── inference/
│   └── caption_generator.py     # 图像描述生成接口
├── evaluation/
│   ├── base_metric.py           # 评估指标基类
│   ├── bleu.py                  # BLEU指标实现
│   ├── meteor.py                # METEOR指标实现
│   ├── rouge.py                 # ROUGE指标实现
│   ├── cider.py                 # CIDEr指标实现
│   ├── spice.py                 # SPICE指标实现
│   ├── composite_metric.py      # 组合评估器
│   └── __init__.py              # 模块导出
├── dataset/processed/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl               # 测评数据集
└── outputs/results/             # 测评结果保存位置
```

---

## 2️⃣ 评估指标说明

### 2.1 支持的评估指标

| 指标类别 | 指标名称 | 说明 | 特点 |
|---------|----------|------|------|
| **N-gram匹配** | BLEU-1 ~ BLEU-4 | n-gram精度加权平均 | 经典机器翻译指标，侧重表面形式匹配 |
| **语义相似度** | METEOR | 基于对齐的相似度计算 | 考虑同义词、词干和词序 |
| **面向召回** | ROUGE-1/2/L | 召回率导向评估 | 侧重内容覆盖度，常用于摘要评估 |
| **共识评估** | CIDEr/CIDEr-D | 基于TF-IDF加权的共识 | 专门为图像描述任务设计 |
| **语义命题** | SPICE | 语义图匹配评估 | 评估语义内容一致性，最接近人类判断 |

### 2.2 指标含义解读

- **BLEU-1**：衡量单词级别匹配
- **BLEU-4**：衡量4-gram级别匹配，要求更高的流畅性
- **METEOR**：考虑同义词和词干，更语言学化的评估
- **ROUGE-L**：基于最长公共子序列，评估句子结构相似性
- **CIDEr**：TF-IDF加权，强调信息性和独特性
- **SPICE**：将文本转换为语义图，评估命题匹配

---

## 3️⃣ 快速开始

### 3.1 环境准备

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装评估专用依赖
pip install nltk pycocoevalcap

# 下载NLTK数据（首次运行需要）
python -c "import nltk; nltk.download('punkt')"
```

### 3.2 准备模型和数据

1. 将训练好的模型权重放入：
```
outputs/checkpoints/
```

2. 确认测试数据格式正确：
```json
{
  "image_path": "images/WOMEN-Blouses_Shirts-id_00002543-03_2_side.jpg",
  "image_id": "WOMEN-Blouses_Shirts-id_00002543-03_2_side.jpg",
  "caption": "her tank top has no sleeves, cotton fabric and graphic patterns"
}
```

---

## 4️⃣ 测评流程

### 4.1 完整多指标评估

从 **项目根目录** 运行：

```bash
python scripts/evaluate_composite.py \
    --checkpoint outputs/checkpoints/model_epoch3_val0.4675.pt \
    --data dataset/processed/test.jsonl \
    --device cuda:0
```

### 4.2 选择性指标评估

```bash
# 只评估BLEU和METEOR指标
python scripts/evaluate_composite.py \
    --checkpoint outputs/checkpoints/model_epoch3_val0.4675.pt \
    --data dataset/processed/test.jsonl \
    --metrics BLEU METEOR \
    --device cuda:0

# 评估所有指标（默认）
python scripts/evaluate_composite.py \
    --checkpoint outputs/checkpoints/model_epoch3_val0.4675.pt \
    --data dataset/processed/test.jsonl \
    --metrics BLEU METEOR ROUGE CIDEr SPICE \
    --device cuda:0
```

### 4.3 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型权重文件路径 | **必填** |
| `--data` | 测评数据 JSONL 文件路径 | **必填** |
| `--device` | 运行设备 | `cuda:0` |
| `--max-len` | 生成文本最大长度 | 32 |
| `--image-size` | 输入图片大小 | 384 |
| `--hf-name` | HuggingFace 模型名称 | `bert-base-uncased` |
| `--spm-model` | SentencePiece 模型路径 | `None` |
| `--metrics` | 选择评估指标 | 所有指标 |
| `--output` | 结果保存路径 | `outputs/results/full_evaluation.json` |

---

## 5️⃣ 评估流程详解

### 5.1 数据处理流程

1. **数据加载**：读取 JSONL 文件，解析图像路径和参考描述
2. **模型推理**：使用 `CaptionGenerator` 对每张图片生成描述文本
3. **进度监控**：每50个样本显示处理进度
4. **指标计算**：并行计算所有选定指标
5. **结果汇总**：按类别分组显示评估结果

### 5.2 输出结果示例

```json
{
  "BLEU-1": 0.3249,
  "BLEU-2": 0.2106,
  "BLEU-3": 0.1447,
  "BLEU-4": 0.0980,
  "METEOR": 0.1697,
  "ROUGE-1": 0.3329,
  "ROUGE-2": 0.3329,
  "ROUGE-L": 0.3329,
  "CIDEr": 0.0819,
  "CIDEr-D": 0.0819,
  "SPICE": 0.1234,
  "SPICE-Precision": 0.1345,
  "SPICE-Recall": 0.1123,
  "SPICE-F1": 0.1234
}
```

### 5.3 控制台输出示例

```
✨ 评估结果：
==================================================

N-gram匹配指标:
------------------------------
  BLEU-1: 0.3249
  BLEU-2: 0.2106
  BLEU-3: 0.1447
  BLEU-4: 0.0980

语义相似度指标:
------------------------------
  METEOR: 0.1697
  ROUGE-1: 0.3329
  ROUGE-2: 0.3329
  ROUGE-L: 0.3329

高级语义指标:
------------------------------
  CIDEr: 0.0819
  CIDEr-D: 0.0819
  SPICE: 0.1234
  SPICE-F1: 0.1234

📁 完整评估结果保存于 → outputs/results/full_evaluation.json
```

---

## 6️⃣ 指标解读指南

### 6.1 性能基准参考

| 指标 | 较差 | 一般 | 良好 | 优秀 |
|------|------|------|------|------|
| BLEU-4 | < 0.10 | 0.10-0.20 | 0.20-0.30 | > 0.30 |
| METEOR | < 0.15 | 0.15-0.25 | 0.25-0.35 | > 0.35 |
| CIDEr | < 0.50 | 0.50-1.00 | 1.00-1.50 | > 1.50 |
| SPICE | < 0.10 | 0.10-0.15 | 0.15-0.20 | > 0.20 |

*注：以上基准基于时尚图像描述任务，不同数据集可能有所差异*

### 6.2 指标组合分析

- **BLEU高 + METEOR高**：表面形式和语义都匹配良好
- **BLEU低 + SPICE高**：表达方式不同但语义内容准确
- **CIDEr高**：生成描述信息丰富且符合共识
- **所有指标均衡**：模型综合性能优秀

---