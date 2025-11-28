当然可以，我帮你写一个完整、专业、适合你这个图像描述生成项目的 **README.md**，重点介绍**测评流程**。你可以直接放在项目根目录下或 scripts 目录下。

---

# **README.md — 图像描述模型评估说明**

```markdown
# 图像描述生成模型 — 测评流程说明

本文档介绍如何对图像描述生成（Image Captioning）模型进行自动化评估，包括模型推理、BLEU指标计算及结果保存流程。适用于本项目的实验与复现。

---

## 1️⃣ 目录结构概览

```

image_captioning_project/
├── scripts/
│   └── evaluate.py          # 模型评估入口脚本
├── inference/
│   └── caption_generator.py # 图像描述生成接口
├── evaluation/
│   └── bleu.py              # BLEU指标实现
├── dataset/processed/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl           # 测评数据集
└── outputs/results/         # 测评结果保存位置

````

---

## 2️⃣ 测评数据格式

测评数据使用 JSONL 格式，每行包含一条样本：

```json
{
  "image_path": "images/WOMEN-Blouses_Shirts-id_00002543-03_2_side.jpg",
  "image_id": "WOMEN-Blouses_Shirts-id_00002543-03_2_side.jpg",
  "caption": "her tank top has no sleeves, cotton fabric and graphic patterns this woman has neckwear there is an accessory on her wrist"
}
````

* `image_path`：图片文件路径
* `image_id`：图片唯一 ID
* `caption`：对应的参考文本（ground-truth caption）

---

## 3️⃣ 测评流程

### 3.1 准备模型

1. 将训练好的模型权重放入：

```
outputs/checkpoints/
```

例如：

```
model_epoch3_val0.4675.pt
```

2. 确认模型生成接口 `CaptionGenerator` 可正确加载权重。

---

### 3.2 运行评估脚本

从 **项目根目录** 运行：

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_epoch3_val0.4675.pt \
    --data dataset/processed/test.jsonl \
    --device cuda:0
```

**参数说明：**

| 参数             | 说明                                            |
| -------------- | --------------------------------------------- |
| `--checkpoint` | 模型权重文件路径                                      |
| `--data`       | 测评数据 JSONL 文件路径                               |
| `--device`     | 运行设备，例如 `cuda:0` 或 `cpu`                      |
| `--max-len`    | 生成文本最大长度（可选，默认32）                             |
| `--image-size` | 输入图片大小（可选，默认384）                              |
| `--hf-name`    | tokenizer 对应 HuggingFace 模型（可选）               |
| `--spm-model`  | SentencePiece 模型路径（可选）                        |
| `--output`     | 测评结果保存路径（默认 `outputs/results/bleu_test.json`） |

---

### 3.3 评估流程说明

1. **数据加载**：读取 JSONL 文件，获取图片路径和参考 caption。
2. **模型推理**：使用 `CaptionGenerator.generate(image_path)` 对每张图片生成描述文本。
3. **指标计算**：调用 `BLEUMetric` 对生成文本与参考文本计算 BLEU-1 ~ BLEU-4 分数。
4. **结果输出**：打印 BLEU 分数，并保存为 JSON 文件。

示例输出：

```json
{
  "BLEU-1": 0.6532,
  "BLEU-2": 0.4825,
  "BLEU-3": 0.3521,
  "BLEU-4": 0.2456
}
```

---

## 4️⃣ 注意事项

1. **确保 Python 环境正确**

   * 安装依赖：`pip install -r requirements.txt`
   * 确保 VSCode 或终端使用的 Python 环境与安装依赖环境一致

2. **项目根目录运行**

   * 必须在 `fashion_image_captioning_/` 根目录下运行脚本，否则会出现 `ModuleNotFoundError`。

3. **图片路径检查**

   * JSONL 中 `image_path` 必须为实际存在的图片路径，否则生成失败。

---