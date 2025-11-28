# 时尚图像描述数据管道原型

当前仓库提供 DeepFashion-MultiModal 数据清洗、分词与 DataLoader 原型，用于后续图像描述模型训练。

## 环境配置

```bash
pip install -r requirements.txt
```

若使用 GPU，请确保 PyTorch/torchvision 版本与 CUDA 匹配。

## 数据准备

1. 将原始数据放到 `dataset/`（或通过 `--root` 指定其他路径），其中至少包含：
   - `image/`：原始图像
   - `captions/`（或单个 JSON 文件）：服饰描述
2. 运行预处理脚本，生成 `processed/*.jsonl`：

```bash
python data/prepare_deepfashion_mm.py \
  --root ./dataset \
  --caption-dir textual_descriptions
```

脚本支持 `--caption-file`、`--train-ratio` 等参数，可通过 `-h` 查看。

## Tokenizer

- 使用 HuggingFace 模型：在 Demo/脚本中传入 `hf_name="bert-base-uncased"` 等名称。
- 或训练 SentencePiece：

```bash
python -m data.tokenizer train_sentencepiece \
  --root ./dataset \
  --output-dir ./dataset/tokenizer \
  --vocab-size 8000
```

## DataLoader 调试

```bash
python used_demo.py
# 或
python scripts/debug_deepfashion_dataloader.py --root ./dataset
```

示例脚本会：

- 初始化 `CaptionTokenizer`
- 构建 `DeepFashionMMDataModule`
- 打印一个 batch（图像 tensor / tokenized caption / 元信息）

## 常见问题

- **ModuleNotFoundError: data.xxx**：请在仓库根目录运行脚本，确保 `data/__init__.py` 存在。
- **找不到 processed/\*.jsonl**：确认已执行预处理脚本，且 `--root` 指向正确目录。
- **Tokenizer padding 异常**：HuggingFace Tokenizer 需与 `requirements.txt` 中版本一致；若仍报错，可在 Demo 中改用 SentencePiece。

## 模型与训练（方案一：ResNet18 + LSTM）

```bash
python scripts/train.py \
  --root ./dataset \
  --hf-name bert-base-uncased \
  --batch-size 16 \
  --epochs 3
```

流程概览：

- 构建 Tokenizer（可选 HuggingFace / SentencePiece）
- 初始化 `DeepFashionMMDataModule`，加载 train/val 数据
- 构建 `CaptioningModel`（ResNet18 全局特征 + LSTM 解码）
- 使用 `CaptionCrossEntropyLoss` 与 `AdamW` 训练，并在 `outputs/checkpoints/` 保存权重

## 小样本端到端 Demo

如需快速验证“数据加载 → 模型训练”全流程，可仅取少量样本运行：

```bash
python scripts/demo_run.py \
  --root ./dataset \
  --num-train 32 \
  --num-val 8 \
  --epochs 1
```

Demo 会：

- 使用 `train.jsonl`、`val.jsonl` 中的前 N 条记录
- 构造独立 DataLoader，打印训练/验证损失
- 训练完成后可在 `outputs/checkpoints/` 查看保存的权重（便于验证整个 pipeline 正常）

## 推理：单张图片生成描述

```bash
python scripts/infer.py \
  --checkpoint outputs/checkpoints/model_epoch1_val9.5388.pt \
  --image dataset/images/MEN-Denim-id_00000080-01_7_additional.jpg \
  --hf-name bert-base-uncased
```

脚本会自动加载 Tokenizer 和模型，输出类似：

```
生成描述：a man wears a long sleeve sweater with solid color patterns ...
```

若使用 SentencePiece 词表，将 `--hf-name` 留空并设置 `--spm-model` 即可。
