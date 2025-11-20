# 时尚图像描述 - 模型 1 原型

## 环境准备

```bash
pip install -r requirements.txt
```

## 训练

```bash
python scripts/train.py --epochs 2 --batch_size 8 --min_freq 2
```

训练脚本会：

- 自动根据 `data/textual_descriptions/captions.json` 构建词表，保存至 `outputs/vocab.json`
- 在 `outputs/checkpoints/` 下保存验证集最优模型

## 评估示例

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/model_epoch_2_val_2.3456.pt \
  --vocab outputs/vocab.json
```

输出包含 BLEU 分数，可作为中期报告的定量指标。
