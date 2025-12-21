# 任务3：构建真实背景服饰图像描述数据集

## 快速开始（常）

### 第1步：准备数据

下载 ModaNet 图像，放到 `./modanet/images/` 目录：
- 下载地址：https://github.com/eBay/modanet
- 图像来源：Paperdoll 数据集

### 第2步：获取 API Key

1. 前往 https://dashscope.console.aliyun.com/
2. 注册阿里云账号，开通 DashScope
3. 复制 API Key

### 第3步：生成描述（耗时较长）

```bash
python scripts/generate_captions_qwen.py --image_dir ./modanet/images --output_dir ./dataset_modanet --api_key 你的API_KEY
```

**测试时先用小数据量：**
```bash
python scripts/generate_captions_qwen.py --image_dir ./modanet/images --output_dir ./dataset_modanet --api_key 你的API_KEY --max_images 100
```

### 第4步：训练模型

```bash
python scripts/train_model3.py --data_root ./dataset_modanet --hf_tokenizer bert-base-chinese --epochs 30
```

### 第5步：查看结果

- 训练指标在 `outputs/model3/history.json`
- 最佳模型在 `outputs/model3/checkpoints/best_cider.pt`

---

## 常见问题

**Q: 中断了怎么办？**
加 `--resume` 继续：
```bash
python scripts/generate_captions_qwen.py --resume --image_dir ./modanet/images --output_dir ./dataset_modanet --api_key 你的API_KEY
```

**Q: 不想用API？**
用本地模型（需16GB显存）：
```bash
pip install transformers accelerate
python scripts/generate_captions_qwen.py --image_dir ./modanet/images --output_dir ./dataset_modanet --local_model Qwen/Qwen-VL-Chat
```
