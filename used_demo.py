import torch
from pathlib import Path
from datasets.tokenizer import CaptionTokenizer
from datasets.datamodule import DeepFashionMMDataModule

def main():
    # ================= 配置区域 =================
    # 数据集根目录 (请确保你已经运行过 prepare_deepfashion_mm.py 并生成了 processed/ 文件夹)
    root_dir = "./data/DeepFashion-MultiModal"
    
    # 选择 Tokenizer (二选一)
    # 选项 A: 使用 HuggingFace 的 Tokenizer (如 BERT)
    tokenizer_name = "bert-base-uncased"
    spm_model_path = None
    
    # 选项 B: 使用训练的 SentencePiece 模型
    # tokenizer_name = None
    # spm_model_path = "./data/DeepFashion-MultiModal/tokenizer/spm.model"
    # ===========================================

    print(f"Initializing Tokenizer...")
    # 初始化 Tokenizer
    tokenizer = CaptionTokenizer(
        hf_name=tokenizer_name,
        spm_model_path=spm_model_path,
        max_len=32,
        lowercase=True
    )

    print(f"Initializing DataModule from {root_dir}...")
    # 初始化 DataModule
    datamodule = DeepFashionMMDataModule(
        root=root_dir,
        tokenizer=tokenizer,
        batch_size=4,          # 演示用小 batch
        num_workers=0,         # Windows下调试建议设为0，Linux可设为4
        image_size=224,        # 模型需要的图片大小
        max_len=32             # 文本最大长度
    )

    # 获取训练集的 DataLoader
    train_loader = datamodule.train_dataloader()
    print(f"Train DataLoader ready. Total batches: {len(train_loader)}")

    # 模拟训练循环：获取一个 batch 并查看数据结构
    print("\n--- Checking one batch ---")
    for batch in train_loader:
        # 1. 检查图片
        images = batch["image"]
        print(f"[Image] Shape: {images.shape} (Expected: [B, 3, H, W])")
        print(f"[Image] Type: {images.dtype}")

        # 2. 检查文本输入 (Input IDs & Attention Mask)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        if input_ids is not None:
            print(f"[Text] Input IDs Shape: {input_ids.shape} (Expected: [B, max_len])")
            print(f"[Text] Attention Mask Shape: {attention_mask.shape}")
            # 可选：打印解码后的文本确认内容
            # print(f"[Text] Decoded[0]: {tokenizer.tokenizer.decode(input_ids[0])}")
        else:
            print("[Text] No text data (input_ids is None)")

        # 3. 检查元数据 (Raw Captions & Paths)
        raw_captions = batch["caption"]
        print(f"[Meta] First caption: '{raw_captions[0]}'")
        print(f"[Meta] First image path: '{batch['path'][0]}'")
        
        print("\n✅ Data pipeline check passed!")
        break

if __name__ == "__main__":
    main()