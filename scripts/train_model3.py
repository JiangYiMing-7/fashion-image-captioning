"""
模型三训练脚本：局部表示+自注意力编码器 → RNN+注意力解码器

使用方法:
    python scripts/train_model3.py --data_root ./dataset --epochs 30

所有可配置参数见下方 argparse 定义。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model_configs import (
    AttentionCaptioningModelConfig,
    AttentionCNNEncoderConfig,
    AttentionRNNDecoderConfig,
)
from data.deepfashion_mm_dataset import DeepFashionMMDataset
from data.tokenizer import CaptionTokenizer
from data.collate import deepfashion_collate_fn
from models.attention_captioning_model import build_attention_captioning_model
from evaluation.bleu import BLEUMetric
from evaluation.cider import CIDErMetric
from evaluation.meteor import METEORMetric
from evaluation.rouge import ROUGEMetric


def set_seed(seed: int) -> None:
    """设置随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练模型三")
    
    # 数据路径
    parser.add_argument("--data_root", type=str, default="./dataset",
                        help="数据集根目录")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="SentencePiece模型路径，默认为 data_root/tokenizer/spm.model")
    parser.add_argument("--hf_tokenizer", type=str, default=None,
                        help="HuggingFace tokenizer名称（如 bert-base-uncased）")
    
    # 模型配置
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="CNN backbone")
    parser.add_argument("--embed_dim", type=int, default=512,
                        help="嵌入维度")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="LSTM隐状态维度")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="自注意力头数")
    parser.add_argument("--num_attention_layers", type=int, default=2,
                        help="自注意力层数")
    parser.add_argument("--ff_dim", type=int, default=2048,
                        help="FFN隐层维度")
    parser.add_argument("--attention_dim", type=int, default=512,
                        help="注意力层维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")
    parser.add_argument("--trainable_backbone", action="store_true",
                        help="是否微调backbone")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="梯度裁剪")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="学习率预热轮数")
    parser.add_argument("--image_size", type=int, default=224,
                        help="输入图像尺寸")
    parser.add_argument("--max_len", type=int, default=30,
                        help="最大序列长度")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers数量")
    parser.add_argument("--output_dir", type=str, default="./outputs/model3",
                        help="输出目录")
    parser.add_argument("--save_every", type=int, default=5,
                        help="每N个epoch保存一次checkpoint")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每N个epoch评估一次")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    
    return parser.parse_args()


def build_dataloaders(args, tokenizer):
    """构建训练和验证数据加载器。"""
    data_root = Path(args.data_root)
    
    train_dataset = DeepFashionMMDataset(
        root=str(data_root),
        ann_file=str(data_root / "processed" / "train.jsonl"),
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_len=args.max_len,
        augment=True,
    )
    
    val_dataset = DeepFashionMMDataset(
        root=str(data_root),
        ann_file=str(data_root / "processed" / "val.jsonl"),
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_len=args.max_len,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=deepfashion_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=deepfashion_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def build_model(args, vocab_size):
    """构建模型。"""
    encoder_config = AttentionCNNEncoderConfig(
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        pretrained=True,
        trainable_backbone=args.trainable_backbone,
        num_heads=args.num_heads,
        num_attention_layers=args.num_attention_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    
    decoder_config = AttentionRNNDecoderConfig(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        encoder_dim=args.embed_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    
    config = AttentionCaptioningModelConfig(
        encoder=encoder_config,
        decoder=decoder_config,
    )
    
    model = build_attention_captioning_model(config, vocab_size)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip):
    """训练一个epoch。"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(images, input_ids)
        
        # 计算损失 (logits: [B, seq_len-1, vocab], targets: [B, seq_len-1])
        targets = input_ids[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证模型。"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        logits = model(images, input_ids)
        targets = input_ids[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_metrics(model, val_loader, tokenizer, device, max_samples=500):
    """计算评估指标。"""
    model.eval()
    
    predictions = []
    references = []
    
    start_token = tokenizer.bos_token_id if tokenizer.bos_token_id else 1
    end_token = tokenizer.eos_token_id if tokenizer.eos_token_id else 2
    
    sample_count = 0
    for batch in tqdm(val_loader, desc="Eval", leave=False):
        if sample_count >= max_samples:
            break
            
        images = batch["image"].to(device)
        captions = batch["caption"]
        
        # 生成
        sequences = model.generate(
            images=images,
            start_token=start_token,
            end_token=end_token,
            max_length=30,
            temperature=1.0,
            top_k=None,  # 贪婪解码
        )
        
        # 解码
        for seq, ref in zip(sequences, captions):
            pred = tokenizer.decode(seq.tolist())
            predictions.append(pred)
            references.append([ref])
            sample_count += 1
            if sample_count >= max_samples:
                break
    
    # 计算指标 (references需要展平为单个字符串列表)
    refs_flat = [r[0] for r in references]
    metrics = {}
    
    try:
        bleu = BLEUMetric()
        bleu_scores = bleu.compute(refs_flat, predictions)
        metrics["BLEU-4"] = bleu_scores.get("BLEU-4", 0.0)
    except Exception as e:
        print(f"BLEU计算失败: {e}")
        metrics["BLEU-4"] = 0.0
    
    try:
        cider = CIDErMetric()
        cider_scores = cider.compute(refs_flat, predictions)
        metrics["CIDEr"] = cider_scores.get("CIDEr", 0.0) if isinstance(cider_scores, dict) else cider_scores
    except Exception as e:
        print(f"CIDEr计算失败: {e}")
        metrics["CIDEr"] = 0.0
    
    try:
        meteor = METEORMetric()
        meteor_scores = meteor.compute(refs_flat, predictions)
        metrics["METEOR"] = meteor_scores.get("METEOR", 0.0) if isinstance(meteor_scores, dict) else meteor_scores
    except Exception as e:
        print(f"METEOR计算失败: {e}")
        metrics["METEOR"] = 0.0
    
    try:
        rouge = ROUGEMetric()
        rouge_scores = rouge.compute(refs_flat, predictions)
        metrics["ROUGE-L"] = rouge_scores.get("ROUGE-L", 0.0) if isinstance(rouge_scores, dict) else rouge_scores
    except Exception as e:
        print(f"ROUGE计算失败: {e}")
        metrics["ROUGE-L"] = 0.0
    
    return metrics, predictions[:5], refs_flat[:5]


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, filename):
    """保存checkpoint。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    
    torch.save(checkpoint, output_dir / filename)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 构建tokenizer
    if args.hf_tokenizer:
        tokenizer = CaptionTokenizer(hf_name=args.hf_tokenizer, max_len=args.max_len)
    else:
        tokenizer_path = args.tokenizer_path or str(Path(args.data_root) / "tokenizer" / "spm.model")
        tokenizer = CaptionTokenizer(spm_model_path=tokenizer_path, max_len=args.max_len)
    
    vocab_size = tokenizer.vocab_size
    print(f"词表大小: {vocab_size}")
    
    # 构建数据加载器
    print("加载数据集...")
    train_loader, val_loader = build_dataloaders(args, tokenizer)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 构建模型
    print("构建模型...")
    model = build_model(args, vocab_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 带warmup的余弦退火调度器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    
    best_val_loss = float("inf")
    best_cider = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # 评估指标
        if epoch % args.eval_every == 0:
            metrics, sample_preds, sample_refs = evaluate_metrics(
                model, val_loader, tokenizer, device
            )
            print(f"  BLEU-4: {metrics['BLEU-4']:.4f}")
            print(f"  CIDEr: {metrics['CIDEr']:.4f}")
            print(f"  METEOR: {metrics['METEOR']:.4f}")
            print(f"  ROUGE-L: {metrics['ROUGE-L']:.4f}")
            
            # 显示样例
            print("\n  样例预测:")
            for i, (pred, ref) in enumerate(zip(sample_preds[:3], sample_refs[:3])):
                print(f"    [{i+1}] 预测: {pred}")
                print(f"        参考: {ref}")
        else:
            metrics = {}
        
        # 记录历史
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            **metrics,
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                output_dir / "checkpoints", "best_loss.pt"
            )
            print("  ✓ 保存最佳损失模型")
        
        if metrics.get("CIDEr", 0) > best_cider:
            best_cider = metrics.get("CIDEr", 0)
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                output_dir / "checkpoints", "best_cider.pt"
            )
            print("  ✓ 保存最佳CIDEr模型")
        
        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                output_dir / "checkpoints", f"epoch_{epoch}.pt"
            )
        
        # 每个epoch后更新scheduler
        for _ in range(len(train_loader)):
            scheduler.step()
    
    # 保存最终模型和训练历史
    save_checkpoint(
        model, optimizer, scheduler, args.epochs, metrics,
        output_dir / "checkpoints", "final.pt"
    )
    
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳CIDEr: {best_cider:.4f}")
    print(f"模型保存至: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
