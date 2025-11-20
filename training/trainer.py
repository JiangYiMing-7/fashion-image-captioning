from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        grad_clip: float = 5.0,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.output_dir = output_dir

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc="训练", leave=False):
            images = batch["images"].to(self.device)
            captions = batch["captions"].to(self.device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(images, inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            running_loss += loss.item()

        if self.scheduler:
            self.scheduler.step()

        return running_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc="验证", leave=False):
            images = batch["images"].to(self.device)
            captions = batch["captions"].to(self.device)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            logits = self.model(images, inputs)
            loss = self.criterion(logits, targets)
            running_loss += loss.item()
        return running_loss / max(len(dataloader), 1)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> Dict[str, float]:
        best_val = float("inf")
        history: Dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else 0.0
            history[f"epoch_{epoch}"] = val_loss

            if val_loader and val_loss < best_val and self.output_dir:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        return history

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.output_dir / f"model_epoch_{epoch}_val_{val_loss:.4f}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"模型已保存至 {ckpt_path}")
