from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        grad_clip: float = 5.0,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc="Train", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(images, inputs)
            loss = self.criterion(logits, targets[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            epoch_loss += loss.item()
        if self.scheduler:
            self.scheduler.step()
        return epoch_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc="Val", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            logits = self.model(images, inputs)
            loss = self.criterion(logits, targets)
            epoch_loss += loss.item()
        return epoch_loss / max(len(dataloader), 1)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> None:
        best_val = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else 0.0
            if val_loader and val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)
            print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"model_epoch{epoch}_val{val_loss:.4f}.pt"
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, ckpt_path)


