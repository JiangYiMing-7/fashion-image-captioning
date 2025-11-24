from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class BaseConfig:
    """全局路径与运行配置."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "dataset")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs")
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "logs").mkdir(parents=True, exist_ok=True)


