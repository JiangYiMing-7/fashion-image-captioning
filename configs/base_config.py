from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class BaseConfig:
    """项目通用配置."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs")
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    def resolve(self, *parts: str) -> Path:
        return self.project_root.joinpath(*parts).resolve()

    def to_dict(self) -> Dict[str, Any]:
        cfg_dict = asdict(self)
        cfg_dict["project_root"] = str(cfg_dict["project_root"])
        cfg_dict["data_dir"] = str(cfg_dict["data_dir"])
        cfg_dict["outputs_dir"] = str(cfg_dict["outputs_dir"])
        return cfg_dict

