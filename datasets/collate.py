"""Batch collation for DeepFashion-MultiModal datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def deepfashion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad text fields and stack images for a batch."""
    if not batch:
        raise ValueError("Empty batch received.")

    images = torch.stack([sample["image"] for sample in batch], dim=0)

    input_ids_list: List[Optional[torch.Tensor]] = [sample.get("input_ids") for sample in batch]
    attention_mask_list: List[Optional[torch.Tensor]] = [sample.get("attention_mask") for sample in batch]

    has_text = any(tensor is not None for tensor in input_ids_list)
    if has_text:
        lengths = [tensor.shape[0] if tensor is not None else 0 for tensor in input_ids_list]
        max_len = max(lengths)
        pad_token_id = 0
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for idx, tensor in enumerate(input_ids_list):
            if tensor is None:
                continue
            length = tensor.shape[0]
            input_ids[idx, :length] = tensor
            attention_mask[idx, :length] = 1
    else:
        input_ids = None
        attention_mask = None

    collated = {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "caption": [sample["caption"] for sample in batch],
        "image_id": [sample["image_id"] for sample in batch],
        "path": [sample["path"] for sample in batch],
    }
    return collated


__all__ = ["deepfashion_collate_fn"]

