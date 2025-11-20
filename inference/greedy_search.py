from __future__ import annotations

import torch

from data.vocabulary import Vocabulary
from models.captioning_model import CaptioningModel


@torch.no_grad()
def greedy_decode(
    model: CaptioningModel,
    image: torch.Tensor,
    vocabulary: Vocabulary,
    max_length: int,
) -> str:
    model.eval()
    image = image.unsqueeze(0).to(next(model.parameters()).device)
    generated_ids = model.predict(image, max_length, vocabulary.start_idx, vocabulary.end_idx)
    caption = vocabulary.decode(generated_ids.squeeze(0).tolist())
    return caption

