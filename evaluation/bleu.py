from __future__ import annotations

from collections import Counter
from typing import Dict, List

from evaluation.base_metric import BaseMetric


class BLEUMetric(BaseMetric):
    name = "bleu"

    def __init__(self, n_gram: int = 4) -> None:
        self.n_gram = n_gram
        self.predictions: List[str] = []
        self.references: List[str] = []

    def update(self, prediction: str, reference: str) -> None:
        self.predictions.append(prediction)
        self.references.append(reference)

    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {f"{self.name}": 0.0}
        scores = [self._bleu_score(p, r) for p, r in zip(self.predictions, self.references)]
        return {f"{self.name}": sum(scores) / len(scores)}

    def reset(self) -> None:
        self.predictions.clear()
        self.references.clear()

    def _bleu_score(self, prediction: str, reference: str) -> float:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        weights = [1.0 / self.n_gram] * self.n_gram
        precisions = []
        for n in range(1, self.n_gram + 1):
            pred_ngrams = self._ngrams(pred_tokens, n)
            ref_ngrams = self._ngrams(ref_tokens, n)
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = max(sum(pred_ngrams.values()), 1)
            precisions.append(overlap / total)

        geo_mean = 1.0
        for weight, precision in zip(weights, precisions):
            geo_mean *= precision ** weight if precision > 0 else 0.0

        brevity_penalty = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
        return brevity_penalty * geo_mean

    def _ngrams(self, tokens: List[str], n: int) -> Counter:
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i : i + n])] += 1
        return ngrams

