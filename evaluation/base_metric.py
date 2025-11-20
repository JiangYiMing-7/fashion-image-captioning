from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseMetric(ABC):
    name: str = "base"

    @abstractmethod
    def update(self, prediction: str, reference: str) -> None: ...

    @abstractmethod
    def compute(self) -> Dict[str, float]: ...

    def reset(self) -> None:
        return None


class MetricCollection:
    def __init__(self, metrics: List[BaseMetric]) -> None:
        self.metrics = metrics

    def update(self, prediction: str, reference: str) -> None:
        for metric in self.metrics:
            metric.update(prediction, reference)

    def compute(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

