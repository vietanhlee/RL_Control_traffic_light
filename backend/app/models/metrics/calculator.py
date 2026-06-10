from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict

from .collector import DirectionMetrics


@dataclass
class IntersectionMetrics:
    local_imbalance: float
    avg_queue: float


class MetricsCalculator:
    @staticmethod
    def local_imbalance(direction_metrics: Dict[int, DirectionMetrics]) -> IntersectionMetrics:
        if not direction_metrics:
            return IntersectionMetrics(local_imbalance=0.0, avg_queue=0.0)
        queues = [m.motorcycle.queue_length + m.car.queue_length for m in direction_metrics.values()]
        avg_queue = mean(queues) if queues else 0.0
        imbalance = sum(abs(q - avg_queue) for q in queues)
        return IntersectionMetrics(local_imbalance=imbalance, avg_queue=avg_queue)

    @staticmethod
    def global_imbalance(intersection_imbalances: Dict[int, float]) -> float:
        return sum(intersection_imbalances.values())
