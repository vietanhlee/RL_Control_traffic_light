from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, Iterable

from core.config import METRICS_WINDOW_SECONDS
from .measurement_zone import ApproachKey


@dataclass
class GroupMetrics:
    avg_speed: float = 0.0
    avg_density: float = 0.0
    queue_length: int = 0


@dataclass
class DirectionMetrics:
    motorcycle: GroupMetrics = field(default_factory=GroupMetrics)
    car: GroupMetrics = field(default_factory=GroupMetrics)


class MetricsCollector:
    """Collects rolling-window traffic metrics for each incoming approach, separated by group."""

    def __init__(self, window_seconds: float = METRICS_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        # Dicts of Dicts: key -> group -> series
        self.crossing_events: Dict[ApproachKey, Dict[str, Deque[tuple[float, float]]]] = defaultdict(lambda: defaultdict(deque))
        self.density_samples: Dict[ApproachKey, Dict[str, Deque[tuple[float, int]]]] = defaultdict(lambda: defaultdict(deque))
        self.queue_lengths: Dict[ApproachKey, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_crossing(self, key: ApproachKey, group: str, now: float, speed_mps: float) -> None:
        events = self.crossing_events[key][group]
        events.append((now, speed_mps))
        self._trim(events, now)

    def record_density(self, key: ApproachKey, group: str, now: float, count: int) -> None:
        samples = self.density_samples[key][group]
        samples.append((now, count))
        self._trim(samples, now)

    def set_queue_length(self, key: ApproachKey, group: str, queue_count: int) -> None:
        self.queue_lengths[key][group] = max(queue_count, 0)

    def snapshot_direction(self, key: ApproachKey, now: float) -> DirectionMetrics:
        result = DirectionMetrics()
        for group in ["motorcycle", "car"]:
            events = self.crossing_events[key][group]
            samples = self.density_samples[key][group]
            self._trim(events, now)
            self._trim(samples, now)
            
            avg_speed = mean(v for _, v in events) if events else 0.0
            avg_density = mean(v for _, v in samples) if samples else 0.0
            
            gm = GroupMetrics(
                avg_speed=avg_speed,
                avg_density=avg_density,
                queue_length=self.queue_lengths[key].get(group, 0)
            )
            if group == "motorcycle":
                result.motorcycle = gm
            else:
                result.car = gm
                
        return result

    def _trim(self, series: Deque[tuple[float, float | int]], now: float) -> None:
        while series and now - series[0][0] > self.window_seconds:
            series.popleft()

    def keys(self) -> Iterable[ApproachKey]:
        seen = set(self.crossing_events.keys())
        seen.update(self.density_samples.keys())
        seen.update(self.queue_lengths.keys())
        return seen
