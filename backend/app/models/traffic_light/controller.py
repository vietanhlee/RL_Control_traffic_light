from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass
class DirectionState:
    queue_length: int
    avg_speed: float
    avg_density: float


@dataclass
class IntersectionState:
    current_phase_source: int
    time: float
    directions: Dict[int, DirectionState]


@dataclass
class SimulationState:
    global_imbalance: float
    intersections: Dict[int, IntersectionState]


class ControllerInterface(ABC):
    """RL-ready controller interface for signal control."""

    @abstractmethod
    def get_action(self, state: SimulationState) -> Dict[int, Dict[int, Dict[str, float]]]:
        """Return a config update map: intersection -> approach -> timing values."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state between episodes."""
