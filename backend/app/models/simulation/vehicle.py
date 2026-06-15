from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class VehicleType(Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"


@dataclass
class Vehicle:
    vehicle_id: int
    path: list[int]
    path_index: int
    progress_m: float
    speed_mps: float
    desired_speed_mps: float
    min_speed_mps: float
    max_speed_mps: float
    max_acceleration: float
    max_deceleration: float
    length_m: float = 4.5
    vehicle_type: VehicleType = VehicleType.CAR
    is_waiting: bool = False
    lane_index: int = 0
    waiting_time_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def current_from(self) -> int:
        return self.path[self.path_index]

    @property
    def current_to(self) -> int:
        return self.path[self.path_index + 1]

    @property
    def remaining_nodes(self) -> int:
        return len(self.path) - self.path_index - 1

    def step_speed(self, dt: float, target_speed: float) -> None:
        if self.speed_mps < target_speed:
            self.speed_mps = min(self.speed_mps + self.max_acceleration * dt, target_speed)
        else:
            self.speed_mps = max(self.speed_mps - self.max_deceleration * dt, target_speed)

    def advance(self, dt: float) -> None:
        self.progress_m += self.speed_mps * dt
