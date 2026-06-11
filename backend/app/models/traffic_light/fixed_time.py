from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class LightColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class SignalTiming:
    green: float
    yellow: float
    red: float
    offset: float = 0.0

    @property
    def cycle_time(self) -> float:
        return self.green + self.yellow + self.red


class TrafficLightController:
    """Per-approach cycle controller with offset support."""

    def __init__(self, approach_timings: Dict[int, SignalTiming]):
        self.approach_timings = approach_timings
        self.manual_offset = 0.0

    def update_timing(self, approach_from: int, timing: SignalTiming) -> None:
        self.approach_timings[approach_from] = timing

    def force_phase_switch(self) -> None:
        if not self.approach_timings:
            return
        # Lấy cycle của một nhánh bất kỳ (tất cả các nhánh trong 1 ngã tư đều có chung cycle)
        sample_timing = next(iter(self.approach_timings.values()))
        num_phases = len(self.approach_timings)
        shift_amount = sample_timing.cycle_time / max(num_phases, 1)
        self.manual_offset += shift_amount

    def get_current_phase(self, now: float) -> int:
        if not self.approach_timings:
            return 0
        sorted_approaches = sorted(self.approach_timings.keys())
        for idx, approach in enumerate(sorted_approaches):
            state = self.get_state(now, approach)
            if state in (LightColor.GREEN, LightColor.YELLOW):
                return idx
        return 0

    def set_active_phase(self, now: float, phase_index: int) -> bool:
        if not self.approach_timings:
            return False
        sorted_approaches = sorted(self.approach_timings.keys())
        if phase_index < 0 or phase_index >= len(sorted_approaches):
            return False
        
        current_phase = self.get_current_phase(now)
        if current_phase == phase_index:
            return False
            
        target_approach = sorted_approaches[phase_index]
        timing = self.approach_timings[target_approach]
        cycle = max(timing.cycle_time, 1e-6)
        self.manual_offset = (-now - timing.offset) % cycle
        return True

    def get_state(self, now: float, approach_from: int) -> LightColor:
        timing = self.approach_timings[approach_from]
        cycle = max(timing.cycle_time, 1e-6)
        phase_time = (now + timing.offset + self.manual_offset) % cycle
        if phase_time < timing.green:
            return LightColor.GREEN
        if phase_time < timing.green + timing.yellow:
            return LightColor.YELLOW
        return LightColor.RED
