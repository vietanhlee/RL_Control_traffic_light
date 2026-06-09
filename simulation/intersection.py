from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from config.constants import (
    DEFAULT_LIGHT_GREEN_SECONDS,
    DEFAULT_LIGHT_RED_SECONDS,
    DEFAULT_LIGHT_YELLOW_SECONDS,
)
from traffic_light.fixed_time import LightColor, SignalTiming, TrafficLightController


@dataclass
class Intersection:
    node_id: int
    incoming_nodes: list[int]
    controller: TrafficLightController

    @classmethod
    def with_defaults(cls, node_id: int, incoming_nodes: Iterable[int]) -> "Intersection":
        incoming = sorted(incoming_nodes)
        cycle = DEFAULT_LIGHT_GREEN_SECONDS + DEFAULT_LIGHT_YELLOW_SECONDS + DEFAULT_LIGHT_RED_SECONDS
        timings: Dict[int, SignalTiming] = {}
        for idx, source in enumerate(incoming):
            offsets = (idx * cycle) / max(len(incoming), 1)
            timings[source] = SignalTiming(
                green=DEFAULT_LIGHT_GREEN_SECONDS,
                yellow=DEFAULT_LIGHT_YELLOW_SECONDS,
                red=DEFAULT_LIGHT_RED_SECONDS,
                offset=offsets,
            )
        return cls(node_id=node_id, incoming_nodes=incoming, controller=TrafficLightController(timings))

    def light_for(self, now: float, incoming_from: int) -> LightColor:
        return self.controller.get_state(now, incoming_from)

    def update_timing(self, incoming_from: int, green: float, yellow: float, red: float) -> None:
        current = self.controller.approach_timings[incoming_from]
        self.controller.update_timing(
            incoming_from,
            SignalTiming(green=green, yellow=yellow, red=red, offset=current.offset),
        )

    def force_switch_phase(self) -> None:
        self.controller.force_phase_switch()

    def timing_snapshot(self) -> Dict[int, SignalTiming]:
        return dict(self.controller.approach_timings)
