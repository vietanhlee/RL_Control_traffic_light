from __future__ import annotations

from dataclasses import dataclass


ApproachKey = tuple[int, int]  # (intersection_id, incoming_from)


@dataclass(frozen=True)
class MeasurementZone:
    intersection_id: int
    incoming_from: int
    zone_length_m: float
    boundary_distance_m: float

    def contains_distance(self, distance_to_intersection_m: float) -> bool:
        return 0.0 <= distance_to_intersection_m <= self.zone_length_m

    def crossed_boundary(self, previous_distance_m: float, current_distance_m: float) -> bool:
        return previous_distance_m > self.boundary_distance_m >= current_distance_m
