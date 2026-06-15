from __future__ import annotations

import math
from statistics import mean
from typing import Iterable


MAX_QUEUE_SCALE = 40.0
MAX_DENSITY_SCALE = 10.0
MAX_SPEED_SCALE = 30.0
MAX_DEGREE = 4.0


def _safe_div(value: float, scale: float) -> float:
    if scale <= 1e-9:
        return 0.0
    return max(-5.0, min(5.0, value / scale))


def _direction_tuple(direction: dict[str, float]) -> tuple[float, float, float]:
    queue = float(direction.get("queue_length", 0.0))
    density = float(direction.get("motorcycle_density", 0.0)) + float(direction.get("car_density", 0.0))
    speed = (
        float(direction.get("motorcycle_avg_speed", 0.0))
        + float(direction.get("car_avg_speed", 0.0))
    ) / 2.0
    return queue, density, speed


def _color_flags(color: str | None) -> tuple[float, float, float]:
    normalized = (color or "").upper()
    return (
        1.0 if normalized == "GREEN" else 0.0,
        1.0 if normalized == "YELLOW" else 0.0,
        1.0 if normalized == "RED" else 0.0,
    )


def build_features(observation: dict[str, object], max_directions: int = 4) -> list[float]:
    directions = observation.get("directions", {})
    if not isinstance(directions, dict):
        directions = {}

    time_s = float(observation.get("time", 0.0))
    local_imbalance = float(observation.get("local_imbalance", 0.0))
    global_imbalance = float(observation.get("global_imbalance", 0.0))

    incoming_nodes = observation.get("incoming_nodes", [])
    if not isinstance(incoming_nodes, list):
        incoming_nodes = []

    if not incoming_nodes:
        # Fallback: Sắp xếp theo ID của hướng tăng dần
        incoming_nodes = sorted([int(k) for k in directions.keys() if k.isdigit()])

    parsed: list[tuple[int, float, float, float, tuple[float, float, float]]] = []
    for inc in incoming_nodes:
        raw_key = str(inc)
        payload = directions.get(raw_key, {})
        if not isinstance(payload, dict):
            continue
        queue, density, speed = _direction_tuple(payload)
        light_states = observation.get("light_states", {})
        color = None
        if isinstance(light_states, dict):
            color = light_states.get(raw_key)
            if color is None:
                color = light_states.get(inc)
        parsed.append((inc, queue, density, speed, _color_flags(color if isinstance(color, str) else None)))

    top = parsed[:max_directions]

    queues = [item[1] for item in parsed]
    densities = [item[2] for item in parsed]
    speeds = [item[3] for item in parsed]

    total_queue = sum(queues)
    total_density = sum(densities)
    avg_queue = mean(queues) if queues else 0.0
    avg_density = mean(densities) if densities else 0.0
    avg_speed = mean(speeds) if speeds else 0.0
    min_queue = min(queues) if queues else 0.0
    max_queue = max(queues) if queues else 0.0
    min_speed = min(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    queue_imbalance = sum(abs(q - avg_queue) for q in queues)

    features: list[float] = [
        1.0,
        math.sin(time_s / 30.0),
        math.cos(time_s / 30.0),
        _safe_div(float(len(parsed)), MAX_DEGREE),
        _safe_div(total_queue, MAX_QUEUE_SCALE),
        _safe_div(avg_queue, MAX_QUEUE_SCALE),
        _safe_div(min_queue, MAX_QUEUE_SCALE),
        _safe_div(max_queue, MAX_QUEUE_SCALE),
        _safe_div(queue_imbalance, MAX_QUEUE_SCALE),
        _safe_div(total_density, MAX_DENSITY_SCALE),
        _safe_div(avg_density, MAX_DENSITY_SCALE),
        _safe_div(avg_speed, MAX_SPEED_SCALE),
        _safe_div(min_speed, MAX_SPEED_SCALE),
        _safe_div(max_speed, MAX_SPEED_SCALE),
        _safe_div(local_imbalance, MAX_QUEUE_SCALE),
        _safe_div(global_imbalance, MAX_QUEUE_SCALE * 4.0),
    ]

    for _, queue, density, speed, flags in top:
        features.extend(
            [
                _safe_div(queue, MAX_QUEUE_SCALE),
                _safe_div(density, MAX_DENSITY_SCALE),
                _safe_div(speed, MAX_SPEED_SCALE),
                flags[0],
                flags[1],
                flags[2],
            ]
        )

    missing = max_directions - len(top)
    for _ in range(missing):
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return features

