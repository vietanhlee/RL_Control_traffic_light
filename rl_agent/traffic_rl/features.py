from __future__ import annotations

import math
from typing import Any
from .config import MAX_QUEUE_SCALE, MAX_PRESSURE_SCALE


def _safe_div(value: float, scale: float) -> float:
    if scale <= 1e-9:
        return 0.0
    return max(-5.0, min(5.0, value / scale))


def build_features(
    observation: dict[str, Any],
    obs_dict: dict[int, dict[str, Any]] | None = None,
    intersection_id: int | None = None,
    max_directions: int = 4,
) -> list[float]:
    """Trích xuất 12 đặc trưng vật lý bất biến (SOTA) cho Agent.

    Dạng vector: [Q_1, P_1, Q_2, P_2, Q_3, P_3, Q_4, P_4, Phase_onehot]
    Trong đó:
      - Q_p: Hàng chờ của pha p (chuẩn hóa theo MAX_QUEUE_SCALE)
      - P_p: Áp suất của pha p (chuẩn hóa theo MAX_PRESSURE_SCALE)
      - Phase_onehot: One-hot vector của pha hiện tại (4 chiều)
    """
    incoming_nodes = observation.get("incoming_nodes", [])
    if not isinstance(incoming_nodes, list):
        incoming_nodes = []

    directions = observation.get("directions", {})
    if not isinstance(directions, dict):
        directions = {}

    if not incoming_nodes:
        # Fallback: Sắp xếp theo ID của hướng tăng dần
        incoming_nodes = sorted([int(k) for k in directions.keys() if k.isdigit()])

    features: list[float] = []

    for idx in range(max_directions):
        if idx < len(incoming_nodes):
            inc = incoming_nodes[idx]
            raw_key = str(inc)
            payload = directions.get(raw_key, {})
            if not isinstance(payload, dict):
                payload = {}

            # 1. Hàng chờ (Queue)
            queue = float(payload.get("queue_length", 0.0))

            # 2. Áp suất (Pressure) = Vehicles_in - Vehicles_out
            density_in = float(payload.get("motorcycle_density", 0.0)) + float(payload.get("car_density", 0.0))

            # Tính outgoing density của các hướng đi ra khỏi nút giao này
            outgoing_densities = []
            if obs_dict is not None and intersection_id is not None:
                # Các ngã rẽ hợp lệ là tất cả incoming_nodes ngoại trừ chính hướng đi vào inc
                for other_inc in incoming_nodes:
                    if other_inc == inc:
                        continue
                    # Đường từ intersection_id -> other_inc, tức là chiều incoming của other_inc từ intersection_id
                    neighbor_state = obs_dict.get(other_inc)
                    if neighbor_state is not None:
                        neighbor_dirs = neighbor_state.get("directions", {})
                        if isinstance(neighbor_dirs, dict):
                            payload_out = neighbor_dirs.get(str(intersection_id), {})
                            if isinstance(payload_out, dict):
                                density_out = float(payload_out.get("motorcycle_density", 0.0)) + float(payload_out.get("car_density", 0.0))
                                outgoing_densities.append(density_out)

            if outgoing_densities:
                avg_density_out = sum(outgoing_densities) / len(outgoing_densities)
            else:
                avg_density_out = 0.0

            pressure = density_in - avg_density_out

            # Nạp đặc trưng đã chuẩn hóa
            features.append(_safe_div(queue, MAX_QUEUE_SCALE))
            features.append(_safe_div(pressure, MAX_PRESSURE_SCALE))
        else:
            # Padding nếu nút giao có ít hơn 4 hướng vào
            features.extend([0.0, 0.0])

    # 3. Một-nóng (One-hot) biểu diễn pha hiện tại (4 chiều)
    current_phase = int(observation.get("current_phase", 0))
    phase_onehot = [0.0] * 4
    if 0 <= current_phase < 4:
        phase_onehot[current_phase] = 1.0
    features.extend(phase_onehot)

    return features


