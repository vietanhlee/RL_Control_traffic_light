from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from .environment import load_dotenv

load_dotenv()

NUM_INTERSECTIONS = 16
SIMULATION_DT_SECONDS = 0.1
GUI_REFRESH_MS = 50
METRICS_WINDOW_SECONDS = 30.0
DENSITY_SAMPLE_SECONDS = 1.0
DB_FLUSH_SECONDS = 5.0
SPAWN_INTERVAL_SECONDS = 0.8

ZONE_LENGTH_METERS = 100.0
BOUNDARY_DISTANCE_METERS = 40.0
STOP_LINE_DISTANCE_METERS = 8.0
QUEUE_SPEED_THRESHOLD = 9

DEFAULT_TARGET_VEHICLE_COUNT = 90
DEFAULT_MIN_SPEED_MPS = 12.0
DEFAULT_MAX_SPEED_MPS = 32.0
DEFAULT_MAX_ACCELERATION = 4.0
DEFAULT_MAX_DECELERATION = 7.0
DEFAULT_SAFE_GAP_METERS = 10.0

DEFAULT_TURN_DISTRIBUTION = {
    "left": 0.1,
    "straight": 0.6,
    "right": 0.3,
}

DEFAULT_LIGHT_GREEN_SECONDS = 40.0
DEFAULT_LIGHT_YELLOW_SECONDS = 5.0
DEFAULT_LIGHT_RED_SECONDS = 50.0


def _build_database_url() -> str:
    configured = os.getenv("DATABASE_URL", "").strip()
    if configured:
        return configured

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "traffic_simulator")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    if not password or password in {"YOUR_PASSWORD_HERE", "CHANGE_ME", "changeme"}:
        return ""
    sslmode = os.getenv("DB_SSLMODE", "prefer")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{name}?sslmode={sslmode}"


DATABASE_URL = _build_database_url()

# Mạng lưới đường bất đối xứng phức tạp (16 nút giao) - Tọa độ nhân 3.0
INTERSECTION_LAYOUT: Dict[int, Tuple[float, float]] = {
    0: (240.0, 240.0),    # Ngã 3 biên
    1: (960.0, 270.0),   # Ngã 4
    2: (1740.0, 240.0),   # Ngã 4
    3: (2640.0, 300.0),  # Ngã 3 biên
    
    4: (270.0, 840.0),   # Ngã 4
    5: (990.0, 930.0),  # Ngã 4 Trung tâm Tây
    6: (1830.0, 870.0),  # Ngã 4 Trung tâm Đông
    7: (2580.0, 900.0),  # Ngã 3
    
    8: (300.0, 1440.0),  # Ngã 4
    9: (1020.0, 1410.0),  # Ngã 4
    10: (1800.0, 1470.0), # Ngã 4
    11: (2670.0, 1410.0), # Ngã 3
    
    12: (240.0, 1950.0),  # Spawn/Cụt
    13: (960.0, 1980.0), # Ngã 3
    14: (1740.0, 1920.0), # Ngã 3
    15: (2640.0, 1950.0), # Spawn/Cụt
}

# (start, end, lanes)
# Đã nâng cấp các tuyến đường thành 2, 3, hoặc 4 làn xe
INTERSECTION_CONNECTIONS = [
    # Đại lộ trung tâm huyết mạch (4 làn xe lớn)
    (4, 5, 4),
    (5, 6, 4),
    (6, 7, 4),
    (9, 10, 4), # Nâng cấp thêm 1 đoạn 4 làn
    
    # Đại lộ phụ (3 làn xe)
    (1, 5, 3),
    (5, 9, 3),
    (9, 13, 3),
    (2, 6, 3),
    (6, 10, 3),
    (10, 14, 3),
    (4, 8, 3), # Nâng cấp thêm 1 đoạn 3 làn
    
    # Đường trục thường (2 làn xe)
    (0, 1, 2),
    (1, 2, 2),
    (2, 3, 2),
    (8, 9, 2),
    (10, 11, 2),
    (12, 13, 2),
    (13, 14, 2),
    (14, 15, 2),
    (0, 4, 2),
    (8, 12, 2),
    (3, 7, 2),
    (7, 11, 2),
    (11, 15, 2),
]


@dataclass(frozen=True)
class SimulationConfig:
    target_vehicle_count: int = DEFAULT_TARGET_VEHICLE_COUNT
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS
    max_speed_mps: float = DEFAULT_MAX_SPEED_MPS
    spawn_interval_seconds: float = SPAWN_INTERVAL_SECONDS
    max_acceleration: float = DEFAULT_MAX_ACCELERATION
    max_deceleration: float = DEFAULT_MAX_DECELERATION
    safe_gap_meters: float = DEFAULT_SAFE_GAP_METERS
    turn_distribution: Dict[str, float] | None = None

    def normalized_turn_distribution(self) -> Dict[str, float]:
        distribution = self.turn_distribution or DEFAULT_TURN_DISTRIBUTION
        total = sum(distribution.values()) or 1.0
        return {k: v / total for k, v in distribution.items()}


# Cấu hình Reward & Trọng số RL (Đồng bộ với Agent)
REWARD_OFFSET = 5
WEIGHT_QUEUE = 3.0           # Giữ nguyên của bạn (Rất tốt để ưu tiên giải tỏa)
WEIGHT_IMBALANCE = 4.0       # Giữ nguyên của bạn (Rất tốt cho traffic lệch)
WEIGHT_RED_PRESSURE = 1.5    # Tăng lên để chống "bỏ đói" nhánh ít xe
WEIGHT_SWITCH_PENALTY = 3.0  # Tăng lên để chống giật/đổi đèn liên tục
WEIGHT_SPEED_BONUS = 0.1   # Giữ nguyên của bạn

SCALE_QUEUE = 10.0
SCALE_IMBALANCE = 5.0
SCALE_RED_PRESSURE = 10.0
SCALE_SPEED = 22.0
REWARD_CLIP = 5.0
