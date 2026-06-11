from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict
from core.env_loader import load_dotenv

load_dotenv()

# ─── Thông số Mô phỏng cơ bản ──────────────────────────────────────────────────
NUM_INTERSECTIONS = 16          # Tổng số nút giao thông tự động trong mạng lưới (mạng 4x4)
SIMULATION_DT_SECONDS = 0.1     # Bước thời gian vật lý của simulation
GUI_REFRESH_MS = 50             # Tốc độ làm mới giao diện người dùng
METRICS_WINDOW_SECONDS = 30.0   # Cửa sổ thời gian dùng để tính trung bình trượt
DENSITY_SAMPLE_SECONDS = 1.0    # Chu kỳ lấy mẫu mật độ giao thông
DB_FLUSH_SECONDS = 5.0          # Chu kỳ đẩy dữ liệu log xuống cơ sở dữ liệu
SPAWN_INTERVAL_SECONDS = 0.8    # Khoảng thời gian mặc định giữa các lần sinh xe

# ─── Kích thước và Quy tắc Nút giao ──────────────────────────────────────────────
ZONE_LENGTH_METERS = 300.0      # Chiều dài vùng đo lường (m)
BOUNDARY_DISTANCE_METERS = 40.0 # Khoảng cách an toàn tối thiểu từ biên bản đồ
STOP_LINE_DISTANCE_METERS = 8.0 # Khoảng cách vạch dừng xe cách tâm giao lộ
QUEUE_SPEED_THRESHOLD = 7       # Ngưỡng vận tốc (m/s) xe bị coi là kẹt

# ─── Thông số Phương tiện mặc định ──────────────────────────────────────────────
DEFAULT_TARGET_VEHICLE_COUNT = 90
DEFAULT_MIN_SPEED_MPS = 12.0
DEFAULT_MAX_SPEED_MPS = 32.0
DEFAULT_MAX_ACCELERATION = 4.0
DEFAULT_MAX_DECELERATION = 7.0
DEFAULT_SAFE_GAP_METERS = 10.0

# Tỷ lệ hướng rẽ mặc định tại các nút giao (Trái, Đi thẳng, Phải)
DEFAULT_TURN_DISTRIBUTION = {
    "left": 0.1,
    "straight": 0.75,
    "right": 0.15,
}

# ─── Cấu hình Đèn tín hiệu mặc định (khi không chạy RL) ─────────────────────────
DEFAULT_LIGHT_GREEN_SECONDS = 50.0
DEFAULT_LIGHT_YELLOW_SECONDS = 5.0
DEFAULT_LIGHT_RED_SECONDS = 60.0


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
PORT = int(os.getenv("PORT", 8011))


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
