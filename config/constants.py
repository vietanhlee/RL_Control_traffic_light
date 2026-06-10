from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from .environment import load_dotenv

load_dotenv()

# ─── Thông số Mô phỏng cơ bản ──────────────────────────────────────────────────
NUM_INTERSECTIONS = 16          # Tổng số nút giao thông tự động trong mạng lưới (mạng 4x4)
SIMULATION_DT_SECONDS = 0.1     # Bước thời gian vật lý của simulation (mỗi tick mô phỏng chạy 0.1 giây)
GUI_REFRESH_MS = 50             # Tốc độ làm mới giao diện người dùng (50ms gửi data cập nhật UI một lần)
METRICS_WINDOW_SECONDS = 30.0   # Cửa sổ thời gian (giây) dùng để tính trung bình trượt các chỉ số (vận tốc, mật độ)
DENSITY_SAMPLE_SECONDS = 1.0    # Chu kỳ lấy mẫu mật độ giao thông (mỗi 1 giây lấy mẫu một lần)
DB_FLUSH_SECONDS = 5.0          # Chu kỳ đẩy dữ liệu log (metrics) xuống cơ sở dữ liệu PostgreSQL (mỗi 5 giây)
SPAWN_INTERVAL_SECONDS = 0.8    # Khoảng thời gian mặc định giữa các lần sinh xe (bị ghi đè bởi sóng sinh xe động)

# ─── Kích thước và Quy tắc Nút giao ──────────────────────────────────────────────
ZONE_LENGTH_METERS = 500.0      # Chiều dài vùng đo lường (m) trước ngã tư để tính mật độ và hàng chờ của làn xe đó
BOUNDARY_DISTANCE_METERS = 40.0 # Khoảng cách an toàn tối thiểu từ biên bản đồ (m)
STOP_LINE_DISTANCE_METERS = 8.0 # Khoảng cách vạch dừng xe (m) cách tâm giao lộ
QUEUE_SPEED_THRESHOLD = 9       # Ngưỡng vận tốc (m/s) – Xe chạy dưới 9 m/s (~32.4 km/h) sẽ bị coi là đang kẹt/chờ hàng dài

# ─── Thông số Phương tiện mặc định ──────────────────────────────────────────────
DEFAULT_TARGET_VEHICLE_COUNT = 90  # Số lượng xe mục tiêu trong mạng (đang bị ghi đè bởi sóng sinh xe hình sin động)
DEFAULT_MIN_SPEED_MPS = 12.0       # Tốc độ tối thiểu của xe (m/s) (~43.2 km/h)
DEFAULT_MAX_SPEED_MPS = 32.0       # Tốc độ tối đa của xe (m/s) (~115.2 km/h)
DEFAULT_MAX_ACCELERATION = 4.0     # Gia tốc tăng tốc lớn nhất của phương tiện (m/s²)
DEFAULT_MAX_DECELERATION = 7.0     # Gia tốc phanh/giảm tốc lớn nhất của phương tiện (m/s²)
DEFAULT_SAFE_GAP_METERS = 10.0     # Khoảng cách an toàn tối thiểu (m) giữa 2 xe nối đuôi nhau

# Tỷ lệ hướng rẽ mặc định tại các nút giao (Trái, Đi thẳng, Phải)
DEFAULT_TURN_DISTRIBUTION = {
    "left": 0.1,
    "straight": 0.75,
    "right": 0.15,
}

# ─── Cấu hình Đèn tín hiệu mặc định (khi không chạy RL) ─────────────────────────
DEFAULT_LIGHT_GREEN_SECONDS = 50.0  # Thời gian đèn XANH mặc định (giây)
DEFAULT_LIGHT_YELLOW_SECONDS = 5.0  # Thời gian đèn VÀNG mặc định (giây)
DEFAULT_LIGHT_RED_SECONDS = 60.0    # Thời gian đèn ĐỎ mặc định (giây)



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


# ─── Cấu hình Reward & Trọng số RL ──────────────────────────────────────────────
# (Tất cả tham số dưới đây được đồng bộ tự động sang lớp RewardWeights của Agent)

# Offset cơ sở để giữ Reward ở mức dương khi giao thông thông thoáng (tránh điểm âm hoàn toàn)
REWARD_OFFSET = 10

# Trọng số phạt tổng số xe đang xếp hàng chờ (queue) ở tất cả các làn vào.
# - ẢNH HƯỞNG: Giá trị càng cao, Agent càng ưu tiên giải tỏa nhanh mọi hàng chờ.
WEIGHT_QUEUE = 7.0

# Trọng số phạt sự mất cân bằng hàng chờ giữa các hướng (ví dụ: hướng Bắc chờ 20 xe, hướng Tây chờ 0 xe).
# - ẢNH HƯỞNG: Giá trị cao thúc đẩy Agent phân bổ thời gian đèn đều cho các hướng, tránh việc một hướng bị tắc cứng.
WEIGHT_IMBALANCE = 8.0

# Trọng số phạt xe phải chờ ở làn đang đèn Đỏ (Red Pressure).
# - ẢNH HƯỞNG: Giá trị cao giúp chống hiện tượng "bỏ đói" (starvation) các làn ít xe. Đèn đỏ có xe chờ quá lâu sẽ buộc phải chuyển xanh.
WEIGHT_RED_PRESSURE = 6.5

# Hình phạt cố định mỗi khi nút giao đổi pha đèn (từ Xanh -> Đỏ).
# - ẢNH HƯỞNG: Ngăn chặn tình trạng nhảy đèn liên tục (flapping). Giá trị cao giữ pha ổn định lâu hơn.
WEIGHT_SWITCH_PENALTY = 6

# Điểm thưởng dựa trên vận tốc trung bình của các xe trong khu vực nút giao.
# - ẢNH HƯỞNG: Khuyến khích tối ưu luồng xe chạy mượt mà, không bị dừng đỗ hẳn.
WEIGHT_SPEED_BONUS = 0.1


# ─── Hệ số chuẩn hóa (Scale Factors) ──────────────────────────────────────────
# Dùng để đưa các giá trị thực tế (số xe, tốc độ) về khoảng [0, 1] trước khi nhân trọng số.
# Giúp tránh lỗi bão hòa Reward (Reward Saturation) khi lượng xe trong simulation quá lớn (~1760 xe).

# Hệ số chia chuẩn hóa cho tổng hàng chờ. (Công thức: queue_total / SCALE_QUEUE)
SCALE_QUEUE = 80.0

# Hệ số chia chuẩn hóa cho độ lệch hàng chờ giữa các hướng.
SCALE_IMBALANCE = 50.0

# Hệ số chia chuẩn hóa cho hàng chờ ở làn đèn đỏ.
SCALE_RED_PRESSURE = 80.0

# Hệ số chia chuẩn hóa cho tốc độ trung bình (thường chia cho tốc độ tối đa của xe).
SCALE_SPEED = 29.0

# Giới hạn giá trị Reward trong khoảng [-REWARD_CLIP, +REWARD_CLIP] để ổn định gradient khi train.
REWARD_CLIP = 10.0

# ─── Hệ số phạt Imbalance Reward toàn mạng ──────────────────────────────────────
# Global Reward cuối cùng = mean(rewards) - GLOBAL_IMBALANCE_WEIGHT * std(rewards)
# - Nếu các nút có reward đồng đều: std ≈ 0 → không bị phạt thêm.
# - Nếu một vài nút bị tắc nghẽn nặng trong khi các nút khác tốt: std lớn → phạt nặng.
# - Khuyến khích Agent cân bằng tải cho toàn mạng thay vì chỉ tối ưu cục bộ từng nút.
GLOBAL_IMBALANCE_WEIGHT = 0.3
