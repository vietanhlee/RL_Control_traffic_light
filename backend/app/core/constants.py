from __future__ import annotations

from typing import Dict, Tuple
from .env_loader import load_dotenv

load_dotenv()

# ─── Mạng lưới đường bất đối xứng phức tạp (16 nút giao) ─────────────────────────
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
INTERSECTION_CONNECTIONS = [
    # Đại lộ trung tâm huyết mạch (4 làn xe lớn)
    (4, 5, 4),
    (5, 6, 4),
    (6, 7, 4),
    (9, 10, 4),
    
    # Đại lộ phụ (3 làn xe)
    (1, 5, 3),
    (5, 9, 3),
    (9, 13, 3),
    (2, 6, 3),
    (6, 10, 3),
    (10, 14, 3),
    (4, 8, 3),
    
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

# ─── Cấu hình Reward & Trọng số RL ──────────────────────────────────────────────
REWARD_OFFSET = 10

# Trọng số phạt tổng số xe đang xếp hàng chờ (queue) ở tất cả các làn vào
WEIGHT_QUEUE = 7.0

# Trọng số phạt sự mất cân bằng hàng chờ giữa các hướng
WEIGHT_IMBALANCE = 8.0

# Trọng số phạt xe phải chờ ở làn đang đèn Đỏ (Red Pressure)
WEIGHT_RED_PRESSURE = 6.5

# Hình phạt cố định mỗi khi nút giao đổi pha đèn (từ Xanh -> Đỏ)
WEIGHT_SWITCH_PENALTY = 6

# Điểm thưởng dựa trên vận tốc trung bình của các xe trong khu vực nút giao
WEIGHT_SPEED_BONUS = 0.1

# ─── Hệ số chuẩn hóa (Scale Factors) ──────────────────────────────────────────
SCALE_QUEUE = 80.0
SCALE_IMBALANCE = 50.0
SCALE_RED_PRESSURE = 80.0
SCALE_SPEED = 29.0

# Giới hạn giá trị Reward trong khoảng [-REWARD_CLIP, +REWARD_CLIP]
REWARD_CLIP = 20.0

# ─── Hệ số phạt Imbalance Reward toàn mạng ──────────────────────────────────────
GLOBAL_IMBALANCE_WEIGHT = 0.55
