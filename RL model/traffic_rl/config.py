"""
config.py – Hyperparameters và cấu hình cho QMIX agent.

Tất cả các hằng số DEFAULT_* được dùng làm giá trị mặc định trong
argparse của train.py và evaluate.py. Có thể override qua CLI args.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

# Tự động thêm thư mục gốc của dự án vào sys.path để import được config.constants
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.constants import (
    REWARD_OFFSET,
    WEIGHT_QUEUE,
    WEIGHT_IMBALANCE,
    WEIGHT_RED_PRESSURE,
    WEIGHT_SWITCH_PENALTY,
    WEIGHT_SPEED_BONUS,
    SCALE_QUEUE,
    SCALE_IMBALANCE,
    SCALE_RED_PRESSURE,
    SCALE_SPEED,
    REWARD_CLIP,
    GLOBAL_IMBALANCE_WEIGHT,
)


# ─── API & Training Loop ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_DECISION_INTERVAL_SECONDS = 1.5 # Giây chờ giữa các quyết định
DEFAULT_SAVE_EVERY = 250                   # Lưu model mỗi N step
DEFAULT_MODEL_PATH = "RL model/artifacts/qmix_agent.pth"
DEFAULT_HISTORY_WINDOW = 32               # Cửa sổ tính moving average

# ─── QMIX Hyperparameters ─────────────────────────────────────────────────────
DEFAULT_N_AGENTS = 16                     # Số agents = số nút giao thông
DEFAULT_LR = 0.0005                       # Learning rate (Adam)
DEFAULT_GAMMA = 0.98                      # Discount factor γ (Tăng từ 0.96 -> 0.98 để Agent nhìn xa trông rộng hơn)
DEFAULT_EPSILON = 1.0                     # Epsilon exploration ban đầu
DEFAULT_MIN_EPSILON = 0.05                # Epsilon tối thiểu
DEFAULT_EPSILON_DECAY = 0.9995            # Hệ số suy giảm epsilon/step
DEFAULT_MIN_PHASE_HOLD_STEPS = 4          # Bước tối thiểu giữ pha đèn
DEFAULT_BATCH_SIZE = 32                   # Kích thước mini-batch từ joint buffer
DEFAULT_BUFFER_CAPACITY = 5000            # Capacity của JointReplayBuffer
DEFAULT_TARGET_UPDATE_FREQ = 50          # Hard-update target nets mỗi N updates
DEFAULT_HIDDEN_DIM = 128                  # Hidden dim của Q-network
DEFAULT_MIXING_HIDDEN_DIM = 32            # Hidden dim của Mixing network


@dataclass(frozen=True)
class RewardWeights:
    """Trọng số và hệ số scale cho hàm reward của từng nút giao.

    Giá trị của các hằng số được lấy trực tiếp từ config/constants.py
    để đảm bảo tính đồng bộ giữa Backend và RL Agent.
    """

    queue: float = WEIGHT_QUEUE
    density: float = 0.7         # Trọng số phạt mật độ xe (reserved)
    imbalance: float = WEIGHT_IMBALANCE
    red_pressure: float = WEIGHT_RED_PRESSURE
    switch_penalty: float = WEIGHT_SWITCH_PENALTY
    speed_bonus: float = WEIGHT_SPEED_BONUS

    # Scale factors (chuẩn hóa các giá trị về khoảng [0, 1])
    queue_scale: float = SCALE_QUEUE
    density_scale: float = 8.0
    imbalance_scale: float = SCALE_IMBALANCE
    red_pressure_scale: float = SCALE_RED_PRESSURE
    speed_scale: float = SCALE_SPEED

    # Reward clip và offset
    reward_clip: float = REWARD_CLIP
    reward_offset: float = float(REWARD_OFFSET)
