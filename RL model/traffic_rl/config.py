"""
config.py – Hyperparameters và cấu hình cho QMIX agent.

Tất cả các hằng số DEFAULT_* được dùng làm giá trị mặc định trong
argparse của train.py và evaluate.py. Có thể override qua CLI args.
"""

from __future__ import annotations

from dataclasses import dataclass

# ─── API & Training Loop ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_DECISION_INTERVAL_SECONDS = 5   # Giây chờ giữa các quyết định
DEFAULT_SAVE_EVERY = 250                   # Lưu model mỗi N step
DEFAULT_MODEL_PATH = "RL model/artifacts/qmix_agent.pth"
DEFAULT_HISTORY_WINDOW = 32               # Cửa sổ tính moving average

# ─── QMIX Hyperparameters ─────────────────────────────────────────────────────
DEFAULT_N_AGENTS = 16                     # Số agents = số nút giao thông
DEFAULT_LR = 0.0005                       # Learning rate (Adam)
DEFAULT_GAMMA = 0.96                      # Discount factor γ
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

    Reward function:
      cost = queue * (queue_total / queue_scale)
             + imbalance * (imbalance / imbalance_scale)
             + red_pressure * (red_pressure / red_pressure_scale)
             + switch_penalty (nếu action=CHANGE)
             - speed_bonus * (avg_speed / speed_scale)
      reward = clip(reward_offset - cost, -reward_clip, +reward_clip)

    Joint reward cho QMIX = mean(reward_i for i in agents).
    """

    queue: float = 2.2           # Trọng số phạt tổng queue (xe đang chờ)
    density: float = 0.7         # Trọng số phạt mật độ xe (reserved)
    imbalance: float = 1.2       # Trọng số phạt mất cân bằng giữa các hướng
    red_pressure: float = 0.6    # Trọng số phạt khi xe xếp hàng dài ở đèn đỏ
    switch_penalty: float = 0.35 # Phạt cố định mỗi khi agent chọn CHANGE
    speed_bonus: float = 0.25    # Thưởng khi xe lưu thông nhanh

    # Scale factors (chuẩn hóa các giá trị về khoảng [0, 1])
    queue_scale: float = 12.0
    density_scale: float = 8.0
    imbalance_scale: float = 10.0
    red_pressure_scale: float = 10.0
    speed_scale: float = 18.0

    # Reward clip và offset
    reward_clip: float = 5.0     # Clip reward vào [-5, +5]
    reward_offset: float = 5.0   # Baseline reward (tránh reward âm hoàn toàn)
