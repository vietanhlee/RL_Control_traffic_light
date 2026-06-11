"""
config.py – Hyperparameters và cấu hình cho QMIX agent.

Tất cả các hằng số DEFAULT_* được dùng làm giá trị mặc định trong
argparse của train.py và evaluate.py. Có thể override qua CLI args.
"""

from __future__ import annotations


# ─── API & Training Loop ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_DECISION_INTERVAL_SECONDS = 1.5 # Giây chờ giữa các quyết định
DEFAULT_SAVE_EVERY = 250                   # Lưu model mỗi N step
DEFAULT_MODEL_PATH = "artifacts/qmix_agent.pth"
DEFAULT_HISTORY_WINDOW = 32               # Cửa sổ tính moving average
GLOBAL_IMBALANCE_WEIGHT = 0.55            # Hệ số phạt imbalance toàn mạng khi gom joint reward

# ─── QMIX Hyperparameters ─────────────────────────────────────────────────────
DEFAULT_N_AGENTS = 16                     # Số agents = số nút giao thông
DEFAULT_LR = 0.0005                       # Learning rate (Adam)
DEFAULT_GAMMA = 0.98                      # Discount factor γ (Tăng từ 0.96 -> 0.98 để Agent nhìn xa trông rộng hơn)
DEFAULT_EPSILON = 1.0                     # Epsilon exploration ban đầu
DEFAULT_MIN_EPSILON = 0.05                # Epsilon tối thiểu
DEFAULT_EPSILON_DECAY = 0.9995            # Hệ số suy giảm epsilon/step
DEFAULT_MIN_PHASE_HOLD_STEPS = 4          # Bước tối thiểu giữ pha đèn
DEFAULT_BATCH_SIZE = 64                   # Kích thước mini-batch từ joint buffer
DEFAULT_BUFFER_CAPACITY = 5000            # Capacity của JointReplayBuffer
DEFAULT_TARGET_UPDATE_FREQ = 50          # Hard-update target nets mỗi N updates
DEFAULT_HIDDEN_DIM = 256                  # Hidden dim của Q-network
DEFAULT_MIXING_HIDDEN_DIM = 128            # Hidden dim của Mixing network



