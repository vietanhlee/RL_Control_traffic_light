"""
reward_fn.py – Hàm tính reward dùng chung (Single Source of Truth).

Module này chứa toàn bộ logic tính reward cho một nút giao thông, được
dùng chung cho cả:
  - Backend simulation loop  (app/services/simulation_service.py)
  - RL Agent training        (RL model/traffic_rl/environment.py)

Bằng cách dùng chung module này, bất kỳ thay đổi nào trong reward function
(công thức, trọng số, phi tuyến tính, v.v.) sẽ tự động áp dụng cho cả hai.

Reward function (per-intersection):
    cost = w_q*(queue/scale_q) + w_i*(imbal/scale_i)
           + w_r*(red_pressure/scale_r) + switch_penalty
           - w_s*(speed/scale_s)
    reward_raw = reward_offset - cost
    # Phạt phi tuyến tính (tắc nghẽn nặng → phạt bình phương)
    if reward_raw < 0: reward_raw = -(abs(reward_raw) ** 2)
    reward = clip(reward_raw, -reward_clip, +reward_clip)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import (
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
)


@dataclass(frozen=True)
class RewardComponents:
    """Kết quả đầy đủ từ hàm tính reward, bao gồm reward và các chỉ số chẩn đoán.

    Attributes:
        reward              : Giá trị reward cuối đã clip vào [-REWARD_CLIP, +REWARD_CLIP].
        reward_raw          : Giá trị reward trước khi clip (sau khi áp dụng phi tuyến).
        cost                : Tổng chi phí (tổng penalty − speed bonus).
        queue_penalty       : Thành phần phạt tổng hàng chờ.
        imbalance_penalty   : Thành phần phạt mất cân bằng hướng.
        red_pressure_penalty: Thành phần phạt xe chờ ở làn đèn đỏ.
        switch_penalty      : Hình phạt cố định khi đổi pha.
        speed_bonus         : Điểm thưởng tốc độ lưu thông.
        queue_pct           : Phần trăm tổng hàng chờ so với SCALE_QUEUE [0-100].
        imbalance_pct       : Phần trăm imbalance so với SCALE_IMBALANCE [0-100].
        red_pressure_pct    : Phần trăm red_pressure so với SCALE_RED_PRESSURE [0-100].
        speed_pct           : Phần trăm tốc độ so với SCALE_SPEED [0-100].
    """
    reward: float
    reward_raw: float
    cost: float
    queue_penalty: float
    imbalance_penalty: float
    red_pressure_penalty: float
    switch_penalty: float
    speed_bonus: float
    queue_pct: float
    imbalance_pct: float
    red_pressure_pct: float
    speed_pct: float


def compute_intersection_reward(
    queue_total: float,
    imbalance: float,
    red_pressure: float,
    speed_avg: float,
    switched: bool,
    *,
    w_queue: float = WEIGHT_QUEUE,
    w_imbalance: float = WEIGHT_IMBALANCE,
    w_red_pressure: float = WEIGHT_RED_PRESSURE,
    w_switch: float = WEIGHT_SWITCH_PENALTY,
    w_speed: float = WEIGHT_SPEED_BONUS,
    scale_queue: float = SCALE_QUEUE,
    scale_imbalance: float = SCALE_IMBALANCE,
    scale_red_pressure: float = SCALE_RED_PRESSURE,
    scale_speed: float = SCALE_SPEED,
    reward_offset: float = float(REWARD_OFFSET),
    reward_clip: float = REWARD_CLIP,
) -> RewardComponents:
    """Tính reward cho một nút giao thông.

    Hàm này là **Single Source of Truth** cho toàn bộ reward logic.
    Được gọi từ cả Backend simulation loop và RL Agent training.

    Args:
        queue_total   : Tổng số xe đang xếp hàng chờ ở tất cả hướng vào.
        imbalance     : Độ mất cân bằng queue giữa các hướng = Σ|q_i − mean(q)|.
        red_pressure  : Tổng queue ở các làn đang đèn đỏ.
        speed_avg     : Tốc độ trung bình các phương tiện (m/s).
        switched      : True nếu vừa thực hiện đổi pha đèn.
        w_queue       : Trọng số phạt queue (mặc định từ constants.py).
        w_imbalance   : Trọng số phạt imbalance.
        w_red_pressure: Trọng số phạt red pressure.
        w_switch      : Hình phạt cố định khi đổi pha.
        w_speed       : Trọng số thưởng tốc độ.
        scale_queue   : Hệ số chuẩn hóa queue.
        scale_imbalance: Hệ số chuẩn hóa imbalance.
        scale_red_pressure: Hệ số chuẩn hóa red pressure.
        scale_speed   : Hệ số chuẩn hóa tốc độ.
        reward_offset : Điểm khởi đầu (trừ dần cost ra).
        reward_clip   : Giới hạn clip reward.

    Returns:
        RewardComponents chứa reward cuối cùng và tất cả thành phần chẩn đoán.
    """
    switch_penalty = w_switch if switched else 0.0

    # Tính các thành phần penalty & bonus
    queue_penalty        = w_queue        * (queue_total  / scale_queue)
    imbalance_penalty    = w_imbalance    * (imbalance    / scale_imbalance)
    red_pressure_penalty = w_red_pressure * (red_pressure / scale_red_pressure)
    speed_bonus          = w_speed        * (speed_avg    / scale_speed)

    cost = (
        queue_penalty
        + imbalance_penalty
        + red_pressure_penalty
        + switch_penalty
        - speed_bonus
    )

    reward_raw = reward_offset - cost

    # Phạt phi tuyến tính: tắc nghẽn nặng (reward âm) bị nhân đôi mức phạt
    # Công thức: reward_raw → −(|reward_raw|²) khi reward_raw < 0
    # Tác dụng: gradient agent mạnh hơn ở vùng tắc nghẽn nặng
    if reward_raw < 0.0:
        reward_raw = -(abs(reward_raw) ** 1.5)

    reward = max(-reward_clip, min(reward_clip, reward_raw))

    return RewardComponents(
        reward=reward,
        reward_raw=reward_raw,
        cost=cost,
        queue_penalty=queue_penalty,
        imbalance_penalty=imbalance_penalty,
        red_pressure_penalty=red_pressure_penalty,
        switch_penalty=switch_penalty,
        speed_bonus=speed_bonus,
        queue_pct=min((queue_total / scale_queue) * 100, 100.0),
        imbalance_pct=min((imbalance / scale_imbalance) * 100, 100.0),
        red_pressure_pct=min((red_pressure / scale_red_pressure) * 100, 100.0),
        speed_pct=min((speed_avg / scale_speed) * 100, 100.0),
    )
