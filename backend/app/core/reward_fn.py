"""
reward_fn.py – Hàm tính reward dùng chung (Single Source of Truth).

Module này chứa toàn bộ logic tính reward cho một nút giao thông, được
dùng chung cho cả:
  - Backend simulation loop  (backend/services/simulation_service.py)
  - RL Agent training        (rl_agent/traffic_rl/environment.py)

Bằng cách dùng chung module này, bất kỳ thay đổi nào trong reward function
(công thức, trọng số, phi tuyến tính, v.v.) sẽ tự động áp dụng cho cả hai.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Kết quả đầy đủ từ hàm tính reward, bao gồm reward và các chỉ số chẩn đoán."""
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
    if reward_raw < 0.0:
        reward_raw = -(abs(reward_raw) ** 2.5)

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
