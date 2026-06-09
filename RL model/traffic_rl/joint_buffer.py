"""
joint_buffer.py – Joint Replay Buffer cho QMIX.

Khác với ReplayBuffer cá nhân trong DQN (lưu từng transition đơn lẻ),
JointReplayBuffer lưu **joint transitions** bao gồm trạng thái và hành động
của TẤT CẢ agents cùng một thời điểm.

Cấu trúc một transition:
  obs_all        : (n_agents, obs_dim)        – local observation của từng agent
  actions_all    : (n_agents,)                – hành động của từng agent
  global_state   : (global_state_dim,)        – ghép toàn bộ obs (input cho mixing net)
  joint_reward   : float                      – mean reward toàn cục
  next_obs_all   : (n_agents, obs_dim)
  next_global_state: (global_state_dim,)
  done           : float                      – 1.0 nếu episode kết thúc

Lưu ý về capacity:
  Mỗi joint transition nặng hơn ~N lần individual transition.
  N=16 agents × obs_dim=40 → ~640 floats/transition.
  Capacity mặc định 5000 là hợp lý cho RAM thông thường.
"""

from __future__ import annotations

import random
from typing import NamedTuple, Sequence

import numpy as np
import torch


class JointBatch(NamedTuple):
    """Batch dữ liệu lấy mẫu từ JointReplayBuffer, đã convert sang Tensor.

    Attributes:
        obs_all         : (B, N, obs_dim)
        actions_all     : (B, N)
        global_state    : (B, global_state_dim)
        joint_reward    : (B,)
        next_obs_all    : (B, N, obs_dim)
        next_global_state: (B, global_state_dim)
        done            : (B,)
    """

    obs_all: torch.Tensor
    actions_all: torch.Tensor
    global_state: torch.Tensor
    joint_reward: torch.Tensor
    next_obs_all: torch.Tensor
    next_global_state: torch.Tensor
    done: torch.Tensor


class JointReplayBuffer:
    """Circular replay buffer lưu joint transitions cho QMIX.

    Args:
        n_agents   : Số agents (16 nút giao thông).
        obs_dim    : Chiều local observation của mỗi agent (obs_dim gốc + n_agents nếu ghép agent ID).
        global_state_dim: Chiều global state = n_agents × obs_dim.
        capacity   : Số joint transitions tối đa (mặc định 5000).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        global_state_dim: int,
        capacity: int = 5000,
    ) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.capacity = capacity

        # Pre-allocate numpy arrays để tránh Python list overhead
        self._obs        = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self._actions    = np.zeros((capacity, n_agents), dtype=np.int64)
        self._global_st  = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self._reward     = np.zeros((capacity,), dtype=np.float32)
        self._next_obs   = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self._next_gst   = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self._done       = np.zeros((capacity,), dtype=np.float32)

        self._position = 0   # Con trỏ ghi tiếp theo (circular)
        self._size = 0       # Số transitions thực sự trong buffer

    def push(
        self,
        obs_all: Sequence[Sequence[float]],
        actions_all: Sequence[int],
        global_state: Sequence[float],
        joint_reward: float,
        next_obs_all: Sequence[Sequence[float]],
        next_global_state: Sequence[float],
        done: bool = False,
    ) -> None:
        """Thêm một joint transition vào buffer.

        Args:
            obs_all          : Local obs của TẤT CẢ agents, shape (n_agents, obs_dim).
            actions_all      : Hành động của TẤT CẢ agents, shape (n_agents,).
            global_state     : Global state vector, shape (global_state_dim,).
            joint_reward     : Mean reward toàn cục của bước này.
            next_obs_all     : Local obs tiếp theo, shape (n_agents, obs_dim).
            next_global_state: Global state tiếp theo, shape (global_state_dim,).
            done             : True nếu episode kết thúc.
        """
        idx = self._position
        self._obs[idx]       = np.array(obs_all, dtype=np.float32)
        self._actions[idx]   = np.array(actions_all, dtype=np.int64)
        self._global_st[idx] = np.array(global_state, dtype=np.float32)
        self._reward[idx]    = float(joint_reward)
        self._next_obs[idx]  = np.array(next_obs_all, dtype=np.float32)
        self._next_gst[idx]  = np.array(next_global_state, dtype=np.float32)
        self._done[idx]      = float(done)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> JointBatch:
        """Lấy mẫu ngẫu nhiên một mini-batch joint transitions.

        Args:
            batch_size: Số transition muốn lấy.

        Returns:
            JointBatch với tất cả tensors đã convert sang torch.

        Raises:
            ValueError: Nếu buffer chứa ít hơn batch_size phần tử.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer chỉ có {self._size} transitions, cần ít nhất {batch_size}."
            )
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return JointBatch(
            obs_all=torch.FloatTensor(self._obs[indices]),
            actions_all=torch.LongTensor(self._actions[indices]),
            global_state=torch.FloatTensor(self._global_st[indices]),
            joint_reward=torch.FloatTensor(self._reward[indices]),
            next_obs_all=torch.FloatTensor(self._next_obs[indices]),
            next_global_state=torch.FloatTensor(self._next_gst[indices]),
            done=torch.FloatTensor(self._done[indices]),
        )

    def __len__(self) -> int:
        """Số transition đang có trong buffer."""
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """Kiểm tra buffer có đủ dữ liệu để sample chưa.

        Args:
            batch_size: Kích thước batch cần sample.

        Returns:
            True nếu buffer có ít nhất batch_size transitions.
        """
        return self._size >= batch_size
