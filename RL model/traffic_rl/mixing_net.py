"""
mixing_net.py – QMIX Mixing Network với Hypernetwork.

Kiến trúc:
  Mixing Network nhận Q-values cục bộ từ tất cả N agents và global state,
  tổng hợp thành một Q_joint duy nhất để tính TD loss trong training.

  Ràng buộc monotonicity (IGM condition):
    ∂Q_joint / ∂Q_i ≥ 0  ∀i
  → đảm bảo: argmax_a Q_joint = (argmax_a Q_1, ..., argmax_a Q_N)
  → agents có thể hành động greedy độc lập mà vẫn tối ưu joint policy.

  Cách đảm bảo: weights W1, W2 của mixing net được sinh bởi hypernetwork
  và ép về ≥ 0 bằng hàm abs().

Pipeline:
  q_agents (B, N) + global_state (B, D)
    → [HyperNet] → W1 (B, N, H), b1 (B, 1, H)
    → ELU activation
    → [HyperNet] → W2 (B, H, 1), b2 (B, 1, 1)
    → Q_joint (B,)

Tham khảo:
  Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation
  for Deep Multi-Agent Reinforcement Learning"
  https://arxiv.org/abs/1803.11485
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """Mixing Network của QMIX.

    Nhận Q-values cá nhân từ N agents và global state, tổng hợp thành
    Q_joint duy nhất theo kiến trúc 2-layer feedforward với monotonicity constraint.

    Hypernetwork sinh ra weights của mixing layers dựa trên global state,
    cho phép mixing weights thích ứng theo trạng thái toàn cục hệ thống.

    Args:
        n_agents         : Số agents (16 nút giao thông).
        global_state_dim : Chiều của global state vector.
        mixing_hidden_dim: Chiều hidden của mixing network (thường 32–64).
    """

    def __init__(
        self,
        n_agents: int,
        global_state_dim: int,
        mixing_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden_dim = mixing_hidden_dim

        # ── Hypernetwork cho lớp 1 ──────────────────────────────────────────
        # Sinh W1: (B, n_agents, mixing_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents * mixing_hidden_dim),
        )
        # Sinh b1: (B, 1, mixing_hidden_dim)
        self.hyper_b1 = nn.Linear(global_state_dim, mixing_hidden_dim)

        # ── Hypernetwork cho lớp 2 ──────────────────────────────────────────
        # Sinh W2: (B, mixing_hidden_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )
        # Sinh b2 (không cần monotone): (B, 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(
        self,
        q_agents: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """Tính Q_joint từ Q-values cá nhân và global state.

        Args:
            q_agents     : Q-value của action đã chọn cho từng agent.
                           Shape: (batch_size, n_agents).
            global_state : Vector trạng thái toàn cục.
                           Shape: (batch_size, global_state_dim).

        Returns:
            Q_joint: Scalar Q-value tổng hợp.
                     Shape: (batch_size,).
        """
        batch = q_agents.size(0)

        # ── Layer 1 ─────────────────────────────────────────────────────────
        # abs() đảm bảo W1 ≥ 0 → monotonicity constraint
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden_dim)

        b1 = self.hyper_b1(global_state)
        b1 = b1.view(batch, 1, self.mixing_hidden_dim)

        # q_agents: (B, N) → (B, 1, N) để nhân ma trận
        q_in = q_agents.unsqueeze(1)  # (B, 1, N)
        hidden = F.elu(torch.bmm(q_in, w1) + b1)  # (B, 1, H)

        # ── Layer 2 ─────────────────────────────────────────────────────────
        w2 = torch.abs(self.hyper_w2(global_state))
        w2 = w2.view(batch, self.mixing_hidden_dim, 1)

        b2 = self.hyper_b2(global_state)
        b2 = b2.view(batch, 1, 1)

        q_joint = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_joint.squeeze(-1).squeeze(-1)  # (B,)
