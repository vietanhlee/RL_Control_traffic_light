"""
agent.py – QMIX Agent cho hệ thống điều khiển đèn giao thông 16 nút.

Kiến trúc QMIX (Monotonic Value Function Factorisation):
  - Centralized Training, Decentralized Execution (CTDE)
  - N agents chia sẻ chung 1 Q-network (Parameter Sharing)
  - Agent ID được nhúng vào local observation (one-hot embedding)
  - Mixing Network tổng hợp Q_i → Q_joint (với monotonicity constraint)
  - Double DQN target để giảm overestimation bias

Thuật toán:
  1. Decentralized execution:
       a_i = argmax_a Q(o_i ⊕ id_i, a)  ← chỉ dùng local obs + agent ID
  2. Centralized training:
       Q_joint  = MixNet(Q_i(o_i, a_i)  for i, global_state)
       Q_target = MixNet(max_a Q_target_i(o_i', a) for i, global_state')
       target   = r + γ * Q_target * (1 - done)
       loss     = MSE(Q_joint, target)

Tham khảo:
  Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation"
  https://arxiv.org/abs/1803.11485
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .joint_buffer import JointBatch, JointReplayBuffer
from .mixing_net import MixingNetwork

# Hành động: 0 = Giữ nguyên pha đèn, 1 = Chuyển sang pha tiếp theo
ACTION_KEEP = 0
ACTION_CHANGE = 1


def _get_device() -> torch.device:
    """Tự động chọn device tốt nhất: CUDA > MPS > CPU.

    CUDA: NVIDIA GPU (phổ biến nhất, hiệu suất cao nhất).
    MPS : Apple Silicon GPU (macOS, M1/M2/M3).
    CPU : Fallback nếu không có GPU.

    Returns:
        torch.device sấn sàng để đưa model và tensors lên.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IndividualQNet(nn.Module):
    """Q-network cục bộ cho mỗi agent (Dueling DQN architecture).

    Nhận: local_obs ⊕ agent_id_onehot (đã ghép bên ngoài)
    Trả ra: Q-values cho 2 hành động [Keep, Change]

    Sử dụng Dueling architecture để tách Value stream / Advantage stream,
    giúp học ổn định hơn đặc biệt khi hành động không luôn quan trọng.

    Args:
        input_dim  : obs_dim + n_agents (local obs + one-hot agent ID).
        hidden_dim : Số neuron mỗi lớp ẩn (mặc định 128).
        n_actions  : Số hành động (mặc định 2: Keep/Change).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_actions: int = 2,
    ) -> None:
        super().__init__()

        # Feature extractor dùng chung
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Advantage stream A(s, a)
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        # Value stream V(s)
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tính Q-values cho tất cả hành động.

        Args:
            x: Input tensor shape (batch_size, input_dim).

        Returns:
            Q-values shape (batch_size, n_actions).
        """
        features = self.feature_layer(x)
        advantages = self.advantage_layer(features)
        values = self.value_layer(features)
        # Dueling: Q = V + (A - mean(A)) → đảm bảo identifiability
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


class QMIXAgent:
    """QMIX Agent điều phối 16 nút giao thông theo phương pháp CTDE.

    Kiến trúc:
      - 1 shared IndividualQNet (online) + 1 target Q-net
      - 1 MixingNetwork (online) + 1 target MixingNetwork
      - JointReplayBuffer lưu joint transitions
      - ε-greedy exploration

    Agent ID được nhúng dưới dạng one-hot vector và ghép vào local obs
    trước khi đưa vào Q-network, giúp shared network phân biệt agents.

    Global state = concatenation of toàn bộ local obs (n_agents × obs_dim).
    Mixing net nhận global state làm điều kiện để sinh adaptive weights.

    Args:
        n_agents          : Số agents = số nút giao (16).
        obs_dim           : Chiều local observation (feature_size, KHÔNG bao gồm agent ID).
        learning_rate     : Learning rate cho Adam (shared optimizer cho Q + Mixing).
        gamma             : Discount factor γ.
        epsilon           : Epsilon exploration ban đầu.
        min_epsilon       : Epsilon tối thiểu.
        epsilon_decay     : Hệ số suy giảm epsilon mỗi step.
        hidden_dim        : Hidden dim của Q-network.
        mixing_hidden_dim : Hidden dim của Mixing network.
        batch_size        : Kích thước mini-batch.
        buffer_capacity   : Capacity của JointReplayBuffer.
        target_update_freq: Số updates giữa 2 lần hard-update target nets.
        seed              : Random seed.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.96,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.9995,
        hidden_dim: int = 128,
        mixing_hidden_dim: int = 32,
        batch_size: int = 32,
        buffer_capacity: int = 5000,
        target_update_freq: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.seed = seed

        # Chiều input thực tế = obs gốc + one-hot agent ID
        self.input_dim = obs_dim + n_agents
        # Global state = ghép toàn bộ obs gốc (không có agent ID để giảm dim)
        self.global_state_dim = n_agents * obs_dim

        # ── Thiết lập seed ─────────────────────────────────────────────────
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)

        # ── Auto-detect device: CUDA → MPS → CPU ───────────────────────────
        self.device = _get_device()
        print(
            f"[DEVICE] Using: {self.device}"
            + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else "")
        )

        # ── Q-networks (shared weights cho tất cả agents) ──────────────────
        self.q_net = IndividualQNet(self.input_dim, hidden_dim).to(self.device)
        self.target_q_net = IndividualQNet(self.input_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # ── Mixing Networks ────────────────────────────────────────────────
        self.mixing_net = MixingNetwork(n_agents, self.global_state_dim, mixing_hidden_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(n_agents, self.global_state_dim, mixing_hidden_dim).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        self.target_mixing_net.eval()

        # ── Optimizer dùng chung cho Q-net + Mixing net ────────────────────
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            lr=learning_rate,
        )

        # ── Replay Buffer (numpy, nằm trên RAM – không cần device) ─────────
        self.buffer = JointReplayBuffer(
            n_agents=n_agents,
            obs_dim=obs_dim,
            global_state_dim=self.global_state_dim,
            capacity=buffer_capacity,
        )

        self.update_count = 0

        # Pre-compute one-hot agent ID tensors trực tiếp trên đúng device
        self._agent_id_onehot = torch.eye(n_agents, device=self.device)  # (N, N)


    # ─── Helper: ghép obs + agent ID ──────────────────────────────────────

    def _augment_obs(self, obs_array: np.ndarray) -> torch.Tensor:
        """Ghép local obs với one-hot agent ID và chuyển lên device.

        Args:
            obs_array: numpy array shape (n_agents, obs_dim).

        Returns:
            Tensor shape (n_agents, obs_dim + n_agents) trên self.device.
        """
        obs_t = torch.FloatTensor(obs_array).to(self.device)   # (N, obs_dim)
        return torch.cat([obs_t, self._agent_id_onehot], dim=-1)  # (N, obs_dim+N)

    def _build_global_state(self, obs_array: np.ndarray) -> np.ndarray:
        """Flatten toàn bộ local obs thành global state vector.

        Args:
            obs_array: numpy array shape (n_agents, obs_dim).

        Returns:
            numpy array shape (n_agents * obs_dim,).
        """
        return obs_array.flatten()

    # ─── Decentralized Execution ───────────────────────────────────────────

    def select_actions(
        self,
        all_obs: dict[int, list[float]],
        agent_ids: list[int],
        explore: bool = True,
    ) -> dict[int, int]:
        """Chọn hành động cho TẤT CẢ agents theo ε-greedy (decentralized).

        Mỗi agent chỉ dùng local obs của mình + agent ID → không cần giao tiếp.

        Args:
            all_obs   : Dict mapping agent_id → local obs features.
            agent_ids : Danh sách agent IDs theo thứ tự cố định.
            explore   : True = bật ε-greedy exploration.

        Returns:
            Dict mapping agent_id → action (0 hoặc 1).
        """
        actions: dict[int, int] = {}

        # Batch inference cho tất cả agents cùng lúc (hiệu quả hơn loop)
        obs_array = np.array([all_obs[aid] for aid in agent_ids], dtype=np.float32)
        augmented = self._augment_obs(obs_array)  # (N, obs_dim+N)

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(augmented)  # (N, 2)
        self.q_net.train()

        for i, agent_id in enumerate(agent_ids):
            if explore and self.rng.random() < self.epsilon:
                actions[agent_id] = self.rng.randint(0, 1)
            else:
                actions[agent_id] = int(q_values[i].argmax().item())

        return actions

    # ─── Centralized Training ──────────────────────────────────────────────

    def update(self) -> float:
        """Thực hiện một bước cập nhật QMIX từ mini-batch trong replay buffer.

           B = batch.obs_all.size(0)   # batch size
        N = self.n_agents

        # Chuyển toàn bộ batch tensors lên đúng device trước khi tính toán
        obs_all        = batch.obs_all.to(self.device)
        actions_all    = batch.actions_all.to(self.device)
        global_state   = batch.global_state.to(self.device)
        joint_reward   = batch.joint_reward.to(self.device)
        next_obs_all   = batch.next_obs_all.to(self.device)
        next_global_st = batch.next_global_state.to(self.device)
        done           = batch.done.to(self.device)

        # ── Online: Q_joint ────────────────────────────────────────────────
        # Reshape (B, N, obs_dim) → (B*N, obs_dim) để batch inference
        obs_flat = obs_all.view(B * N, self.obs_dim)

        # Ghép agent ID: tile one-hot theo batch (đã trên device)
        agent_ids_tiled = self._agent_id_onehot.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        agent_ids_flat  = agent_ids_tiled.reshape(B * N, N)

        obs_aug = torch.cat([obs_flat, agent_ids_flat], dim=-1)    # (B*N, input_dim)
        q_all   = self.q_net(obs_aug).view(B, N, 2)                # (B, N, 2)

        # Lấy Q-value của action đã thực hiện
        actions_exp = actions_all.unsqueeze(-1)                    # (B, N, 1)
        q_taken     = q_all.gather(2, actions_exp).squeeze(-1)     # (B, N)

        q_joint = self.mixing_net(q_taken, global_state)           # (B,)

        # ── Target: Q_joint_target ─────────────────────────────────────────
        next_obs_flat = next_obs_all.view(B * N, self.obs_dim)
        next_obs_aug  = torch.cat([next_obs_flat, agent_ids_flat], dim=-1)

        with torch.no_grad():
            # Double DQN: online net chọn action, target net đánh giá giá trị
            next_q_online  = self.q_net(next_obs_aug).view(B, N, 2)
            best_actions   = next_q_online.argmax(dim=-1, keepdim=True)   # (B, N, 1)

            next_q_target  = self.target_q_net(next_obs_aug).view(B, N, 2)
            next_q_taken   = next_q_target.gather(2, best_actions).squeeze(-1)  # (B, N)

            q_joint_target = self.target_mixing_net(next_q_taken, next_global_st)  # (B,)

            # Bellman equation
            td_target = joint_reward + self.gamma * q_joint_target * (1.0 - done)
            td_target = torch.clamp(td_target, -50.0, 50.0)  # safety clip

        # MSE loss
        loss = nn.functional.mse_loss(q_joint, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping toàn bộ params (Q-net + Mixing net)
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(loss.item())
ext_obs_flat = batch.next_obs_all.view(B * N, self.obs_dim)
        next_obs_aug = torch.cat([next_obs_flat, agent_ids_flat], dim=-1)

        with torch.no_grad():
            # Double DQN: online net chọn action, target net đánh giá
            next_q_online = self.q_net(next_obs_aug).view(B, N, 2)
            best_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (B, N, 1)

            next_q_target = self.target_q_net(next_obs_aug).view(B, N, 2)
            next_q_taken = next_q_target.gather(2, best_actions).squeeze(-1)  # (B, N)

            q_joint_target = self.target_mixing_net(
                next_q_taken, batch.next_global_state
            )  # (B,)

            # Bellman equation
            td_target = batch.joint_reward + self.gamma * q_joint_target * (1.0 - batch.done)
            td_target = torch.clamp(td_target, -50.0, 50.0)  # safety clip

        # MSE loss
        loss = nn.functional.mse_loss(q_joint, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping toàn bộ params (Q-net + Mixing net)
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(loss.item())

    def _update_target_networks(self) -> None:
        """Hard-update target networks từ online networks."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def decay_epsilon(self) -> None:
        """Giảm epsilon theo lịch nhân. Gọi mỗi step sau khi update."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ─── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Lưu toàn bộ trạng thái QMIX agent vào file .pth.

        Lưu bao gồm: hyperparameters + state_dicts của Q-net và Mixing net.
        Replay buffer và target nets KHÔNG được lưu (tái tạo khi load).

        Args:
            path: Đường dẫn file đầu ra (.pth).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            # Metadata kiến trúc
            "algorithm": "QMIX",
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "input_dim": self.input_dim,
            "global_state_dim": self.global_state_dim,
            # Hyperparameters
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "seed": self.seed,
            # Weights – luôn lưu trên CPU để load được trên bất kỳ device nào
            "q_net_state_dict": {k: v.cpu() for k, v in self.q_net.state_dict().items()},
            "mixing_net_state_dict": {k: v.cpu() for k, v in self.mixing_net.state_dict().items()},
        }
        torch.save(payload, target_path)
        print(f"[INFO] QMIX model saved -> {target_path} (device={self.device})")


    @classmethod
    def load(
        cls,
        path: str | Path,
        default_n_agents: int,
        default_obs_dim: int,
        **kwargs: Any,
    ) -> "QMIXAgent":
        """Tải QMIX agent từ file .pth.

        Nếu file không tồn tại hoặc bị lỗi → khởi tạo agent mới.

        Args:
            path             : Đường dẫn file .pth.
            default_n_agents : n_agents dùng khi không tìm thấy file.
            default_obs_dim  : obs_dim dùng khi không tìm thấy file.
            **kwargs         : Override hyperparameters (vd: learning_rate=0.001).

        Returns:
            QMIXAgent đã restore weights, hoặc agent mới nếu lỗi/không có file.
        """
        source_path = Path(path)
        if not source_path.exists():
            print(f"[INFO] Không tìm thấy QMIX model tại '{path}'. Khởi tạo mới.")
            return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)

        try:
            payload = torch.load(source_path, map_location="cpu", weights_only=False)

            # Kiểm tra algorithm tag
            if payload.get("algorithm") != "QMIX":
                print(f"[WARNING] File '{path}' không phải QMIX model. Khởi tạo mới.")
                return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)

            agent = cls(
                n_agents=int(payload["n_agents"]),
                obs_dim=int(payload["obs_dim"]),
                learning_rate=kwargs.get("learning_rate", float(payload.get("learning_rate", 0.0005))),
                gamma=kwargs.get("gamma", float(payload.get("gamma", 0.96))),
                epsilon=kwargs.get("epsilon", float(payload.get("epsilon", 1.0))),
                min_epsilon=kwargs.get("min_epsilon", float(payload.get("min_epsilon", 0.05))),
                epsilon_decay=kwargs.get("epsilon_decay", float(payload.get("epsilon_decay", 0.9995))),
                batch_size=kwargs.get("batch_size", int(payload.get("batch_size", 32))),
                target_update_freq=kwargs.get(
                    "target_update_freq", int(payload.get("target_update_freq", 50))
                ),
                seed=int(payload.get("seed", 42)),
            )
            agent.q_net.load_state_dict(payload["q_net_state_dict"])
            agent.target_q_net.load_state_dict(payload["q_net_state_dict"])
            agent.mixing_net.load_state_dict(payload["mixing_net_state_dict"])
            agent.target_mixing_net.load_state_dict(payload["mixing_net_state_dict"])
            print(f"[INFO] QMIX model loaded from '{path}'.")
            return agent
        except Exception as exc:
            print(f"[WARNING] Không thể tải QMIX model từ '{path}' (lỗi: {exc}). Khởi tạo mới.")
            return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)

    # ─── Backward compatibility alias (giữ để evaluate.py cũ không lỗi) ──

    @property
    def feature_size(self) -> int:
        """Alias backward-compat: trả về obs_dim."""
        return self.obs_dim


# Alias backward compatibility – giữ cho evaluate.py và các script cũ
LinearQAgent = QMIXAgent
DuelingQAgent = QMIXAgent
