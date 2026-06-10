# -*- coding: utf-8 -*-
"""
agent.py – QMIX Agent cho he thong dieu khien den giao thong 16 nut.


Kiến trúc QMIX (Monotonic Value Function Factorisation):
  - Centralized Training, Decentralized Execution (CTDE)
  - N agents chia sẻ chung 1 Q-network (Parameter Sharing)
  - Agent ID được nhúng vào local observation (one-hot embedding)
  - Mixing Network tổng hợp Q_i → Q_joint (với monotonicity constraint)
  - Double DQN target để giảm overestimation bias

Thuật toán:
  1. Decentralized execution:
       a_i = argmax_a Q(o_i ⊕ id_i, a)  <- chỉ dùng local obs + agent ID
  2. Centralized training:
       Q_joint  = MixNet(Q_i(o_i, a_i)  for i, global_state)
       Q_target = MixNet(max_a Q_target_i(o_i', a) for i, global_state')
       target   = r + gamma * Q_target * (1 - done)
       loss     = MSE(Q_joint, target)

Device support:
  Tự động phát hiện và sử dụng CUDA > MPS > CPU.
  Toàn bộ model và tensors được đưa lên device phù hợp.

Tham khảo:
  Rashid et al. (2018) "QMIX: Monotonic Value Function Factorisation"
  https://arxiv.org/abs/1803.11485
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from .joint_buffer import JointBatch, JointReplayBuffer
from .mixing_net import MixingNetwork

# Hanh dong: 0 = Giu nguyen pha den, 1 = Chuyen sang pha tiep theo
ACTION_KEEP = 0
ACTION_CHANGE = 1


def _get_device() -> torch.device:
    """Tu dong chon device tot nhat: CUDA > MPS > CPU.

    CUDA: NVIDIA GPU (pho bien nhat, hieu suat cao nhat).
    MPS : Apple Silicon GPU (macOS, M1/M2/M3).
    CPU : Fallback neu khong co GPU.

    Returns:
        torch.device san sang de dua model va tensors len.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IndividualQNet(nn.Module):
    """Q-network cuc bo cho moi agent (Dueling DQN architecture).

    Nhan: local_obs concat agent_id_onehot (da ghep ben ngoai)
    Tra ra: Q-values cho 2 hanh dong [Keep, Change]

    Su dung Dueling architecture de tach Value stream / Advantage stream,
    giup hoc on dinh hon dac biet khi hanh dong khong luon quan trong.

    Args:
        input_dim  : obs_dim + n_agents (local obs + one-hot agent ID).
        hidden_dim : So neuron moi lop an (mac dinh 128).
        n_actions  : So hanh dong (mac dinh 2: Keep/Change).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_actions: int = 2,
    ) -> None:
        super().__init__()

        # Feature extractor dung chung
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
        """Tinh Q-values cho tat ca hanh dong.

        Args:
            x: Input tensor shape (batch_size, input_dim).

        Returns:
            Q-values shape (batch_size, n_actions).
        """
        features = self.feature_layer(x)
        advantages = self.advantage_layer(features)
        values = self.value_layer(features)
        # Dueling: Q = V + (A - mean(A))
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


class QMIXAgent:
    """QMIX Agent dieu phoi 16 nut giao thong theo phuong phap CTDE.

    Kien truc:
      - 1 shared IndividualQNet (online) + 1 target Q-net
      - 1 MixingNetwork (online) + 1 target MixingNetwork
      - JointReplayBuffer luu joint transitions
      - epsilon-greedy exploration
      - Auto-detect device: CUDA > MPS > CPU

    Agent ID duoc nhung duoi dang one-hot vector va ghep vao local obs
    truoc khi dua vao Q-network, giup shared network phan biet agents.

    Global state = concatenation of toan bo local obs (n_agents x obs_dim).
    Mixing net nhan global state lam dieu kien de sinh adaptive weights.

    Args:
        n_agents          : So agents = so nut giao (16).
        obs_dim           : Chieu local observation (obs_dim goc cua moi agent).
        learning_rate     : Learning rate cho Adam.
        gamma             : Discount factor.
        epsilon           : Epsilon exploration ban dau.
        min_epsilon       : Epsilon toi thieu.
        epsilon_decay     : He so suy giam epsilon moi step.
        hidden_dim        : Hidden dim cua Q-network.
        mixing_hidden_dim : Hidden dim cua Mixing network.
        batch_size        : Kich thuoc mini-batch.
        buffer_capacity   : Capacity cua JointReplayBuffer.
        target_update_freq: So updates giua 2 lan hard-update target nets.
        seed              : Random seed.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        layout: dict[int, tuple[float, float]] | None = None,
        connections: list[tuple[int, int, int]] | None = None,
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

        # Chieu input thuc te = obs goc + one-hot agent ID
        self.input_dim = obs_dim + n_agents
        # Global state = ghep toan bo obs goc (khong co agent ID de giam dim)
        self.global_state_dim = n_agents * obs_dim

        # Thiet lap seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)

        # ── Auto-detect device: CUDA → MPS → CPU ───────────────────────────
        self.device = _get_device()
        gpu_name = (
            f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""
        )
        print(f"[DEVICE] Using: {self.device}{gpu_name}")

        # ── Q-networks (shared weights cho tat ca agents) ──────────────────
        self.q_net = IndividualQNet(self.input_dim, hidden_dim).to(self.device)
        self.target_q_net = IndividualQNet(self.input_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # ── Build weighted normalized adjacency matrix for GNN ─────────────
        A = np.zeros((n_agents, n_agents), dtype=np.float32)
        import math
        
        raw_weights = []
        edges_list = []
        conn = connections if connections is not None else []
        lay = layout if layout is not None else {}
        
        for u, v, lanes in conn:
            if u < n_agents and v < n_agents:
                # Tính khoảng cách Euclidean giữa 2 nút giao
                pos_u = lay.get(u, (0.0, 0.0))
                pos_v = lay.get(v, (0.0, 0.0))
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                dist = math.hypot(dx, dy)
                
                # Trọng số gốc tỉ lệ thuận với số làn xe, tỉ lệ nghịch với khoảng cách
                w_raw = lanes / max(dist, 1.0)
                raw_weights.append(w_raw)
                edges_list.append((u, v, w_raw))
                
        # Chuẩn hóa để trọng số trung bình giữa các cạnh kề là 1.0 (cân bằng với self-loop)
        mean_w = np.mean(raw_weights) if raw_weights else 1.0
        for u, v, w_raw in edges_list:
            w_norm = w_raw / mean_w
            A[u, v] = w_norm
            A[v, u] = w_norm
        
        # Self-loops
        A_tilde = A + np.eye(n_agents, dtype=np.float32)
        # Degree matrix D_tilde
        D_tilde = np.sum(A_tilde, axis=1)
        # D_tilde^{-1/2}
        D_tilde_inv_sqrt = np.zeros_like(D_tilde)
        np.power(D_tilde, -0.5, where=D_tilde > 0, out=D_tilde_inv_sqrt)
        D_inv_sqrt_mat = np.diag(D_tilde_inv_sqrt)
        # Normalized Adjacency A_hat = D^{-1/2} * A_tilde * D^{-1/2}
        A_hat = D_inv_sqrt_mat @ A_tilde @ D_inv_sqrt_mat
        
        self.adj = torch.from_numpy(A_hat).float().to(self.device)

        # ── Mixing Networks ────────────────────────────────────────────────
        self.mixing_net = MixingNetwork(
            n_agents, obs_dim, mixing_hidden_dim, adj=self.adj
        ).to(self.device)
        self.target_mixing_net = MixingNetwork(
            n_agents, obs_dim, mixing_hidden_dim, adj=self.adj
        ).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        self.target_mixing_net.eval()

        # ── Optimizer dung chung cho Q-net + Mixing net ────────────────────
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            lr=learning_rate,
        )

        # ── Replay Buffer (numpy, tren RAM - khong can device) ─────────────
        self.buffer = JointReplayBuffer(
            n_agents=n_agents,
            obs_dim=obs_dim,
            global_state_dim=self.global_state_dim,
            capacity=buffer_capacity,
        )

        self.update_count = 0

        # Pre-compute one-hot agent ID tensors tren dung device
        self._agent_id_onehot = torch.eye(n_agents, device=self.device)  # (N, N)

    # ─── Helper: ghep obs + agent ID ──────────────────────────────────────

    def _augment_obs(self, obs_array: np.ndarray) -> torch.Tensor:
        """Ghep local obs voi one-hot agent ID va chuyen len device.

        Args:
            obs_array: numpy array shape (n_agents, obs_dim).

        Returns:
            Tensor shape (n_agents, obs_dim + n_agents) tren self.device.
        """
        obs_t = torch.FloatTensor(obs_array).to(self.device)          # (N, obs_dim)
        return torch.cat([obs_t, self._agent_id_onehot], dim=-1)       # (N, obs_dim+N)

    def _build_global_state(self, obs_array: np.ndarray) -> np.ndarray:
        """Flatten toan bo local obs thanh global state vector.

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
        """Chon hanh dong cho TAT CA agents theo epsilon-greedy (decentralized).

        Moi agent chi dung local obs cua minh + agent ID -> khong can giao tiep.

        Args:
            all_obs   : Dict mapping agent_id -> local obs features.
            agent_ids : Danh sach agent IDs theo thu tu co dinh.
            explore   : True = bat epsilon-greedy exploration.

        Returns:
            Dict mapping agent_id -> action (0 hoac 1).
        """
        actions: dict[int, int] = {}

        # Batch inference cho tat ca agents cung luc (hieu qua hon loop)
        obs_array = np.array([all_obs[aid] for aid in agent_ids], dtype=np.float32)
        augmented = self._augment_obs(obs_array)  # (N, obs_dim+N) tren device

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(augmented)  # (N, 2) tren device
        self.q_net.train()

        for i, agent_id in enumerate(agent_ids):
            if explore and self.rng.random() < self.epsilon:
                actions[agent_id] = self.rng.randint(0, 1)
            else:
                actions[agent_id] = int(q_values[i].argmax().item())

        return actions

    # ─── Centralized Training ──────────────────────────────────────────────

    def update(self) -> float:
        """Thuc hien mot buoc cap nhat QMIX tu mini-batch trong replay buffer.

        Quy trinh:
          1. Sample joint batch tu buffer.
          2. Tinh Q_joint qua online Q-nets + mixing net.
          3. Tinh Q_joint_target qua target Q-nets + target mixing net.
          4. Bellman target: r + gamma * Q_joint_target * (1 - done).
          5. MSE loss, backprop, gradient clip, optimizer step.
          6. Dinh ky hard-update target networks.

        Returns:
            Loss value (float). Tra ve 0.0 neu buffer chua du du lieu.
        """
        if not self.buffer.is_ready(self.batch_size):
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        loss_val = self._compute_qmix_loss(batch)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._update_target_networks()

        return loss_val

    def _compute_qmix_loss(self, batch: JointBatch) -> float:
        """Tinh QMIX loss va thuc hien gradient step.

        Chuyen tat ca batch tensors len self.device truoc khi tinh toan
        de dam bao tinh nhat quan (CPU / CUDA / MPS).

        Args:
            batch: JointBatch tu replay buffer (tensors tren CPU).

        Returns:
            Loss value sau khi update.
        """
        B = batch.obs_all.size(0)
        N = self.n_agents

        # Chuyen toan bo batch tensors len dung device
        obs_all        = batch.obs_all.to(self.device)
        actions_all    = batch.actions_all.to(self.device)
        global_state   = batch.global_state.to(self.device)
        joint_reward   = batch.joint_reward.to(self.device)
        next_obs_all   = batch.next_obs_all.to(self.device)
        next_global_st = batch.next_global_state.to(self.device)
        done           = batch.done.to(self.device)

        # ── Online: Q_joint ────────────────────────────────────────────────
        obs_flat = obs_all.view(B * N, self.obs_dim)

        # Agent ID one-hot da tren device (pre-computed trong __init__)
        agent_ids_tiled = self._agent_id_onehot.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        agent_ids_flat  = agent_ids_tiled.reshape(B * N, N)

        obs_aug = torch.cat([obs_flat, agent_ids_flat], dim=-1)    # (B*N, input_dim)
        q_all   = self.q_net(obs_aug).view(B, N, 2)                # (B, N, 2)

        actions_exp = actions_all.unsqueeze(-1)                    # (B, N, 1)
        q_taken     = q_all.gather(2, actions_exp).squeeze(-1)     # (B, N)

        q_joint = self.mixing_net(q_taken, global_state)           # (B,)

        # ── Target: Q_joint_target (Double DQN) ───────────────────────────
        next_obs_flat = next_obs_all.view(B * N, self.obs_dim)
        next_obs_aug  = torch.cat([next_obs_flat, agent_ids_flat], dim=-1)

        with torch.no_grad():
            # Online net chon action, target net danh gia gia tri
            next_q_online  = self.q_net(next_obs_aug).view(B, N, 2)
            best_actions   = next_q_online.argmax(dim=-1, keepdim=True)   # (B, N, 1)

            next_q_target  = self.target_q_net(next_obs_aug).view(B, N, 2)
            next_q_taken   = next_q_target.gather(2, best_actions).squeeze(-1)  # (B, N)

            q_joint_target = self.target_mixing_net(next_q_taken, next_global_st)  # (B,)

            # Bellman equation
            td_target = joint_reward + self.gamma * q_joint_target * (1.0 - done)
            td_target = torch.clamp(td_target, -50.0, 50.0)

        loss = nn.functional.mse_loss(q_joint, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(loss.item())

    def _update_target_networks(self) -> None:
        """Hard-update target networks tu online networks."""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def decay_epsilon(self) -> None:
        """Giam epsilon theo lich nhan. Goi moi step sau khi update."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ─── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Luu toan bo trang thai QMIX agent vao file .pth.

        Luu bao gom: hyperparameters + state_dicts cua Q-net va Mixing net.
        Replay buffer va target nets KHONG duoc luu (tai tao khi load).
        Weights luon duoc luu tren CPU de co the load tren bat ky device nao.

        Args:
            path: Duong dan file dau ra (.pth).
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            # Metadata kien truc
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
            # Weights – luu tren CPU de load duoc tren bat ky device nao
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
        """Tai QMIX agent tu file .pth.

        Neu file khong ton tai hoac bi loi -> khoi tao agent moi.
        Weights duoc load len CPU truoc, agent tu dong dua len device phat hien.

        Args:
            path             : Duong dan file .pth.
            default_n_agents : n_agents dung khi khong tim thay file.
            default_obs_dim  : obs_dim dung khi khong tim thay file.
            **kwargs         : Override hyperparameters (vd: learning_rate=0.001).

        Returns:
            QMIXAgent da restore weights, hoac agent moi neu loi/khong co file.
        """
        source_path = Path(path)
        if not source_path.exists():
            print(f"[INFO] Khong tim thay QMIX model tai '{path}'. Khoi tao moi.")
            return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)

        try:
            # map_location="cpu": load weights ve CPU truoc, agent se dua len device sau
            payload = torch.load(source_path, map_location="cpu", weights_only=False)

            # Kiem tra algorithm tag
            if payload.get("algorithm") != "QMIX":
                print(f"[WARNING] File '{path}' khong phai QMIX model. Khoi tao moi.")
                return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)

            agent = cls(
                n_agents=int(payload["n_agents"]),
                obs_dim=int(payload["obs_dim"]),
                layout=kwargs.get("layout"),
                connections=kwargs.get("connections"),
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
            # load_state_dict roi dua len device (agent.__init__ da goi .to(device) roi)
            agent.q_net.load_state_dict(payload["q_net_state_dict"])
            agent.target_q_net.load_state_dict(payload["q_net_state_dict"])
            agent.mixing_net.load_state_dict(payload["mixing_net_state_dict"])
            agent.target_mixing_net.load_state_dict(payload["mixing_net_state_dict"])
            print(f"[INFO] QMIX model loaded from '{path}' -> device={agent.device}")
            return agent
        except Exception as exc:
            print(f"[WARNING] Khong the tai QMIX model tu '{path}' (loi: {exc}). Khoi tao moi.")
            return cls(n_agents=default_n_agents, obs_dim=default_obs_dim, **kwargs)
