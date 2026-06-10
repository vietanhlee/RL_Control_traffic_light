"""
mixing_net.py – QMIX Mixing Network với Hypernetwork tích hợp GNN (Graph Neural Network).

Kiến trúc:
  Mixing Network nhận Q-values cục bộ từ tất cả N agents và global state,
  tổng hợp thành một Q_joint duy nhất để tính TD loss trong training.

  Thay vì dùng chuỗi trạng thái phẳng (flattened global state), mô hình này sử dụng
  Graph Convolutional Network (GCN) để xử lý cấu trúc đồ thị thực tế của 16 nút giao,
  giúp mô hình học được sự liên kết vật lý và không gian (luồng xe đi từ nút này sang nút khác).

Ràng buộc monotonicity (IGM condition):
  ∂Q_joint / ∂Q_i ≥ 0  ∀i
  → weights W1, W2 của mixing net được sinh bởi hypernetwork và ép về ≥ 0 bằng hàm abs().

Pipeline:
  obs_all (B, N, obs_dim) → [GCN Layers] → node_embeddings (B, N, G_dim)
                          → [Mean Pooling] → graph_embedding (B, G_dim)
                          → [HyperNet] → W1, b1, W2, b2
  q_agents (B, N) + W1, b1, W2, b2 → Q_joint (B,)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Lớp tích chập đồ thị (Graph Convolutional Network Layer) thuần túy bằng PyTorch.

    Công thức: Output = ReLU(A * X * W)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass cho GCN Layer.

        Args:
            x   : Đặc trưng của các Node. Shape: (batch_size, n_agents, in_features).
            adj : Ma trận kề chuẩn hóa. Shape: (n_agents, n_agents) hoặc (batch_size, n_agents, n_agents).

        Returns:
            Tensor đặc trưng Node mới. Shape: (batch_size, n_agents, out_features).
        """
        # Nếu ma trận kề là 2D, mở rộng chiều Batch để nhân ma trận theo lô (torch.bmm)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Nhân ma trận kề với đặc trưng Node để tổng hợp thông tin từ hàng xóm
        support = torch.bmm(adj, x)  # (batch_size, n_agents, in_features)
        return F.relu(self.projection(support))


class MixingNetwork(nn.Module):
    """Mixing Network tích hợp GNN cho QMIX.

    Sử dụng GCN để sinh Graph Embedding từ local observations của các nút giao
    và topology đường đi, sau đó đưa vào Hypernetwork để sinh weights cho Mixing Layers.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        mixing_hidden_dim: int = 32,
        gnn_hidden_dim: int = 64,
        adj: torch.Tensor | None = None,
    ) -> None:
        """Khởi tạo Mixing Network.

        Args:
            n_agents          : Số lượng agents (16 nút giao).
            obs_dim           : Chiều local observation của mỗi nút giao (Node Features dim).
            mixing_hidden_dim : Chiều hidden của Mixing Layers (mặc định 32).
            gnn_hidden_dim    : Chiều hidden của các lớp GCN (mặc định 64).
            adj               : Ma trận kề chuẩn hóa. Nếu None, mặc định là ma trận đơn vị.
        """
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        # Đăng ký ma trận kề làm buffer để PyTorch tự quản lý device (CPU/GPU)
        if adj is not None:
            self.register_buffer("adj", adj)
        else:
            self.register_buffer("adj", torch.eye(n_agents))

        # ── Hai lớp GCN để tổng hợp thông tin không gian (2-hops) ───────────
        self.gcn1 = GCNLayer(obs_dim, gnn_hidden_dim)
        self.gcn2 = GCNLayer(gnn_hidden_dim, gnn_hidden_dim)

        # ── Hypernetwork cho lớp 1 ──────────────────────────────────────────
        # Nhận đầu vào là Graph Embedding kích thước (gnn_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents * mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(gnn_hidden_dim, mixing_hidden_dim)

        # ── Hypernetwork cho lớp 2 ──────────────────────────────────────────
        self.hyper_w2 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, mixing_hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )

    def forward(
        self,
        q_agents: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """Tính Q_joint từ Q-values cá nhân và global state bằng GNN.

        Args:
            q_agents     : Q-value của action đã chọn cho từng agent.
                           Shape: (batch_size, n_agents).
            global_state : Vector trạng thái toàn cục phẳng (flattened).
                           Shape: (batch_size, n_agents * obs_dim).

        Returns:
            Q_joint: Scalar Q-value tổng hợp. Shape: (batch_size,).
        """
        batch = q_agents.size(0)

        # 1. Khôi phục trạng thái phẳng về dạng đồ thị: (B, n_agents, obs_dim)
        x = global_state.view(batch, self.n_agents, self.obs_dim)

        # 2. Chạy qua các lớp GNN
        h = self.gcn1(x, self.adj)  # (B, n_agents, gnn_hidden_dim)
        h = self.gcn2(h, self.adj)  # (B, n_agents, gnn_hidden_dim)

        # 3. Pooling (Mean) trên chiều Node để thu được Graph Embedding
        graph_embed = h.mean(dim=1)  # (B, gnn_hidden_dim)

        # ── Layer 1 ─────────────────────────────────────────────────────────
        # Sinh W1 và b1 từ Graph Embedding
        w1 = torch.abs(self.hyper_w1(graph_embed))
        w1 = w1.view(batch, self.n_agents, self.mixing_hidden_dim)

        b1 = self.hyper_b1(graph_embed)
        b1 = b1.view(batch, 1, self.mixing_hidden_dim)

        # Nhân ma trận: Q_in (B, 1, N) * W1 (B, N, H) + b1 (B, 1, H)
        q_in = q_agents.unsqueeze(1)  # (B, 1, N)
        hidden = F.elu(torch.bmm(q_in, w1) + b1)  # (B, 1, H)

        # ── Layer 2 ─────────────────────────────────────────────────────────
        # Sinh W2 và b2 từ Graph Embedding
        w2 = torch.abs(self.hyper_w2(graph_embed))
        w2 = w2.view(batch, self.mixing_hidden_dim, 1)

        b2 = self.hyper_b2(graph_embed)
        b2 = b2.view(batch, 1, 1)

        # Nhân ma trận: hidden (B, 1, H) * W2 (B, H, 1) + b2 (B, 1, 1)
        q_joint = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_joint.squeeze(-1).squeeze(-1)  # (B,)
