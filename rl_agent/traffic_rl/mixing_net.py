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


class GATLayer(nn.Module):
    """Lớp Graph Attention Network (GAT) Layer thuần túy bằng PyTorch.

    Công thức: Output = activation(Attention(A, X) * X * W)
    Sử dụng Multi-Head Attention tính song song bằng torch.einsum để đạt hiệu năng tối ưu.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Khởi tạo ma trận chiếu W cho các head: shape (n_heads, in_features, out_features)
        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Khởi tạo tham số attention vector a cho mỗi head: shape (n_heads, 2 * out_features, 1)
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass cho GAT Layer.

        Args:
            h   : Đặc trưng đầu vào của Node. Shape: (batch_size, n_agents, in_features).
            adj : Ma trận kề chuẩn hóa. Shape: (n_agents, n_agents) hoặc (batch_size, n_agents, n_agents).

        Returns:
            Tensor đặc trưng Node sau Attention.
            Nếu concat=True: Shape (batch_size, n_agents, n_heads * out_features).
            Nếu concat=False: Shape (batch_size, n_agents, out_features).
        """
        B, N, _ = h.size()
        K = self.n_heads

        # 1. Chiếu đặc trưng sang không gian mới cho từng head song song
        # h_exp: (B, K, N, in_features)
        h_exp = h.unsqueeze(1).expand(-1, K, -1, -1)
        # h_prime: (B, K, N, F_out) qua phép nhân einsum
        h_prime = torch.einsum("bknf, kfo -> bkno", h_exp, self.W)

        # 2. Chuẩn bị đầu vào tính Attention hệ số (cặp i và j)
        # h_prime_i: (B, K, N, N, F_out)
        # h_prime_j: (B, K, N, N, F_out)
        h_prime_i = h_prime.unsqueeze(3).expand(-1, -1, -1, N, -1)
        h_prime_j = h_prime.unsqueeze(2).expand(-1, -1, N, -1, -1)

        # Concat đặc trưng cặp Node: (B, K, N, N, 2 * F_out)
        a_input = torch.cat([h_prime_i, h_prime_j], dim=-1)

        # 3. Tính hệ số attention thô e_ij = LeakyReLU(a_input @ a) -> (B, K, N, N)
        e = torch.einsum("bkijd, kdo -> bkijo", a_input, self.a).squeeze(-1)
        e = self.leakyrelu(e)

        # 4. Masking: Chỉ tính attention cho các node kề nhau (adj_ij > 0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).unsqueeze(1).expand(B, K, -1, -1)
        elif adj.dim() == 3:
            adj = adj.unsqueeze(1).expand(-1, K, -1, -1)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax để chuẩn hóa attention trên các node kề
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 5. Tổng hợp đặc trưng mới: (B, K, N, N) @ (B, K, N, F_out) -> (B, K, N, F_out)
        h_out = torch.matmul(attention, h_prime)

        # 6. Trả về kết quả ghép nối hoặc trung bình
        if self.concat:
            # Ghép nối các đầu chú ý: (B, N, K * F_out)
            h_out = h_out.permute(0, 2, 1, 3).reshape(B, N, K * self.out_features)
            return F.elu(h_out)
        else:
            # Trung bình các đầu chú ý: (B, N, F_out)
            h_out = h_out.mean(dim=1)
            return h_out


class MixingNetwork(nn.Module):
    """Mixing Network tích hợp GNN (phiên bản GAT 3 tầng sâu nâng cao cực đại) cho QMIX.

    Sử dụng 3 tầng Graph Attention Network để sinh Graph Embedding từ local observations của các nút giao
    và topology đường đi, sau đó đưa vào Hypernetwork 3 lớp ẩn để sinh weights cho Mixing Layers.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        mixing_hidden_dim: int = 256,
        gnn_hidden_dim: int = 128,
        gat_heads: int = 16,
        dropout: float = 0.0,
        adj: torch.Tensor | None = None,
    ) -> None:
        """Khởi tạo Mixing Network.

        Args:
            n_agents          : Số lượng agents (16 nút giao).
            obs_dim           : Chiều local observation của mỗi nút giao (Node Features dim).
            mixing_hidden_dim : Chiều hidden của Mixing Layers (mặc định 256).
            gnn_hidden_dim    : Chiều hidden của các lớp GAT (mặc định 128).
            gat_heads         : Số lượng đầu chú ý Multi-Head Attention lớp thứ nhất (mặc định 16).
            dropout           : Tỷ lệ Dropout cho Attention weights.
            adj               : Ma trận kề chuẩn hóa. Nếu None, mặc định là ma trận đơn vị.
        """
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gat_heads = gat_heads

        # Đăng ký ma trận kề làm buffer để PyTorch tự quản lý device (CPU/GPU)
        if adj is not None:
            self.register_buffer("adj", adj)
        else:
            self.register_buffer("adj", torch.eye(n_agents))

        # ── Ba lớp GAT để tổng hợp thông tin không gian sâu hơn (3-hops) ──────
        heads1 = gat_heads
        heads2 = max(1, gat_heads // 2)
        heads3 = max(1, gat_heads // 4)

        # Lớp 1: output = heads1 * gnn_hidden_dim (16 * 128 = 2048)
        self.gat1 = GATLayer(obs_dim, gnn_hidden_dim, n_heads=heads1, dropout=dropout, concat=True)
        # Lớp 2: output = heads2 * gnn_hidden_dim (8 * 128 = 1024)
        self.gat2 = GATLayer(heads1 * gnn_hidden_dim, gnn_hidden_dim, n_heads=heads2, dropout=dropout, concat=True)
        # Lớp 3: output = gnn_hidden_dim (128) do concat=False (lấy trung bình các heads)
        self.gat3 = GATLayer(heads2 * gnn_hidden_dim, gnn_hidden_dim, n_heads=heads3, dropout=dropout, concat=False)

        # ── Hypernetwork cho lớp 1 (3 lớp ẩn, 512 neuron) ─────────────────────
        self.hyper_w1 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_agents * mixing_hidden_dim),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, mixing_hidden_dim),
        )

        # ── Hypernetwork cho lớp 2 (3 lớp ẩn, 512 neuron) ─────────────────────
        self.hyper_w2 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, mixing_hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        q_agents: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """Tính Q_joint từ Q-values cá nhân và global state bằng GAT 3 tầng.

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

        # 2. Chạy qua các lớp GAT 3 tầng
        h = self.gat1(x, self.adj)   # (B, n_agents, heads1 * gnn_hidden_dim)
        h = self.gat2(h, self.adj)   # (B, n_agents, heads2 * gnn_hidden_dim)
        h = self.gat3(h, self.adj)   # (B, n_agents, gnn_hidden_dim)

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
