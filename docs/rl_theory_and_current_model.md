# Lý Thuyết Reinforcement Learning và Model Hiện Tại (QMIX)

## 1. Reinforcement Learning Là Gì?

Reinforcement Learning (RL) là hướng học máy trong đó agent học ra quyết định bằng cách tương tác với môi trường. Không có nhãn đáp án cố định – agent thực hiện hành động, nhận reward, rồi tự điều chỉnh chiến lược để tối đa hóa tổng reward dài hạn.

Một bài toán RL gồm 5 thành phần:
- `State (s)`: trạng thái môi trường
- `Action (a)`: hành động agent thực hiện
- `Reward (r)`: tín hiệu thưởng/phạt
- `Policy (π)`: chiến lược chọn hành động
- `Return (G)`: tổng reward tích lũy theo thời gian

---

## 2. Công Thức Cơ Bản

```text
G_t = r_{t+1} + γr_{t+2} + γ^2r_{t+3} + ...
```
- `γ` gần 1 → ưu tiên dài hạn; `γ` nhỏ → ưu tiên ngắn hạn.

---

## 3. Các Nhóm Thuật Toán RL

### 3.1. Value-Based Methods

#### Q-Learning
```text
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s', a') - Q(s,a)]
```

#### Deep Q-Network (DQN)
Thay bảng Q bằng Neural Network. Đầu vào: state features. Đầu ra: Q-value mỗi action.

#### Dueling DQN
Chia output thành 2 nhánh:
```text
Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
```
Ổn định hơn DQN chuẩn, đặc biệt ở các trạng thái "nhàn rỗi".

### 3.2. Multi-Agent Value-Based Methods

#### QMIX ← **Model hiện tại đang dùng**

QMIX (Monotonic Value Function Factorisation) là thuật toán MARL hợp tác theo kiến trúc **CTDE (Centralized Training, Decentralized Execution)**:

- **Decentralized Execution**: Mỗi agent chỉ cần local obs để ra quyết định
- **Centralized Training**: Mixing Network kết hợp Q_i của tất cả agents → Q_joint để tính loss

Công thức:
```text
Q_joint = MixNet(Q_1(o_1,a_1), ..., Q_N(o_N,a_N) | global_state)

Ràng buộc: ∂Q_joint / ∂Q_i ≥ 0 (monotonicity → IGM condition)
```

Ý nghĩa monotonicity: `argmax_a Q_joint = (argmax_a Q_1, ..., argmax_a Q_N)` → agents có thể hành động greedy độc lập mà vẫn đảm bảo tối ưu joint.

### 3.3. Policy-Based Methods

#### PPO (Proximal Policy Optimization)
Actor-Critic on-policy, dùng clipping ratio để ổn định update policy. Không dùng replay buffer.

#### MAPPO
PPO mở rộng cho MARL với centralized critic nhận global state.

---

## 4. Các Thành Phần Quan Trọng

### 4.1. Exploration vs Exploitation
Cân bằng qua **ε-greedy**:
- Xác suất `ε` → chọn ngẫu nhiên (exploration)
- Xác suất `1-ε` → chọn argmax Q (exploitation)
- `ε` giảm dần theo thời gian training

### 4.2. Reward Design
Với bài toán giao thông, reward cục bộ mỗi nút giao:
```text
cost = w_q*(queue_total/scale_q) + w_i*(imbalance/scale_i)
       + w_r*(red_pressure/scale_r) + switch_penalty - w_s*(avg_speed/scale_s)
reward = clip(reward_offset - cost, -5, +5)
```
**Joint reward cho QMIX** = `mean(reward_i for i in 16 agents)` → cooperative signal.

### 4.3. Experience Replay
Lưu transitions vào buffer, sample ngẫu nhiên → phá vỡ temporal correlation.

**QMIX dùng JointReplayBuffer**: mỗi entry là transition của **tất cả 16 agents cùng lúc**.

---

## 5. Model Hiện Tại: QMIX

### 5.1. Kiến Trúc

```
┌─────────────────────────────────────────────┐
│           DECENTRALIZED EXECUTION            │
│                                              │
│  Agent 1: Q_1(o_1 ⊕ id_1) → a_1            │
│  Agent 2: Q_2(o_2 ⊕ id_2) → a_2            │  ← Shared IndividualQNet
│  ...                                         │    (1 mạng dùng chung)
│  Agent 16: Q_16(o_16 ⊕ id_16) → a_16       │
└─────────────────────────────────────────────┘
                      ↓ (only during training)
┌─────────────────────────────────────────────┐
│           CENTRALIZED TRAINING               │
│                                              │
│  [Q_1, Q_2, ..., Q_16]                      │
│  + global_state (= flatten all local obs)   │
│           ↓                                  │
│  MixingNetwork (hypernetwork)               │
│           ↓                                  │
│       Q_joint (scalar)                       │
│           ↓                                  │
│  TD Loss = MSE(Q_joint, r + γ*Q_joint_tgt)  │
└─────────────────────────────────────────────┘
```

### 5.2. Chi Tiết Kỹ Thuật

| Thành phần | Chi tiết |
|---|---|
| **Q-network** | IndividualQNet (Dueling architecture, shared weights) |
| **Input dim** | `obs_dim (40) + n_agents (16) = 56` (+ one-hot agent ID) |
| **Global state dim** | `16 × 40 = 640` (flatten tất cả local obs) |
| **Mixing hidden dim** | 32 |
| **Joint Replay Buffer** | 5,000 joint transitions (pre-allocated numpy) |
| **Batch size** | 32 joint transitions |
| **Target update** | Hard-update mỗi 50 gradient steps |
| **Loss function** | MSE(Q_joint, Bellman target) |
| **Optimizer** | Adam (lr=0.0005) cho cả Q-net + Mixing net |
| **Double DQN** | Online net chọn action, Target net đánh giá giá trị |
| **Gradient clip** | max_norm=10.0 |

### 5.3. Hyperparameters Mặc Định

```python
DEFAULT_LR               = 0.0005
DEFAULT_GAMMA            = 0.98
DEFAULT_EPSILON          = 1.0
DEFAULT_MIN_EPSILON      = 0.05
DEFAULT_EPSILON_DECAY    = 0.9995
DEFAULT_BATCH_SIZE       = 32
DEFAULT_BUFFER_CAPACITY  = 5000
DEFAULT_TARGET_UPDATE_FREQ = 50
DEFAULT_HIDDEN_DIM       = 128
DEFAULT_MIXING_HIDDEN_DIM = 32
```

### 5.4. Files Cấu Thành

```
rl_agent/
├── train.py                      ← Vòng lặp training QMIX
├── evaluate.py                   ← Đánh giá policy đã train
├── artifacts/
│   └── qmix_agent.pth            ← Model checkpoint (Q-net + Mixing net)
└── traffic_rl/
    ├── agent.py                  ← QMIXAgent (chính)
    ├── mixing_net.py             ← MixingNetwork (hypernetwork)
    ├── joint_buffer.py           ← JointReplayBuffer
    ├── environment.py            ← TrafficEnvironment (observe/reward/advance)
    ├── features.py               ← build_features() (local obs → feature vector)
    ├── client.py                 ← REST API client
    └── config.py                 ← Hyperparameters & RewardWeights
```

---

## 6. Lý Do Chọn QMIX

| Tiêu chí | QMIX | Dueling DQN (cũ) |
|---|---|---|
| **Phối hợp agents** | ✅ Mixing Network | ❌ Không có |
| **IGM guarantee** | ✅ Đảm bảo | ❌ Không |
| **Off-policy** | ✅ Replay buffer | ✅ |
| **Sample efficiency** | ✅ Cao | ✅ |
| **Phù hợp cooperative** | ✅ Rất tốt | ❌ Kém |

---

## 7. Định Hướng Tương Lai

### Bước tiếp theo

1. **Prioritized Experience Replay (PER)**: Ưu tiên sample các transition có TD error lớn → tăng tốc độ hội tụ.

2. **QMIX + GNN (Graph Neural Network)**: Thay `global_state = flatten all obs` bằng graph message passing theo topology mạng lưới đường → agent biết chính xác "hàng xóm" của mình.

3. **QPLEX**: Mở rộng QMIX cho phép Q_joint là linear combination phức tạp hơn (không chỉ monotone), tổng quát hơn cho các bài toán có Q_joint không factorizable đơn giản.

### Lộ trình
```
Bước 1 (hoàn thành): Dueling DQN + Parameter Sharing
Bước 2 (hoàn thành): QMIX + Shared Q-net + Agent ID embedding
Bước 3 (tương lai):  QMIX + PER
Bước 4 (tương lai):  QMIX + GNN (Graph-based global state)
Bước 5 (tương lai):  QPLEX hoặc MAPPO để so sánh
```

---

## 8. Hướng Dẫn Huấn Luyện

### Chạy training mới
```bash
# Đứng từ thư mục gốc dự án
.venv\Scripts\python -m rl_agent.train --steps 5000
```

### Fine-tune từ checkpoint
```bash
# Đứng từ thư mục gốc dự án
.venv\Scripts\python -m rl_agent.train \
  --model-path rl_agent/artifacts/qmix_agent.pth \
  --steps 2000 \
  --epsilon 0.1 \
  --lr 0.0002 \
  --epsilon-decay 0.9999
```

### Evaluate policy
```bash
# Đứng từ thư mục gốc dự án
.venv\Scripts\python -m rl_agent.evaluate \
  --model-path rl_agent/artifacts/qmix_agent.pth \
  --steps 500
```

### Các tham số CLI quan trọng
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--steps` | 5000 | Tổng số training steps |
| `--lr` | 0.0005 | Learning rate |
| `--epsilon` | 1.0 | Epsilon ban đầu |
| `--min-epsilon` | 0.05 | Epsilon tối thiểu |
| `--batch-size` | 32 | Kích thước mini-batch |
| `--mixing-hidden-dim` | 32 | Hidden dim mixing net |
| `--target-update-freq` | 50 | Tần suất hard-update target nets |
| `--reset-first` | False | Reset simulation trước khi train |
| `--no-explore` | False | Tắt ε-greedy (pure greedy) |

---

## 9. Kết Luận

Cấu trúc RL hiện tại của hệ thống:

| Thành phần | Giá trị |
|---|---|
| **Algorithm** | QMIX (Cooperative MARL, CTDE) |
| **Environment** | Backend traffic simulator (FastAPI + WebSocket) |
| **State (local)** | queue, density, speed, imbalance, light states (40 features/agent) |
| **State (global)** | Flatten tất cả local obs của 16 agents (640 dim) |
| **Action** | `Keep (0) / Change (1)` cho từng nút giao |
| **Reward** | Mean of individual rewards (cooperative) |
| **Agent** | QMIXAgent: Shared IndividualQNet (Dueling) + MixingNetwork |
| **n_agents** | 16 nút giao thông |
