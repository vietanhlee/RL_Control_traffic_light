# RL Agent (QMIX Cooperative MARL)

Thư mục này chứa agent học máy tăng cường đa tác tử hợp tác (Cooperative Multi-Agent Reinforcement Learning) sử dụng thuật toán **QMIX** chạy độc lập để điều khiển các pha đèn tín hiệu giao thông qua API của Backend.

## Kiểu mô hình (Model Architecture)

Hệ thống sử dụng thuật toán **QMIX** theo kiến trúc **CTDE (Centralized Training, Decentralized Execution)**:
- **Decentralized Execution (Chạy độc lập):** Mỗi agent ngã tư sử dụng mạng cục bộ `IndividualQNet` (kiến trúc Dueling DQN có parameter sharing và nhúng Agent ID one-hot) để tự quyết định hành động `Keep (0) / Change (1)` dựa trên local observation của riêng nó.
- **Centralized Training (Huấn luyện tập trung):** Mixing Network sử dụng một Hypernetwork nhận đầu vào là Global State (concatenation của tất cả local observations) để ước tính giá trị Q-joint từ các Q-values cục bộ của 16 agents, phục vụ cho việc tính loss và huấn luyện đồng bộ.
- Mạng nơ-ron được xây dựng hoàn toàn trên **PyTorch** và xử lý mảng bằng **NumPy**, tự động chạy trên GPU (CUDA/MPS) nếu có.

## Cấu trúc thư mục

```
rl_agent/
├── train.py                     # Vòng lặp huấn luyện online QMIX
├── evaluate.py                  # Chạy đánh giá policy đã huấn luyện ở chế độ greedy
├── artifacts/
│   └── qmix_agent.pth           # Checkpoint lưu trữ trọng số model (Q-net + Mixing net)
└── traffic_rl/
    ├── __init__.py
    ├── agent.py                 # Định nghĩa QMIXAgent (chính)
    ├── mixing_net.py            # Mạng Mixing Network (hypernetwork)
    ├── joint_buffer.py          # Bộ đệm trải nghiệm JointReplayBuffer
    ├── environment.py           # Định nghĩa TrafficEnvironment kết nối với API Backend
    ├── features.py              # Xây dựng vector đặc trưng từ API response
    ├── client.py                # REST API client gửi request lên Backend
    └── config.py                # Hyperparameters cấu hình mô hình
```

## Cách chạy thử nghiệm (Người dùng tự chạy)

1. Khởi động Backend API:
   ```bash
   cd backend/app
   python main.py
   ```

2. Huấn luyện Agent (chạy từ thư mục gốc dự án):
   ```bash
   .venv/Scripts/python -m rl_agent.train --steps 5000
   ```

3. Chạy đánh giá Agent (chạy từ thư mục gốc dự án):
   ```bash
   .venv/Scripts/python -m rl_agent.evaluate
   ```

## Reset Episode

Trước mỗi episode huấn luyện, script huấn luyện sẽ tự động gửi request đến API `POST /api/v1/reset` của Backend để làm sạch trạng thái xe cộ, đưa thời gian mô phỏng về 0 để đảm bảo tính nhất quán của môi trường.
