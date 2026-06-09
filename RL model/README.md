# RL model

Thư mục này chứa agent RL chạy độc lập để điều khiển pha đèn qua API hiện có của backend.

## Kiểu model

Mình chọn `Approximate Q-Learning` tuyến tính:
- Action space nhỏ: `0 = Keep`, `1 = Change`
- State là các đặc trưng giao thông liên tục
- Không cần thư viện ngoài như `torch` hay `numpy`
- Có thể train online trực tiếp với backend đang chạy

## Cấu trúc

- `train.py`: chạy train online và lưu model ra JSON
- `evaluate.py`: chạy policy đã học ở chế độ greedy
- `traffic_rl/`: package chứa client HTTP, feature engineering, agent và environment

## Cách chạy

1. Khởi động backend FastAPI như bình thường.
2. Chạy train:

```bash
./.venv/Scripts/python.exe "RL model/train.py" --steps 5000
```

3. Chạy đánh giá:

```bash
./.venv/Scripts/python.exe "RL model/evaluate.py"
```

## Gợi ý

Nếu muốn train theo episode, hãy dùng endpoint `POST /api/v1/reset` trước mỗi episode.
