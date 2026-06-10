from math import acos
from typing import Tuple

def calculate_turn_type(
    in_pos: Tuple[float, float],
    via_pos: Tuple[float, float],
    out_pos: Tuple[float, float]
) -> str:
    """Xác định hướng rẽ (straight, left, right) tại một nút giao dựa trên tọa độ 3 điểm.

    Args:
        in_pos: Tọa độ nút bắt đầu (x, y).
        via_pos: Tọa độ nút giao (x, y).
        out_pos: Tọa độ nút tiếp theo (x, y).

    Returns:
        Chuỗi kết quả: "straight", "left", hoặc "right".
    """
    v1 = (via_pos[0] - in_pos[0], via_pos[1] - in_pos[1])
    v2 = (out_pos[0] - via_pos[0], out_pos[1] - via_pos[1])

    norm1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    norm2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if norm1 <= 1e-6 or norm2 <= 1e-6:
        return "straight"

    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
    dot = max(-1.0, min(1.0, dot))
    angle = acos(dot)
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if angle < 0.5:
        return "straight"
    return "left" if cross > 0 else "right"
