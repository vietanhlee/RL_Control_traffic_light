"""
environment.py – Định nghĩa môi trường RL giao tiếp với backend simulation.

Module này cung cấp lớp TrafficEnvironment đóng vai trò là "interface" giữa
RL agent và backend simulation, tuân theo giao thức Observe → Act → Reward.

Giao thức:
  1. bootstrap() / reset()   : Khởi tạo môi trường, lấy danh sách nút giao.
  2. observe_all()            : Lấy observation từ tất cả nút giao.
  3. apply_actions(actions)   : Gửi hành động lên backend.
  4. advance()                : Chờ decision_interval_seconds → trả về observation mới.
  5. reward_for(obs, action)  : Tính phần thưởng cho một nút giao từ observation.

Reward function:
  Reward = reward_offset − (queue_penalty + imbalance_penalty + red_pressure_penalty
           + switch_penalty − speed_bonus)
  Được clip vào khoảng [-reward_clip, +reward_clip].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .client import TrafficApiClient
from .config import RewardWeights


@dataclass
class IntersectionRecord:
    """Theo dõi trạng thái thời gian của một nút giao.

    Được dùng để kiểm soát minimum phase hold – ngăn agent chuyển đèn
    quá nhanh (flapping) gây bất ổn cho giao thông thực tế.

    Attributes:
        last_switch_step  : Bước (step) cuối cùng agent thực hiện hành động CHANGE.
        steps_since_switch: Số bước đã trôi qua kể từ lần chuyển pha cuối.
                            Giá trị mặc định lớn (9999) để cho phép switch ngay từ đầu.
    """

    last_switch_step: int = -9999
    steps_since_switch: int = 9999


@dataclass
class TrafficEnvironment:
    """Môi trường RL kết nối với backend simulation qua REST API.

    Tuân theo giao thức Gym-like (observe → act → reward) nhưng không
    kế thừa gym.Env để giữ dependency tối thiểu.

    Attributes:
        client                    : Client HTTP để gọi backend API.
        decision_interval_seconds : Thời gian chờ giữa hai step (giây).
                                    Phải khớp với chu kỳ simulation.
        reward_weights            : Các trọng số và hệ số trong reward function.
        min_phase_hold_steps      : Số step tối thiểu giữ nguyên pha sau khi CHANGE.
        intersection_ids          : Danh sách ID các nút giao trong mạng.
        records                   : Dict lưu IntersectionRecord cho từng nút giao.
    """

    client: TrafficApiClient
    decision_interval_seconds: float = 1.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    min_phase_hold_steps: int = 2
    intersection_ids: list[int] = field(default_factory=list)
    records: dict[int, IntersectionRecord] = field(default_factory=dict)

    def bootstrap(self) -> list[int]:
        """Lấy danh sách nút giao từ backend và khởi tạo records.

        Gọi một lần sau khi kết nối để biết cấu trúc mạng.

        Returns:
            Danh sách ID nút giao đã được sắp xếp tăng dần.

        Raises:
            RuntimeError: Nếu backend trả về mạng rỗng hoặc sai định dạng.
        """
        network = self.client.get_network()
        nodes = network.get("nodes", {})
        if not isinstance(nodes, dict) or not nodes:
            raise RuntimeError("Backend network response is empty")

        self.intersection_ids = sorted(int(node_id) for node_id in nodes.keys())
        self.records = {intersection_id: IntersectionRecord() for intersection_id in self.intersection_ids}
        return self.intersection_ids

    def reset(self) -> None:
        """Reset hoàn toàn simulation về trạng thái ban đầu.

        Gọi endpoint /reset trên backend và bootstrap lại danh sách nút giao.
        Dùng khi muốn bắt đầu training từ đầu (không kế thừa trạng thái xe).
        """
        self.client.reset()
        self.bootstrap()

    def observe(self, intersection_id: int) -> dict[str, Any]:
        """Lấy observation chi tiết cho một nút giao cụ thể.

        Args:
            intersection_id: ID của nút giao muốn quan sát.

        Returns:
            Dict chứa: directions, light_states, timings, imbalance, v.v.
        """
        return self.client.get_state(intersection_id)

    def observe_all(self) -> dict[int, dict[str, Any]]:
        """Lấy observation từ tất cả nút giao trong mạng.

        Nếu intersection_ids chưa được khởi tạo, tự động gọi bootstrap().

        Returns:
            Dict mapping intersection_id → observation dict.
        """
        if not self.intersection_ids:
            self.bootstrap()
        return {intersection_id: self.observe(intersection_id) for intersection_id in self.intersection_ids}

    # ─── Các hàm tính feature phụ trợ cho reward ──────────────────────────────

    def _queue_total(self, observation: dict[str, Any]) -> float:
        """Tính tổng queue (số xe chờ) từ tất cả hướng vào của nút giao.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tổng queue_length của tất cả hướng vào (float).
        """
        directions = observation.get("directions", {})
        if not isinstance(directions, dict):
            return 0.0
        total = 0.0
        for payload in directions.values():
            if isinstance(payload, dict):
                total += float(payload.get("queue_length", 0.0))
        return total

    def _density_total(self, observation: dict[str, Any]) -> float:
        """Tính tổng mật độ phương tiện (motorcycle + car) từ tất cả hướng vào.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tổng mật độ (xe/m) trung bình từ tất cả hướng.
        """
        directions = observation.get("directions", {})
        if not isinstance(directions, dict):
            return 0.0
        total = 0.0
        for payload in directions.values():
            if isinstance(payload, dict):
                total += float(payload.get("motorcycle_density", 0.0))
                total += float(payload.get("car_density", 0.0))
        return total

    def _speed_average(self, observation: dict[str, Any]) -> float:
        """Tính tốc độ trung bình tất cả phương tiện từ tất cả hướng vào.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tốc độ trung bình (m/s). Trả về 0.0 nếu không có hướng nào.
        """
        directions = observation.get("directions", {})
        if not isinstance(directions, dict) or not directions:
            return 0.0
        total = 0.0
        count = 0
        for payload in directions.values():
            if isinstance(payload, dict):
                total += (
                    float(payload.get("motorcycle_avg_speed", 0.0))
                    + float(payload.get("car_avg_speed", 0.0))
                ) / 2.0
                count += 1
        return total / count if count else 0.0

    def _imbalance(self, observation: dict[str, Any]) -> float:
        """Tính độ mất cân bằng queue giữa các hướng vào của nút giao.

        Imbalance = sum(|queue_i - mean_queue|) cho tất cả hướng i.
        Giá trị cao → một số hướng bị tắc nghẽn nặng hơn hướng khác.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Độ mất cân bằng (float, ≥ 0).
        """
        directions = observation.get("directions", {})
        if not isinstance(directions, dict) or not directions:
            return 0.0
        queues = [
            float(payload.get("queue_length", 0.0))
            for payload in directions.values()
            if isinstance(payload, dict)
        ]
        if not queues:
            return 0.0
        avg_queue = sum(queues) / len(queues)
        return sum(abs(queue - avg_queue) for queue in queues)

    def reward_for(self, observation: dict[str, Any], action: int) -> float:
        """Tính phần thưởng cho một nút giao dựa trên trạng thái và hành động.

        Reward Function:
            cost = w_q*(queue/scale_q) + w_i*(imbalance/scale_i)
                   + w_r*(red_pressure/scale_r) + switch_penalty - w_s*(speed/scale_s)
            reward = clip(reward_offset - cost, -clip_val, +clip_val)

        Diễn giải:
          - Queue penalty     : Phạt khi nhiều xe đang chờ → giảm queue
          - Imbalance penalty : Phạt khi các hướng mất cân bằng → phân phối đều
          - Red pressure      : Phạt riêng khi đèn đỏ nhưng xe vẫn xếp hàng dài
          - Switch penalty    : Phạt khi agent thực hiện CHANGE (tránh flapping)
          - Speed bonus       : Thưởng khi xe lưu thông nhanh

        Args:
            observation: Dict observation của nút giao sau khi thực hiện hành động.
            action     : Hành động vừa áp dụng (0=Keep, 1=Change).

        Returns:
            Phần thưởng đã được clip vào [-reward_clip, +reward_clip].
        """
        queue_total = self._queue_total(observation)
        imbalance = self._imbalance(observation)
        speed_avg = self._speed_average(observation)
        red_pressure = 0.0

        # Tính red_pressure: tổng queue ở các hướng đang đèn đỏ
        light_states = observation.get("light_states", {})
        if isinstance(light_states, dict):
            for raw_key, color in light_states.items():
                if isinstance(color, str) and color.upper() == "RED":
                    try:
                        queue = float(observation["directions"][raw_key]["queue_length"])  # type: ignore[index]
                    except Exception:
                        queue = 0.0
                    if queue > 0.0:
                        red_pressure += queue

        # Chi phí chuyển pha (khuyến khích giữ pha ổn định)
        penalty = self.reward_weights.switch_penalty if action == 1 else 0.0

        cost = (
            self.reward_weights.queue * (queue_total / self.reward_weights.queue_scale)
            + self.reward_weights.imbalance * (imbalance / self.reward_weights.imbalance_scale)
            + self.reward_weights.red_pressure * (red_pressure / self.reward_weights.red_pressure_scale)
            + penalty
            - self.reward_weights.speed_bonus * (speed_avg / self.reward_weights.speed_scale)
        )
        reward = self.reward_weights.reward_offset - cost
        return max(-self.reward_weights.reward_clip, min(self.reward_weights.reward_clip, reward))

    def apply_actions(self, actions: dict[int, int]) -> None:
        """Gửi hành động lên backend và cập nhật IntersectionRecord.

        Với hành động CHANGE (1): đặt lại bộ đếm steps_since_switch về 0.
        Với hành động KEEP  (0): tăng steps_since_switch thêm 1.

        Args:
            actions: Dict mapping intersection_id → action (0 hoặc 1).
        """
        filtered_actions = {intersection_id: action for intersection_id, action in actions.items() if action in (0, 1)}
        if filtered_actions:
            self.client.post_actions(filtered_actions)

        for intersection_id, action in filtered_actions.items():
            record = self.records.setdefault(intersection_id, IntersectionRecord())
            if action == 1:
                record.last_switch_step = record.steps_since_switch
                record.steps_since_switch = 0
            else:
                record.steps_since_switch += 1

    def hold_required(self, intersection_id: int) -> bool:
        """Kiểm tra xem nút giao có đang trong giai đoạn bắt buộc giữ pha không.

        Ngăn agent chuyển đèn quá nhanh ngay sau lần CHANGE trước đó.
        Nếu steps_since_switch < min_phase_hold_steps → buộc action = 0 (KEEP).

        Args:
            intersection_id: ID nút giao cần kiểm tra.

        Returns:
            True nếu nút giao phải giữ pha (action bị override về 0).
        """
        record = self.records.setdefault(intersection_id, IntersectionRecord())
        return record.steps_since_switch < self.min_phase_hold_steps

    def advance(self) -> dict[int, dict[str, Any]]:
        """Chờ một decision interval rồi trả về observation mới.

        Gọi `time.sleep(decision_interval_seconds)` để đồng bộ với
        chu kỳ simulation backend (thường 0.3s = 3 steps × 0.1s/step).
        Sau đó tăng steps_since_switch cho tất cả nút giao.

        Returns:
            Dict mapping intersection_id → observation dict mới nhất.
        """
        time.sleep(self.decision_interval_seconds)
        # Tăng counter cho tất cả nút giao (thời gian trôi qua)
        for record in self.records.values():
            record.steps_since_switch += 1
        return self.observe_all()
