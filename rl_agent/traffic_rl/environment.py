"""
environment.py â€“ Äá»‹nh nghÄ©a mÃ´i trÆ°á»ng RL giao tiáº¿p vá»›i backend simulation.

Module nÃ y cung cáº¥p lá»›p TrafficEnvironment Ä‘Ã³ng vai trÃ² lÃ  "interface" giá»¯a
RL agent vÃ  backend simulation, tuÃ¢n theo giao thá»©c Observe â†’ Act â†’ Reward.

Giao thá»©c:
  1. bootstrap() / reset()   : Khá»Ÿi táº¡o mÃ´i trÆ°á»ng, láº¥y danh sÃ¡ch nÃºt giao.
  2. observe_all()            : Láº¥y observation tá»« táº¥t cáº£ nÃºt giao.
  3. apply_actions(actions)   : Gá»�i hÃ nh Ä‘á»™ng lÃªn backend.
  4. advance()                : Chá» decision_interval_seconds â†’ tráº£ vá» observation má»›i.
  5. reward_for(obs, action)  : TÃ�nh pháº§n thÆ°á»Ÿng tá»« config.reward_fn (dÃ¹ng chung vá»›i BE).

Reward function (Single Source of Truth: config/reward_fn.py):
  reward = clip(reward_offset âˆ’ cost, âˆ’clip, +clip)
  cost   = w_q*(q/s_q) + w_i*(imbal/s_i) + w_r*(rp/s_r) + switch âˆ’ w_s*(speed/s_s)
  Náº¿u reward_raw < 0: reward_raw = âˆ’(|reward_raw|Â²)  (â†’ phi tuyáº¿n, pháº¡t táº¯c ngháº¿n náº·ng)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import TrafficApiClient
from .config import SCALE_REWARD_PRESSURE, SCALE_REWARD_DWT, W_BACKEND, W_DWT, W_PRESSURE


@dataclass
class IntersectionRecord:
    """Theo dÃµi tráº¡ng thÃ¡i thá»i gian cá»§a má»™t nÃºt giao.

    ÄÆ°á»£c dÃ¹ng Ä‘á»ƒ kiá»ƒm soÃ¡t minimum phase hold â€“ ngÄƒn agent chuyá»ƒn Ä‘Ã¨n
    quÃ¡ nhanh (flapping) gÃ¢y báº¥t á»•n cho giao thÃ´ng thá»±c táº¿.

    Attributes:
        last_switch_step  : BÆ°á»›c (step) cuá»‘i cÃ¹ng agent thá»±c hiá»‡n hÃ nh Ä‘á»™ng CHANGE.
        steps_since_switch: Sá»‘ bÆ°á»›c Ä‘Ã£ trÃ´i qua ká»ƒ tá»« láº§n chuyá»ƒn pha cuá»‘i.
                            GiÃ¡ trá»‹ máº·c Ä‘á»‹nh lá»›n (9999) Ä‘á»ƒ cho phÃ©p switch ngay tá»« Ä‘áº§u.
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
        min_phase_hold_steps      : Số step tối thiểu giữ nguyên pha sau khi CHANGE.
        intersection_ids          : Danh sách ID các nút giao trong mạng.
        records                   : Dict lưu IntersectionRecord cho từng nút giao.
    """

    client: TrafficApiClient
    decision_interval_seconds: float = 1.0
    min_phase_hold_steps: int = 2
    intersection_ids: list[int] = field(default_factory=list)
    records: dict[int, IntersectionRecord] = field(default_factory=dict)
    intersection_layout: dict[int, tuple[float, float]] = field(default_factory=dict)
    intersection_connections: list[tuple[int, int, int]] = field(default_factory=list)
    last_obs: dict[int, dict[str, Any]] = field(default_factory=dict)

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
        edges = network.get("edges", [])
        if not isinstance(nodes, dict) or not nodes:
            raise RuntimeError("Backend network response is empty")

        self.intersection_ids = sorted(int(node_id) for node_id in nodes.keys())
        self.records = {intersection_id: IntersectionRecord() for intersection_id in self.intersection_ids}
        
        # Lưu layout (x, y) và các kết nối để phục vụ cho GMIX GNN
        self.intersection_layout = {int(k): (float(v["x"]), float(v["y"])) for k, v in nodes.items()}
        self.intersection_connections = [(int(e["start"]), int(e["end"]), int(e["lanes"])) for e in edges]
        
        return self.intersection_ids

    def reset(self) -> None:
        """Reset hoàn toàn simulation về trạng thái ban đầu.

        Gọi endpoint /reset trên backend và bootstrap lại danh sách nút giao.
        Dùng khi muốn bắt đầu training từ đầu (không kế thừa trạng thái xe).
        """
        self.client.reset()
        self.bootstrap()
        self.last_obs = self.observe_all()

    def observe(self, intersection_id: int) -> dict[str, Any]:
        """Lấy observation chi tiết cho một nút giao cụ thể.

        Args:
            intersection_id: ID của nút giao muốn quan sát.

        Returns:
            Dict chứa: directions, light_states, timings, imbalance, v.v.
        """
        return self.client.get_state(intersection_id)

    def observe_all(self) -> dict[int, dict[str, Any]]:
        """Lấy observation từ tất cả nút giao trong mạng thông qua batch endpoint.

        Nếu intersection_ids chưa được khởi tạo, tự động gọi bootstrap().

        Returns:
            Dict mapping intersection_id → observation dict.
        """
        if not self.intersection_ids:
            self.bootstrap()
        states = self.client.get_states()
        # Chuyển đổi key từ chuỗi (do JSON) thành int
        return {int(k): v for k, v in states.items()}

    # ── Các hàm lấy features từ backend đã tính sẵn ──────────────────────────

    def _queue_total(self, observation: dict[str, Any]) -> float:
        """Lấy tổng queue (số xe chờ) đã tính sẵn từ Backend.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tổng queue_length của tất cả hướng vào (float).
        """
        return float(observation.get("queue_total", 0.0))

    def _density_total(self, observation: dict[str, Any]) -> float:
        """Lấy tổng mật độ phương tiện đã tính sẵn từ Backend.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tổng mật độ (xe/m) trung bình từ tất cả hướng.
        """
        return float(observation.get("density_total", 0.0))

    def _speed_average(self, observation: dict[str, Any]) -> float:
        """Lấy tốc độ trung bình đã tính sẵn từ Backend.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Tốc độ trung bình (m/s).
        """
        return float(observation.get("speed_avg", 0.0))

    def _imbalance(self, observation: dict[str, Any]) -> float:
        """Lấy độ mất cân bằng queue đã tính sẵn từ Backend.

        Args:
            observation: Dict observation của một nút giao.

        Returns:
            Độ mất cân bằng (float, >= 0).
        """
        return float(observation.get("imbalance", 0.0))

    def reward_for(
        self,
        intersection_id: int,
        obs_dict: dict[int, dict[str, Any]],
        action: int,
        reward_type: str = "pressure",
        last_obs_dict: dict[int, dict[str, Any]] | None = None,
    ) -> float:
        """Lấy phần thưởng cho một nút giao (hỗ trợ 3 chế độ).

        Args:
            intersection_id : ID nút giao cần tính reward.
            obs_dict        : Dict chứa observations mới của tất cả nút giao.
            action          : Hành động vừa thực hiện.
            reward_type     : Chế độ tính: "pressure", "dwt", hoặc "backend".
            last_obs_dict   : Dict chứa observations ở bước trước đó (dành cho dwt).

        Returns:
            Phần thưởng (float).
        """
        observation = obs_dict.get(intersection_id, {})
        if reward_type == "backend":
            return float(observation.get("reward", 0.0))

        if reward_type == "combined":
            # 1. Backend Reward (Thời gian chờ tích lũy)
            waiting_time_total = float(observation.get("reward_metrics", {}).get("waiting_time", 0.0))
            r_backend = - (waiting_time_total / 100.0)
            r_backend = max(-30.0, r_backend)

            # 2. DWT Reward (Vi phân thời gian chờ)
            if last_obs_dict is not None:
                w_t = float(last_obs_dict.get(intersection_id, {}).get("reward_metrics", {}).get("waiting_time", 0.0))
                w_t1 = waiting_time_total
                r_dwt = (w_t - w_t1) / SCALE_REWARD_DWT
            else:
                r_dwt = 0.0

            # 3. Pressure Reward (Áp suất dòng xe)
            incoming_nodes = observation.get("incoming_nodes", [])
            if not incoming_nodes:
                incoming_nodes = sorted([int(k) for k in observation.get("directions", {}).keys() if k.isdigit()])
            
            total_abs_pressure = 0.0
            directions = observation.get("directions", {})
            for inc in incoming_nodes:
                payload = directions.get(str(inc), {})
                density_in = float(payload.get("motorcycle_density", 0.0)) + float(payload.get("car_density", 0.0))
                
                outgoing_densities = []
                for other_inc in incoming_nodes:
                    if other_inc == inc:
                        continue
                    neighbor_state = obs_dict.get(other_inc)
                    if neighbor_state is not None:
                        neighbor_dirs = neighbor_state.get("directions", {})
                        payload_out = neighbor_dirs.get(str(intersection_id), {}) if isinstance(neighbor_dirs, dict) else {}
                        density_out = float(payload_out.get("motorcycle_density", 0.0)) + float(payload_out.get("car_density", 0.0))
                        outgoing_densities.append(density_out)
                
                avg_density_out = sum(outgoing_densities) / len(outgoing_densities) if outgoing_densities else 0.0
                pressure = density_in - avg_density_out
                total_abs_pressure += abs(pressure)
            
            r_pressure = -total_abs_pressure / SCALE_REWARD_PRESSURE

            # Tổ hợp tuyến tính
            return W_BACKEND * r_backend + W_DWT * r_dwt + W_PRESSURE * r_pressure

        if reward_type == "dwt":
            if last_obs_dict is None:
                return 0.0
            # dwt = W_t - W_{t+1}
            w_t = float(last_obs_dict.get(intersection_id, {}).get("reward_metrics", {}).get("waiting_time", 0.0))
            w_t1 = float(observation.get("reward_metrics", {}).get("waiting_time", 0.0))
            return (w_t - w_t1) / SCALE_REWARD_DWT

        # Mặc định là chế độ "pressure": R = - \sum |Pressure_p| / SCALE_REWARD_PRESSURE
        incoming_nodes = observation.get("incoming_nodes", [])
        if not isinstance(incoming_nodes, list):
            incoming_nodes = []
        directions = observation.get("directions", {})
        if not isinstance(directions, dict):
            directions = {}

        if not incoming_nodes:
            incoming_nodes = sorted([int(k) for k in directions.keys() if k.isdigit()])

        total_abs_pressure = 0.0
        for inc in incoming_nodes:
            payload = directions.get(str(inc), {})
            if not isinstance(payload, dict):
                continue

            density_in = float(payload.get("motorcycle_density", 0.0)) + float(payload.get("car_density", 0.0))

            outgoing_densities = []
            for other_inc in incoming_nodes:
                if other_inc == inc:
                    continue
                neighbor_state = obs_dict.get(other_inc)
                if neighbor_state is not None:
                    neighbor_dirs = neighbor_state.get("directions", {})
                    if isinstance(neighbor_dirs, dict):
                        payload_out = neighbor_dirs.get(str(intersection_id), {})
                        if isinstance(payload_out, dict):
                            density_out = float(payload_out.get("motorcycle_density", 0.0)) + float(payload_out.get("car_density", 0.0))
                            outgoing_densities.append(density_out)

            if outgoing_densities:
                avg_density_out = sum(outgoing_densities) / len(outgoing_densities)
            else:
                avg_density_out = 0.0

            pressure = density_in - avg_density_out
            total_abs_pressure += abs(pressure)

        return -total_abs_pressure / SCALE_REWARD_PRESSURE

    def apply_actions(self, actions: dict[int, int]) -> None:
        """Gửi hành động lên backend và cập nhật IntersectionRecord.

        Args:
            actions: Dict mapping intersection_id → action (pha mong muốn: 0, 1, 2 hoặc 3).
        """
        filtered_actions = {intersection_id: action for intersection_id, action in actions.items() if action in (0, 1, 2, 3)}
        if filtered_actions:
            self.client.post_actions(filtered_actions)

        for intersection_id, action in filtered_actions.items():
            record = self.records.setdefault(intersection_id, IntersectionRecord())
            obs = self.last_obs.get(intersection_id, {})
            current_phase = obs.get("current_phase", 0)
            
            if action != current_phase:
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
        self.last_obs = self.observe_all()
        return self.last_obs
