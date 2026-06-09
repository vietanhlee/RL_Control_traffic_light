from __future__ import annotations

import concurrent.futures
import random
from dataclasses import dataclass, field
from math import acos
from typing import Dict, List

from config.constants import (
    BOUNDARY_DISTANCE_METERS,
    DB_FLUSH_SECONDS,
    DENSITY_SAMPLE_SECONDS,
    DATABASE_URL,
    METRICS_WINDOW_SECONDS,
    QUEUE_SPEED_THRESHOLD,
    SIMULATION_DT_SECONDS,
    STOP_LINE_DISTANCE_METERS,
    ZONE_LENGTH_METERS,
    SimulationConfig,
)
from database.repository import create_repository
from metrics.calculator import MetricsCalculator
from metrics.collector import DirectionMetrics, MetricsCollector
from metrics.measurement_zone import ApproachKey, MeasurementZone
from simulation.intersection import Intersection
from simulation.road import RoadNetwork
from simulation.vehicle import Vehicle, VehicleType
from traffic_light.controller import DirectionState, IntersectionState, SimulationState
from traffic_light.fixed_time import LightColor


@dataclass
class VehicleRenderState:
    vehicle_id: int
    x: float
    y: float
    speed_mps: float
    type: str
    angle: float


@dataclass
class SimulationSnapshot:
    now: float
    vehicles: list[VehicleRenderState]
    light_states: dict[ApproachKey, LightColor]
    direction_metrics: dict[ApproachKey, DirectionMetrics]
    local_imbalance: dict[int, float]
    global_imbalance: float


@dataclass
class SimulationEngine:
    config: SimulationConfig
    database_url: str = DATABASE_URL
    network: RoadNetwork = field(default_factory=RoadNetwork.from_defaults)

    def __post_init__(self) -> None:
        self.db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.repository = create_repository(self.database_url)
        self.collector = MetricsCollector(window_seconds=METRICS_WINDOW_SECONDS)
        self.calculator = MetricsCalculator()
        self.time_s = 0.0
        self.last_spawn_at = 0.0
        self.last_density_sample = 0.0
        self.last_db_flush = 0.0
        self.next_vehicle_id = 1
        self.available_vehicle_ids: set[int] = set()
        self.vehicles: list[Vehicle] = []
        self.intersections: Dict[int, Intersection] = {}
        self.measurement_zones: Dict[ApproachKey, MeasurementZone] = {}
        self.local_imbalance: Dict[int, float] = {}
        self.global_imbalance: float = 0.0
        self.random = random.Random(42)
        self.last_switch_time: Dict[int, float] = {}
        self.boundary_nodes = [n for n in self.network.all_nodes() if self.network.is_boundary_node(n)]
        self._build_intersections()
        self._build_zones()
        self._load_light_configs()

    def _replace_config(self, **kwargs) -> None:
        self.config = SimulationConfig(
            target_vehicle_count=kwargs.get("target_vehicle_count", self.config.target_vehicle_count),
            min_speed_mps=kwargs.get("min_speed_mps", self.config.min_speed_mps),
            max_speed_mps=kwargs.get("max_speed_mps", self.config.max_speed_mps),
            spawn_interval_seconds=kwargs.get("spawn_interval_seconds", self.config.spawn_interval_seconds),
            max_acceleration=kwargs.get("max_acceleration", self.config.max_acceleration),
            max_deceleration=kwargs.get("max_deceleration", self.config.max_deceleration),
            safe_gap_meters=kwargs.get("safe_gap_meters", self.config.safe_gap_meters),
            turn_distribution=kwargs.get("turn_distribution", self.config.turn_distribution),
        )

    def _build_intersections(self) -> None:
        for node in self.network.all_nodes():
            incoming = self.network.incoming_neighbors(node)
            self.intersections[node] = Intersection.with_defaults(node, incoming)

    def _build_zones(self) -> None:
        for node, intersection in self.intersections.items():
            for incoming_from in intersection.incoming_nodes:
                key = (node, incoming_from)
                self.measurement_zones[key] = MeasurementZone(
                    intersection_id=node,
                    incoming_from=incoming_from,
                    zone_length_m=ZONE_LENGTH_METERS,
                    boundary_distance_m=BOUNDARY_DISTANCE_METERS,
                )

    def _load_light_configs(self) -> None:
        stored = self.repository.load_light_config()
        for (intersection_id, incoming_from), (green, yellow, red) in stored.items():
            if intersection_id in self.intersections and incoming_from in self.intersections[intersection_id].incoming_nodes:
                self.intersections[intersection_id].update_timing(incoming_from, green, yellow, red)

    def apply_light_timing(self, intersection_id: int, incoming_from: int, green: float, yellow: float, red: float) -> None:
        intersection = self.intersections[intersection_id]
        intersection.update_timing(incoming_from, green, yellow, red)
        self.repository.save_light_config(intersection_id, incoming_from, green, yellow, red)

    def force_switch_phase(self, intersection_id: int) -> None:
        if intersection_id in self.intersections:
            self.intersections[intersection_id].force_switch_phase()
            self.last_switch_time[intersection_id] = self.time_s

    def set_vehicle_target(self, target: int) -> None:
        self._replace_config(target_vehicle_count=max(1, target))

    def set_speed_range(self, min_speed: float, max_speed: float) -> None:
        min_v = max(0.1, min(min_speed, max_speed))
        max_v = max(min_v + 0.1, max(min_speed, max_speed))
        self._replace_config(min_speed_mps=min_v, max_speed_mps=max_v)
        for vehicle in self.vehicles:
            vehicle.min_speed_mps = min_v
            vehicle.max_speed_mps = max_v
            vehicle.desired_speed_mps = min(vehicle.desired_speed_mps, max_v)

    def set_turn_distribution(self, left: float, straight: float, right: float) -> None:
        distribution = {
            "left": max(0.0, left),
            "straight": max(0.0, straight),
            "right": max(0.0, right),
        }
        total = sum(distribution.values())
        if total <= 1e-6:
            distribution = {"left": 0.2, "straight": 0.6, "right": 0.2}
        self._replace_config(turn_distribution=distribution)

    def step(self, dt: float = SIMULATION_DT_SECONDS) -> SimulationSnapshot:
        self.time_s += dt
        self._spawn_vehicles_if_needed()
        self._update_vehicles(dt)

        if self.time_s - self.last_density_sample >= DENSITY_SAMPLE_SECONDS:
            self.last_density_sample = self.time_s
            self._sample_density()

        self._update_queues_and_imbalance()

        if self.time_s - self.last_db_flush >= DB_FLUSH_SECONDS:
            self.last_db_flush = self.time_s
            self._flush_metrics_to_db()

        return self.snapshot()

    def snapshot(self) -> SimulationSnapshot:
        direction_metrics: Dict[ApproachKey, DirectionMetrics] = {}
        light_states: Dict[ApproachKey, LightColor] = {}
        for key in self.measurement_zones:
            direction_metrics[key] = self.collector.snapshot_direction(key, self.time_s)
            inter_id, incoming = key
            light_states[key] = self.intersections[inter_id].light_for(self.time_s, incoming)

        return SimulationSnapshot(
            now=self.time_s,
            vehicles=[self._vehicle_render_state(v) for v in self.vehicles],
            light_states=light_states,
            direction_metrics=direction_metrics,
            local_imbalance=dict(self.local_imbalance),
            global_imbalance=self.global_imbalance,
        )

    def rl_state(self) -> SimulationState:
        intersections: Dict[int, IntersectionState] = {}
        for intersection_id, intersection in self.intersections.items():
            directions: Dict[int, DirectionState] = {}
            active_source = intersection.incoming_nodes[0] if intersection.incoming_nodes else -1
            for incoming in intersection.incoming_nodes:
                metrics = self.collector.snapshot_direction((intersection_id, incoming), self.time_s)
                directions[incoming] = DirectionState(
                    queue_length=metrics.motorcycle.queue_length + metrics.car.queue_length,
                    avg_speed=(metrics.motorcycle.avg_speed + metrics.car.avg_speed) / 2.0,
                    avg_density=metrics.motorcycle.avg_density + metrics.car.avg_density,
                )
                if intersection.light_for(self.time_s, incoming) == LightColor.GREEN:
                    active_source = incoming
            intersections[intersection_id] = IntersectionState(
                current_phase_source=active_source,
                time=self.time_s,
                directions=directions,
            )
        return SimulationState(global_imbalance=self.global_imbalance, intersections=intersections)

    def _spawn_vehicles_if_needed(self) -> None:
        import math
        wave = (math.sin(self.time_s / 50.0) + 1.0) / 2.0
        dynamic_target = int(250 + wave * 1450)
        dynamic_interval = 0.015 + (1.0 - wave) * 1.0
        
        # Giới hạn số xe spawn tối đa mỗi step để tránh spike lag đột ngột khi reset
        max_spawns = 20
        spawns = 0

        while (len(self.vehicles) < dynamic_target and 
               self.time_s - self.last_spawn_at >= dynamic_interval and 
               spawns < max_spawns):
            if self.last_spawn_at == 0.0:
                self.last_spawn_at = self.time_s
            else:
                self.last_spawn_at += dynamic_interval
            self._spawn_one_vehicle()
            spawns += 1

    def _spawn_one_vehicle(self) -> None:
        origin = self.random.choice(self.boundary_nodes) if self.boundary_nodes else self.random.choice(list(self.network.all_nodes()))
        path = self._generate_route(origin)
        if len(path) < 2:
            return

        # 60% motorcycle, 28% car, 12% bus
        rand_val = self.random.random()
        if rand_val < 0.6:
            v_type = VehicleType.MOTORCYCLE
            desired_speed = self.random.uniform(8.0, 25.0)
            length_m = 2.0
        elif rand_val < 0.88:
            v_type = VehicleType.CAR
            desired_speed = self.random.uniform(5.0, 35.0)
            length_m = 4.5
        else:
            v_type = VehicleType.BUS
            desired_speed = self.random.uniform(4.0, 30.0)
            length_m = 10.0

        if self.available_vehicle_ids:
            assigned_id = self.available_vehicle_ids.pop()
        else:
            assigned_id = self.next_vehicle_id
            self.next_vehicle_id += 1

        # Tìm làn có ít xe nhất trên Edge đầu tiên để spawn đều làn
        first_edge_key = (path[0], path[1])
        first_edge = self.network.edge(*first_edge_key)
        lane_counts = {l: 0 for l in range(first_edge.lanes)}
        for v in self.vehicles:
            if v.current_from == path[0] and v.current_to == path[1]:
                if v.lane_index in lane_counts:
                    lane_counts[v.lane_index] += 1
        chosen_lane = min(lane_counts, key=lane_counts.get)

        vehicle = Vehicle(
            vehicle_id=assigned_id,
            path=path,
            path_index=0,
            progress_m=0.0,
            speed_mps=desired_speed * 0.6,
            desired_speed_mps=desired_speed,
            min_speed_mps=self.config.min_speed_mps,
            max_speed_mps=self.config.max_speed_mps,
            max_acceleration=self.config.max_acceleration,
            max_deceleration=self.config.max_deceleration,
            length_m=length_m,
            vehicle_type=v_type,
            lane_index=chosen_lane
        )
        self.vehicles.append(vehicle)

    def _generate_route(self, origin: int, max_hops: int = 25) -> list[int]:
        neighbors = self.network.neighbors(origin)
        if not neighbors:
            return [origin]

        path = [origin, self.random.choice(neighbors)]
        dist = self.config.normalized_turn_distribution()

        while len(path) < max_hops + 1:
            incoming_from = path[-2]
            via_node = path[-1]
            candidates = [n for n in self.network.neighbors(via_node) if n != incoming_from]
            if not candidates:
                break

            weighted = []
            for nxt in candidates:
                turn = self.choose_turn_type(incoming_from, via_node, nxt)
                weight = dist.get(turn, 0.1)
                
                # Ưu tiên xe đi vào lõi, hạn chế đi quanh rìa
                if self.network.is_boundary_node(nxt):
                    weight *= 0.3
                else:
                    weight *= 2.5
                
                weighted.append((nxt, weight))

            total = sum(weight for _, weight in weighted)
            if total <= 1e-6:
                path.append(self.random.choice(candidates))
                continue

            pick = self.random.uniform(0.0, total)
            current = 0.0
            chosen = candidates[0]
            for node, weight in weighted:
                current += weight
                if pick <= current:
                    chosen = node
                    break
            path.append(chosen)

            # Randomly terminate route to preserve diverse trip lengths.
            if len(path) >= 4 and self.network.is_boundary_node(chosen):
                if self.random.random() < 0.60:
                    break

        return path

    def _update_vehicles(self, dt: float) -> None:
        vehicles_by_edge: Dict[tuple[int, int], List[Vehicle]] = {}
        for v in self.vehicles:
            if not self.network.has_edge(v.current_from, v.current_to):
                continue
            vehicles_by_edge.setdefault((v.current_from, v.current_to), []).append(v)

        # Phase 1: Lane changing / Overtaking logic
        for edge_key, edge_vehicles in vehicles_by_edge.items():
            edge = self.network.edge(*edge_key)
            if edge.lanes <= 1:
                continue
            
            # Sắp xếp xe từ trước ra sau
            edge_vehicles.sort(key=lambda v: v.progress_m, reverse=True)
            
            lanes_vehicles = {l: [] for l in range(edge.lanes)}
            for v in edge_vehicles:
                lanes_vehicles[v.lane_index].append(v)
                
            front_same_lane_dict = {}
            for l, v_list in lanes_vehicles.items():
                prev_v = None
                for curr_v in v_list:
                    front_same_lane_dict[id(curr_v)] = prev_v
                    prev_v = curr_v
            
            for v in edge_vehicles:
                # Tìm xe phía trước cùng làn
                front_same_lane = front_same_lane_dict[id(v)]
                
                # Nếu có xe phía trước cùng làn và khoảng cách nhỏ hơn khoảng cách an toàn
                if front_same_lane is not None:
                    gap = front_same_lane.progress_m - v.progress_m - front_same_lane.length_m
                    safe_gap = self.config.safe_gap_meters + v.speed_mps * 0.5
                    
                    # Muốn vượt nếu bị cản trở bởi xe trước đi chậm hơn 95% tốc độ mong muốn của mình
                    if gap < safe_gap and front_same_lane.speed_mps < v.desired_speed_mps * 0.95:
                        possible_lanes = []
                        if v.lane_index > 0:
                            possible_lanes.append(v.lane_index - 1)
                        if v.lane_index < edge.lanes - 1:
                            possible_lanes.append(v.lane_index + 1)
                            
                        # Thử xem làn nào an toàn để vượt
                        for target_lane in possible_lanes:
                            target_list = lanes_vehicles[target_lane]
                            front_target = None
                            back_target = None
                            for other in target_list:
                                if other.progress_m > v.progress_m:
                                    front_target = other
                                else:
                                    back_target = other
                                    break
                            
                            # Kiểm tra an toàn trước
                            safe_front = True
                            if front_target is not None:
                                gap_front = front_target.progress_m - v.progress_m - front_target.length_m
                                if gap_front < (v.speed_mps * 0.5 + 4.0):
                                    safe_front = False
                                    
                            # Kiểm tra an toàn sau
                            safe_back = True
                            if back_target is not None:
                                gap_back = v.progress_m - back_target.progress_m - v.length_m
                                if gap_back < (back_target.speed_mps * 0.5 + 4.0):
                                    safe_back = False
                                    
                            # Nếu cả trước và sau đều an toàn, chuyển làn ngay lập tức
                            if safe_front and safe_back:
                                old_lane = v.lane_index
                                v.lane_index = target_lane
                                lanes_vehicles[old_lane].remove(v)
                                insert_idx = 0
                                for i, tv in enumerate(target_list):
                                    if tv.progress_m < v.progress_m:
                                        break
                                    insert_idx = i + 1
                                target_list.insert(insert_idx, v)
                                
                                front_same_lane_dict.clear()
                                for l_idx, v_list_new in lanes_vehicles.items():
                                    prev_v = None
                                    for curr_v in v_list_new:
                                        front_same_lane_dict[id(curr_v)] = prev_v
                                        prev_v = curr_v
                                break

        # Phase 2: Speed and position updates (bám đuôi cùng làn)
        for edge_key, edge_vehicles in vehicles_by_edge.items():
            edge = self.network.edge(*edge_key)
            edge_vehicles.sort(key=lambda v: v.progress_m, reverse=True)

            lanes_vehicles = {l: [] for l in range(edge.lanes)}
            for v in edge_vehicles:
                lanes_vehicles[v.lane_index].append(v)
                
            front_vehicle_dict = {}
            for l, v_list in lanes_vehicles.items():
                prev_v = None
                for curr_v in v_list:
                    front_vehicle_dict[id(curr_v)] = prev_v
                    prev_v = curr_v

            for v in edge_vehicles:
                previous_distance = edge.length_m - v.progress_m
                
                # Tính ngẫu nhiên: Xe có thể chạy nhanh hoặc chậm hơn bình thường 15%
                stochastic_factor = self.random.uniform(0.85, 1.15)
                target_speed = v.desired_speed_mps * stochastic_factor
                
                v.is_waiting = False

                next_intersection = self.intersections[v.current_to]
                light_state = next_intersection.light_for(self.time_s, v.current_from)

                # Tính rành giớì giao lộ (khớp với frontend: R = lanes * 10 * 1.15)
                # Đường hiện tại có edge.lanes làn → ước tính bán kính giao lộ
                junction_boundary = edge.lanes * 10.0 * 1.15  # ≈ world_units (= mét)
                # Vạch dừng: tại ranh giới giao lộ (khớp với vạch vẽ trên frontend)
                stop_line_dist = junction_boundary + STOP_LINE_DISTANCE_METERS

                # Khoảng cách cần để phanh từ tốc độ hiện tại về 0
                braking_dist = (v.speed_mps ** 2) / (2.0 * max(self.config.max_deceleration, 0.1))
                # Trigger phanh sớm: dựa vào tốc độ thực tế
                trigger_dist = stop_line_dist + braking_dist + 8.0

                # Khoảng cách để quyết định xe đã qua hẳn vạch dừng chưa
                # Trừ đi 2.0m để cho phép sai số dừng lố vạch một chút
                passed_stop_line_threshold = stop_line_dist - 2.0

                if light_state == LightColor.RED and previous_distance <= trigger_dist:
                    if previous_distance < passed_stop_line_threshold:
                        # Đã qua hẳn vạch dừng -> cho đi tiếp để thoát nút giao
                        pass
                    elif previous_distance <= stop_line_dist:
                        # Đã đến vạch dừng → dừng hẳn
                        target_speed = 0.0
                        v.is_waiting = True
                    else:
                        # Đang trong vùng phanh: giảm tốc dần theo khoảng cách
                        remaining_to_stop = previous_distance - stop_line_dist
                        slow_speed = max(0.0, (remaining_to_stop / max(braking_dist, 1.0)) * v.speed_mps)
                        target_speed = min(target_speed, slow_speed)
                        if slow_speed <= 0.5:
                            v.is_waiting = True
                elif light_state == LightColor.YELLOW and previous_distance <= stop_line_dist + 5.0:
                    if previous_distance < passed_stop_line_threshold:
                        # Đã qua hẳn vạch dừng -> đi tiếp
                        pass
                    else:
                        target_speed = min(target_speed, 2.0)

                # Tìm xe phía trước cùng làn để phanh/bám đuôi
                front_vehicle = front_vehicle_dict[id(v)]

                if front_vehicle is not None:
                    gap = front_vehicle.progress_m - v.progress_m - front_vehicle.length_m
                    safe_gap = self.config.safe_gap_meters + v.speed_mps * 0.5
                    if gap < safe_gap:
                        target_speed = min(target_speed, max(0.0, front_vehicle.speed_mps - 2.0))
                        if target_speed <= 0.2:
                            v.is_waiting = True

                target_speed = max(0.0, min(target_speed, v.max_speed_mps))
                v.step_speed(dt, target_speed)
                v.advance(dt)

                current_distance = max(0.0, edge.length_m - v.progress_m)
                approach_key = (v.current_to, v.current_from)
                zone = self.measurement_zones[approach_key]
                if zone.crossed_boundary(previous_distance, current_distance):
                    group = "motorcycle" if v.vehicle_type == VehicleType.MOTORCYCLE else "car"
                    self.collector.record_crossing(approach_key, group, self.time_s, v.speed_mps)

                if v.progress_m >= edge.length_m:
                    overflow = v.progress_m - edge.length_m
                    moved = self._advance_vehicle_to_next_edge(v, overflow)
                    if not moved:
                        continue

        retained_vehicles = []
        for v in self.vehicles:
            if v.remaining_nodes >= 1:
                retained_vehicles.append(v)
            else:
                self.available_vehicle_ids.add(v.vehicle_id)
        self.vehicles = retained_vehicles

    def _advance_vehicle_to_next_edge(self, vehicle: Vehicle, overflow_m: float) -> bool:
        vehicle.path_index += 1
        current_node = vehicle.path[vehicle.path_index]

        if vehicle.path_index >= len(vehicle.path) - 1:
            if self.network.is_boundary_node(current_node):
                return False
            
            boundary_nodes = [n for n in self.network.all_nodes() if n != current_node and self.network.is_boundary_node(n)]
            if boundary_nodes:
                destination = self.random.choice(boundary_nodes)
                extension = self.network.shortest_path(current_node, destination)
                if len(extension) > 1:
                    vehicle.path = vehicle.path[: vehicle.path_index] + extension
                else:
                    return False
            else:
                return False

        # Re-route occasionally to keep the stream dynamic and stochastic.
        elif self.random.random() < 0.35 or vehicle.remaining_nodes < 2:
            boundary_nodes = [n for n in self.network.all_nodes() if n != current_node and self.network.is_boundary_node(n)]
            if boundary_nodes:
                destination = self.random.choice(boundary_nodes)
            else:
                destination = self.random.choice([n for n in self.network.all_nodes() if n != current_node])
            extension = self.network.shortest_path(current_node, destination)
            if len(extension) > 1:
                # Keep the completed prefix intact and re-seed route from the current node.
                vehicle.path = vehicle.path[: vehicle.path_index] + extension

        if not self.network.has_edge(vehicle.current_from, vehicle.current_to):
            return False

        # Đảm bảo lane_index của xe hợp lệ với số làn của Edge mới
        new_edge = self.network.edge(vehicle.current_from, vehicle.current_to)
        vehicle.lane_index = min(vehicle.lane_index, new_edge.lanes - 1)

        vehicle.progress_m = max(0.0, overflow_m)
        return True

    def _sample_density(self) -> None:
        density_counter: Dict[ApproachKey, Dict[str, int]] = {k: {"motorcycle": 0, "car": 0} for k in self.measurement_zones.keys()}
        for vehicle in self.vehicles:
            if not self.network.has_edge(vehicle.current_from, vehicle.current_to):
                continue
            edge = self.network.edge(vehicle.current_from, vehicle.current_to)
            distance_to_target = edge.length_m - vehicle.progress_m
            key = (vehicle.current_to, vehicle.current_from)
            zone = self.measurement_zones[key]
            if zone.contains_distance(distance_to_target):
                group = "motorcycle" if vehicle.vehicle_type == VehicleType.MOTORCYCLE else "car"
                density_counter[key][group] += 1

        for key, groups in density_counter.items():
            for group, count in groups.items():
                self.collector.record_density(key, group, self.time_s, count)

    def _update_queues_and_imbalance(self) -> None:
        per_intersection_metrics: Dict[int, Dict[int, DirectionMetrics]] = {}

        queue_counter: Dict[ApproachKey, Dict[str, int]] = {k: {"motorcycle": 0, "car": 0} for k in self.measurement_zones.keys()}
        for vehicle in self.vehicles:
            if not self.network.has_edge(vehicle.current_from, vehicle.current_to):
                continue
            key = (vehicle.current_to, vehicle.current_from)
            edge = self.network.edge(vehicle.current_from, vehicle.current_to)
            distance_to_target = edge.length_m - vehicle.progress_m
            light_state = self.intersections[vehicle.current_to].light_for(self.time_s, vehicle.current_from)
            if (
                light_state == LightColor.RED
                and distance_to_target <= ZONE_LENGTH_METERS
                and vehicle.speed_mps <= QUEUE_SPEED_THRESHOLD
            ):
                group = "motorcycle" if vehicle.vehicle_type == VehicleType.MOTORCYCLE else "car"
                queue_counter[key][group] += 1

        for key, groups in queue_counter.items():
            for group, count in groups.items():
                self.collector.set_queue_length(key, group, count)

        self.local_imbalance = {}
        for intersection_id, intersection in self.intersections.items():
            direction_metrics = {
                incoming: self.collector.snapshot_direction((intersection_id, incoming), self.time_s)
                for incoming in intersection.incoming_nodes
            }
            per_intersection_metrics[intersection_id] = direction_metrics
            local = self.calculator.local_imbalance(direction_metrics)
            self.local_imbalance[intersection_id] = local.local_imbalance

        self.global_imbalance = self.calculator.global_imbalance(self.local_imbalance)

    def _flush_metrics_to_db(self) -> None:
        rows = []
        for intersection_id, intersection in self.intersections.items():
            local = self.local_imbalance.get(intersection_id, 0.0)
            for incoming in intersection.incoming_nodes:
                key = (intersection_id, incoming)
                dm = self.collector.snapshot_direction(key, self.time_s)
                rows.append(
                    (
                        self.time_s,
                        intersection_id,
                        incoming,
                        dm.motorcycle.avg_speed,
                        dm.motorcycle.avg_density,
                        dm.motorcycle.queue_length,
                        dm.car.avg_speed,
                        dm.car.avg_density,
                        dm.car.queue_length,
                        local,
                        self.global_imbalance,
                    )
                )
        self.db_executor.submit(self.repository.save_metrics_batch, rows)

    def _vehicle_render_state(self, vehicle: Vehicle) -> VehicleRenderState:
        if not self.network.has_edge(vehicle.current_from, vehicle.current_to):
            x, y = self.network.positions[vehicle.current_from]
            return VehicleRenderState(vehicle_id=vehicle.vehicle_id, x=x, y=y, speed_mps=0.0, type=vehicle.vehicle_type.value, angle=0.0)

        start = self.network.positions[vehicle.current_from]
        end = self.network.positions[vehicle.current_to]
        edge = self.network.edge(vehicle.current_from, vehicle.current_to)
        ratio = min(max(vehicle.progress_m / max(edge.length_m, 1e-6), 0.0), 1.0)
        
        x_mid = start[0] + (end[0] - start[0]) * ratio
        y_mid = start[1] + (end[1] - start[1]) * ratio
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        L = (dx**2 + dy**2)**0.5
        
        angle = 0.0
        if L > 1e-6:
            import math
            angle = math.atan2(dy, dx)
            
            # Vector pháp tuyến quay sang phải 90 độ để đi bên phải đường
            nx = dy / L
            ny = -dx / L
            lane_width = 10.0  # Tăng lên 10.0 để khớp hoàn toàn với tỷ lệ vẽ ở Frontend
            lane_index = getattr(vehicle, "lane_index", 0)
            # Với đường > 2 làn có dải phân cách cứng (rộng 0.9 * laneWidth),
            # cần lùi thêm barrier_offset để xe không chạy vào vùng phân cách.
            barrier_extra = (0.45 * lane_width) if edge.lanes > 2 else 0.0
            # Dịch chuyển lệch bên phải tim đường dựa vào làn đường
            offset = (edge.lanes - 1 - lane_index + 0.5) * lane_width + barrier_extra
            x = x_mid + nx * offset
            y = y_mid + ny * offset
        else:
            x = x_mid
            y = y_mid
            
        return VehicleRenderState(
            vehicle_id=vehicle.vehicle_id, 
            x=x, 
            y=y, 
            speed_mps=vehicle.speed_mps, 
            type=vehicle.vehicle_type.value, 
            angle=angle
        )

    def choose_turn_type(self, incoming_from: int, via_node: int, outgoing_to: int) -> str:
        """Utility for future policy features and RL state enrichment."""
        in_x, in_y = self.network.positions[incoming_from]
        via_x, via_y = self.network.positions[via_node]
        out_x, out_y = self.network.positions[outgoing_to]

        v1 = (via_x - in_x, via_y - in_y)
        v2 = (out_x - via_x, out_y - via_y)

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
