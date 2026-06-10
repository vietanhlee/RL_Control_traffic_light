import asyncio
import math
from dataclasses import asdict
from models.traffic_light.fixed_time import LightColor
import api.v1.state as config
from api.v1.state import engine_lock, active_connections, logger
from core.config import SIMULATION_DT_SECONDS
from core.constants import GLOBAL_IMBALANCE_WEIGHT
from core.reward_fn import compute_intersection_reward

def calculate_intersection_reward(engine_inst, intersection_id: int, time_s: float, light_states_dict: dict = None) -> dict:
    """
    Thu thập metrics thô từ engine, sau đó gọi compute_intersection_reward()
    để tính reward. Kết quả được serialise thành dict để FE render.

    Được dùng chung cho cả simulation loop và API Endpoint.
    """
    intersection = engine_inst.intersections[intersection_id]
    total_queue = 0
    red_pressure = 0.0
    queues = []
    avg_speed = 0.0
    direction_count = 0

    for incoming in intersection.incoming_nodes:
        metrics = engine_inst.collector.snapshot_direction((intersection_id, incoming), time_s)
        q_val = metrics.motorcycle.queue_length + metrics.car.queue_length
        total_queue += q_val
        queues.append(q_val)

        avg_speed += (metrics.motorcycle.avg_speed + metrics.car.avg_speed) / 2.0
        direction_count += 1

        # Xác định đèn đỏ hay không
        if light_states_dict is not None:
            is_red = light_states_dict.get((intersection_id, incoming)) == LightColor.RED
        else:
            is_red = intersection.light_for(time_s, incoming) == LightColor.RED

        if is_red and q_val > 0.0:
            red_pressure += q_val

    if direction_count:
        avg_speed /= direction_count

    avg_q = sum(queues) / len(queues) if queues else 0.0
    imbalance = sum(abs(q - avg_q) for q in queues)

    switched = (time_s - engine_inst.last_switch_time.get(intersection_id, -9999.0)) < 1.1

    # ── Gọi hàm tính reward từ module dùng chung ────────────────────────
    rc = compute_intersection_reward(
        queue_total=total_queue,
        imbalance=imbalance,
        red_pressure=red_pressure,
        speed_avg=avg_speed,
        switched=switched,
    )

    # Trả về đầy đủ các trường để Frontend chỉ việc render
    return {
        "queue_length":        total_queue,
        "queue_penalty":       rc.queue_penalty,
        "queue_pct":           rc.queue_pct,

        "imbalance":           imbalance,
        "imbalance_penalty":   rc.imbalance_penalty,
        "imbalance_pct":       rc.imbalance_pct,

        "red_pressure":        red_pressure,
        "red_pressure_penalty": rc.red_pressure_penalty,
        "red_pressure_pct":    rc.red_pressure_pct,

        "switch_penalty":      rc.switch_penalty,

        "speed_avg":           avg_speed,
        "speed_bonus":         rc.speed_bonus,
        "speed_pct":           rc.speed_pct,

        "cost":                rc.cost,
        "reward":              rc.reward,
    }

def _run_simulation_steps(engine_inst, dt: float, steps_count: int, get_rl_state: bool):
    snapshot = None
    for _ in range(steps_count):
        snapshot = engine_inst.step(dt)
    
    rl_state_dict = None
    if get_rl_state and snapshot is not None:
        rl_state_dict = asdict(engine_inst.rl_state())
        
    return snapshot, rl_state_dict

async def simulation_loop():
    logger.info("Starting simulation background task...")
    while True:
        try:
            render_data = None
            snapshot = None
            rl_state_dict = None
            
            # Truy cập engine động qua config.engine để tránh stale reference khi reset
            current_engine = config.engine
            
            async with engine_lock:
                get_rl_state = bool(active_connections)
                snapshot, rl_state_dict = await asyncio.to_thread(
                    _run_simulation_steps,
                    current_engine,
                    SIMULATION_DT_SECONDS,
                    3,
                    get_rl_state
                )
            
            # Nhường Event Loop cho các API requests
            await asyncio.sleep(0)

            if active_connections and snapshot is not None and rl_state_dict is not None:
                intersection_rewards = {}

                # Trích xuất light states
                light_states_dict = {
                    (inter, inc): color 
                    for (inter, inc), color in snapshot.light_states.items()
                }
                
                for intersection_id in current_engine.intersections.keys():
                    res = calculate_intersection_reward(
                        current_engine, 
                        intersection_id, 
                        snapshot.now, 
                        light_states_dict
                    )
                    intersection_rewards[intersection_id] = res

                # ── Tính Global Reward cân bằng mạng ───────────────────────
                # Công thức: global = mean(rewards) - α × std(rewards)
                # std lớn → các nút chênh lệch nhau nhiều → bị phạt thêm
                reward_values = [r["reward"] for r in intersection_rewards.values()]
                n = len(reward_values)
                if n > 0:
                    mean_reward = sum(reward_values) / n
                    variance = sum((r - mean_reward) ** 2 for r in reward_values) / n
                    std_reward = math.sqrt(variance)
                else:
                    mean_reward = 0.0
                    std_reward = 0.0

                imbalance_deduction = GLOBAL_IMBALANCE_WEIGHT * std_reward
                avg_global_reward = mean_reward - imbalance_deduction
                
                # Tính toán tổng hàng chờ toàn mạng (global queue)
                global_queue = sum(r["queue_length"] for r in intersection_rewards.values())

                render_data = {
                    "time_s": snapshot.now,
                    "vehicles": [
                        {"id": v.vehicle_id, "x": v.x, "y": v.y, "speed": v.speed_mps, "type": v.type, "angle": v.angle}
                        for v in snapshot.vehicles
                    ],
                    "lights": [
                        {"intersection": inter, "incoming": inc, "color": color.name}
                        for (inter, inc), color in snapshot.light_states.items()
                    ],
                    "metrics": rl_state_dict,
                    "global_reward": avg_global_reward,
                    "global_reward_mean": mean_reward,
                    "global_reward_std": std_reward,
                    "global_imbalance_deduction": imbalance_deduction,
                    "global_queue": global_queue,
                    "reward_metrics": intersection_rewards,
                }

            # Broadcast tới websockets bên ngoài lock
            if render_data is not None:
                disconnected = set()
                for connection in list(active_connections):
                    try:
                        await connection.send_json(render_data)
                    except Exception as e:
                        logger.error(f"Error sending data to websocket client: {e}")
                        disconnected.add(connection)
                for d in disconnected:
                    active_connections.discard(d)
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}", exc_info=True)

        await asyncio.sleep(0.3)
