import asyncio
import logging
from dataclasses import asdict
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simulation.engine import SimulationEngine
from config.constants import (
    SimulationConfig,
    SIMULATION_DT_SECONDS,
    REWARD_OFFSET,
    WEIGHT_QUEUE,
    WEIGHT_IMBALANCE,
    WEIGHT_RED_PRESSURE,
    WEIGHT_SWITCH_PENALTY,
    WEIGHT_SPEED_BONUS,
    SCALE_QUEUE,
    SCALE_IMBALANCE,
    SCALE_RED_PRESSURE,
    SCALE_SPEED,
    REWARD_CLIP,
)
from traffic_light.fixed_time import LightColor

# Cấu hình logging ghi ra cả console và file simulation.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("simulation.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffic Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SimulationEngine(config=SimulationConfig())
active_connections: Set[WebSocket] = set()
engine_lock = asyncio.Lock()

class ActionRequest(BaseModel):
    action: int  # 0 for Keep, 1 for Change


class ActionsRequest(BaseModel):
    actions: Dict[int, int]

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    logger.info("Starting simulation background task...")
    while True:
        try:
            render_data = None
            async with engine_lock:
                snapshot = None
                for _ in range(3):
                    snapshot = engine.step(SIMULATION_DT_SECONDS)
                if active_connections and snapshot is not None:
                    rl_state = engine.rl_state()
                    
                    # 1. Tính toán reward_metrics cho tất cả các nút giao
                    intersection_rewards = {}
                    global_reward = 0.0
                    for intersection_id, intersection in engine.intersections.items():
                        total_queue = 0
                        red_pressure = 0.0
                        queues = []
                        avg_speed = 0.0
                        direction_count = 0
                        for incoming in intersection.incoming_nodes:
                            metrics = engine.collector.snapshot_direction((intersection_id, incoming), engine.time_s)
                            q_val = metrics.motorcycle.queue_length + metrics.car.queue_length
                            total_queue += q_val
                            queues.append(q_val)
                            avg_speed += (metrics.motorcycle.avg_speed + metrics.car.avg_speed) / 2.0
                            direction_count += 1
                            if intersection.light_for(engine.time_s, incoming) == LightColor.RED:
                                if q_val > 0.0:
                                    red_pressure += q_val
                        if direction_count:
                            avg_speed /= direction_count
                        avg_q = sum(queues) / len(queues) if queues else 0.0
                        imbalance = sum(abs(q - avg_q) for q in queues)
                        
                        switched = (engine.time_s - engine.last_switch_time.get(intersection_id, -9999.0)) < 1.1
                        switch_penalty_val = WEIGHT_SWITCH_PENALTY if switched else 0.0
                        
                        queue_penalty = WEIGHT_QUEUE * (total_queue / SCALE_QUEUE)
                        imbalance_penalty = WEIGHT_IMBALANCE * (imbalance / SCALE_IMBALANCE)
                        red_pressure_penalty = WEIGHT_RED_PRESSURE * (red_pressure / SCALE_RED_PRESSURE)
                        speed_bonus_val = WEIGHT_SPEED_BONUS * (avg_speed / SCALE_SPEED)
                        
                        cost = queue_penalty + imbalance_penalty + red_pressure_penalty + switch_penalty_val - speed_bonus_val
                        reward = max(-REWARD_CLIP, min(REWARD_CLIP, REWARD_OFFSET - cost))
                        intersection_rewards[intersection_id] = {
                            "queue_length": total_queue,
                            "imbalance": imbalance,
                            "red_pressure": red_pressure,
                            "switch_penalty": switch_penalty_val,
                            "speed_avg": avg_speed,
                            "speed_bonus": speed_bonus_val,
                            "cost": cost,
                            "reward": reward,
                        }
                        global_reward += reward
                    
                    avg_global_reward = global_reward / max(len(engine.intersections), 1)

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
                        "metrics": asdict(rl_state),
                        "global_reward": avg_global_reward,
                        "reward_metrics": intersection_rewards,
                    }

            # Broadcast to websockets outside the lock
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

@app.get("/api/v1/state/{intersection_id}")
async def get_state(intersection_id: int):
    async with engine_lock:
        if intersection_id not in engine.intersections:
            return {"error": "Intersection not found"}

        intersection = engine.intersections[intersection_id]
        state_data = {}
        light_states = {}
        timing_snapshot = {}

        for incoming in intersection.incoming_nodes:
            metrics = engine.collector.snapshot_direction((intersection_id, incoming), engine.time_s)
            state_data[incoming] = {
                "motorcycle_density": metrics.motorcycle.avg_density,
                "car_density": metrics.car.avg_density,
                "motorcycle_avg_speed": metrics.motorcycle.avg_speed,
                "car_avg_speed": metrics.car.avg_speed,
                "queue_length": metrics.motorcycle.queue_length + metrics.car.queue_length,
            }
            light_states[incoming] = intersection.light_for(engine.time_s, incoming).name
            timing = intersection.timing_snapshot()[incoming]
            timing_snapshot[incoming] = asdict(timing)

        return {
            "intersection_id": intersection_id,
            "time": engine.time_s,
            "incoming_nodes": intersection.incoming_nodes,
            "directions": state_data,
            "light_states": light_states,
            "timings": timing_snapshot,
            "local_imbalance": engine.local_imbalance.get(intersection_id, 0.0),
            "global_imbalance": engine.global_imbalance,
        }

@app.get("/api/v1/reward_metrics/{intersection_id}")
async def get_reward_metrics(intersection_id: int):
    # Retrieve reward metrics: Wait time, Queue length, Throughput, Lật pha
    # This is a simplified version; real reward requires keeping track of wait times and throughput in the engine
    async with engine_lock:
        if intersection_id not in engine.intersections:
            return {"error": "Intersection not found"}

        # 1. Tính Queue Total và Red Pressure
        total_queue = 0
        red_pressure = 0.0
        queues = []
        avg_speed = 0.0
        direction_count = 0
        
        intersection = engine.intersections[intersection_id]
        for incoming in intersection.incoming_nodes:
            metrics = engine.collector.snapshot_direction((intersection_id, incoming), engine.time_s)
            q_val = metrics.motorcycle.queue_length + metrics.car.queue_length
            total_queue += q_val
            queues.append(q_val)
            
            avg_speed += (metrics.motorcycle.avg_speed + metrics.car.avg_speed) / 2.0
            direction_count += 1
            
            # Nếu đèn đang đỏ, cộng vào red_pressure
            if intersection.light_for(engine.time_s, incoming) == LightColor.RED:
                if q_val > 0.0:
                    red_pressure += q_val
                    
        if direction_count:
            avg_speed /= direction_count
            
        # 2. Tính Imbalance
        avg_q = sum(queues) / len(queues) if queues else 0.0
        imbalance = sum(abs(q - avg_q) for q in queues)
        
        # 3. Tính Switch Penalty
        # Kiểm tra xem nút giao vừa đổi pha trong vòng 1.1 giây gần nhất
        switched = (engine.time_s - engine.last_switch_time.get(intersection_id, -9999.0)) < 1.1
        switch_penalty_val = WEIGHT_SWITCH_PENALTY if switched else 0.0
        
        # 4. Tính toán Penalty và Bonus có trọng số
        queue_penalty = WEIGHT_QUEUE * (total_queue / SCALE_QUEUE)
        imbalance_penalty = WEIGHT_IMBALANCE * (imbalance / SCALE_IMBALANCE)
        red_pressure_penalty = WEIGHT_RED_PRESSURE * (red_pressure / SCALE_RED_PRESSURE)
        speed_bonus_val = WEIGHT_SPEED_BONUS * (avg_speed / SCALE_SPEED)
        
        cost = queue_penalty + imbalance_penalty + red_pressure_penalty + switch_penalty_val - speed_bonus_val
        reward = max(-REWARD_CLIP, min(REWARD_CLIP, REWARD_OFFSET - cost))

        return {
            "queue_length": total_queue,
            "imbalance": imbalance,
            "red_pressure": red_pressure,
            "switch_penalty": switch_penalty_val,
            "speed_avg": avg_speed,
            "speed_bonus": speed_bonus_val,
            "cost": cost,
            "reward": reward,
        }

@app.post("/api/v1/action/{intersection_id}")
async def post_action(intersection_id: int, request: ActionRequest):
    async with engine_lock:
        if intersection_id not in engine.intersections:
            return {"error": "Intersection not found"}

        # 0 = keep, 1 = change
        if request.action == 1:
            engine.force_switch_phase(intersection_id)
        return {"status": "ok", "action_taken": request.action}


@app.post("/api/v1/actions")
async def post_actions(request: ActionsRequest):
    async with engine_lock:
        applied = 0
        for intersection_id, action in request.actions.items():
            if intersection_id not in engine.intersections:
                continue
            if action == 1:
                engine.force_switch_phase(intersection_id)
            applied += 1
        return {"status": "ok", "applied": applied}

@app.get("/api/v1/network")
async def get_network():
    async with engine_lock:
        nodes = {node: {"x": engine.network.positions[node][0], "y": engine.network.positions[node][1]} for node in engine.network.all_nodes()}
        edges = []
        for (start, end), edge in engine.network.directed_edges.items():
            if start < end: # Return undirected logical edges for drawing
                edges.append({
                    "start": start,
                    "end": end,
                    "lanes": edge.lanes,
                })
        return {"nodes": nodes, "edges": edges}


@app.post("/api/v1/reset")
async def reset_simulation():
    global engine
    async with engine_lock:
        engine = SimulationEngine(config=SimulationConfig())
        return {
            "status": "ok",
            "time": engine.time_s,
            "intersection_count": len(engine.intersections),
        }

@app.websocket("/ws/simulation/render")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error in endpoint: {e}")
    finally:
        active_connections.discard(websocket)
