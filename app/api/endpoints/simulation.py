from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from dataclasses import asdict
from typing import Dict

import app.core.config as config
from app.core.config import engine_lock, active_connections, logger
from app.schemas.simulation import ActionRequest, ActionsRequest
from app.services.simulation_service import calculate_intersection_reward
from simulation.engine import SimulationEngine
from config.constants import SimulationConfig

# Router cho các HTTP endpoints (sẽ được mount dưới prefix /api/v1)
router = APIRouter()

# Router riêng cho WebSocket endpoints (sẽ được mount trực tiếp ở root /)
ws_router = APIRouter()

@router.get("/states")
async def get_states():
    current_engine = config.engine
    async with engine_lock:
        result = {}
        for intersection_id, intersection in current_engine.intersections.items():
            state_data = {}
            light_states = {}
            timing_snapshot = {}

            for incoming in intersection.incoming_nodes:
                metrics = current_engine.collector.snapshot_direction((intersection_id, incoming), current_engine.time_s)
                state_data[incoming] = {
                    "motorcycle_density": metrics.motorcycle.avg_density,
                    "car_density": metrics.car.avg_density,
                    "motorcycle_avg_speed": metrics.motorcycle.avg_speed,
                    "car_avg_speed": metrics.car.avg_speed,
                    "queue_length": metrics.motorcycle.queue_length + metrics.car.queue_length,
                }
                light_states[incoming] = intersection.light_for(current_engine.time_s, incoming).name
                timing = intersection.timing_snapshot()[incoming]
                timing_snapshot[incoming] = asdict(timing)

            result[intersection_id] = {
                "intersection_id": intersection_id,
                "time": current_engine.time_s,
                "incoming_nodes": intersection.incoming_nodes,
                "directions": state_data,
                "light_states": light_states,
                "timings": timing_snapshot,
                "local_imbalance": current_engine.local_imbalance.get(intersection_id, 0.0),
                "global_imbalance": current_engine.global_imbalance,
            }
        return result

@router.get("/state/{intersection_id}")
async def get_state(intersection_id: int):
    current_engine = config.engine
    async with engine_lock:
        if intersection_id not in current_engine.intersections:
            raise HTTPException(status_code=404, detail="Intersection not found")

        intersection = current_engine.intersections[intersection_id]
        state_data = {}
        light_states = {}
        timing_snapshot = {}

        for incoming in intersection.incoming_nodes:
            metrics = current_engine.collector.snapshot_direction((intersection_id, incoming), current_engine.time_s)
            state_data[incoming] = {
                "motorcycle_density": metrics.motorcycle.avg_density,
                "car_density": metrics.car.avg_density,
                "motorcycle_avg_speed": metrics.motorcycle.avg_speed,
                "car_avg_speed": metrics.car.avg_speed,
                "queue_length": metrics.motorcycle.queue_length + metrics.car.queue_length,
            }
            light_states[incoming] = intersection.light_for(current_engine.time_s, incoming).name
            timing = intersection.timing_snapshot()[incoming]
            timing_snapshot[incoming] = asdict(timing)

        return {
            "intersection_id": intersection_id,
            "time": current_engine.time_s,
            "incoming_nodes": intersection.incoming_nodes,
            "directions": state_data,
            "light_states": light_states,
            "timings": timing_snapshot,
            "local_imbalance": current_engine.local_imbalance.get(intersection_id, 0.0),
            "global_imbalance": current_engine.global_imbalance,
        }

@router.get("/reward_metrics/{intersection_id}")
async def get_reward_metrics(intersection_id: int):
    current_engine = config.engine
    async with engine_lock:
        if intersection_id not in current_engine.intersections:
            raise HTTPException(status_code=404, detail="Intersection not found")
        
        # Tái sử dụng hàm tính toán tập trung từ simulation_service
        return calculate_intersection_reward(current_engine, intersection_id, current_engine.time_s)

@router.post("/action/{intersection_id}")
async def post_action(intersection_id: int, request: ActionRequest):
    current_engine = config.engine
    async with engine_lock:
        if intersection_id not in current_engine.intersections:
            raise HTTPException(status_code=404, detail="Intersection not found")

        # 0 = keep, 1 = change
        if request.action == 1:
            current_engine.force_switch_phase(intersection_id)
        return {"status": "ok", "action_taken": request.action}

@router.post("/actions")
async def post_actions(request: ActionsRequest):
    current_engine = config.engine
    async with engine_lock:
        applied = 0
        for intersection_id, action in request.actions.items():
            if intersection_id not in current_engine.intersections:
                continue
            if action == 1:
                current_engine.force_switch_phase(intersection_id)
            applied += 1
        return {"status": "ok", "applied": applied}

@router.get("/network")
async def get_network():
    current_engine = config.engine
    async with engine_lock:
        nodes = {node: {"x": current_engine.network.positions[node][0], "y": current_engine.network.positions[node][1]} for node in current_engine.network.all_nodes()}
        edges = []
        for (start, end), edge in current_engine.network.directed_edges.items():
            if start < end: # Return undirected logical edges for drawing
                edges.append({
                    "start": start,
                    "end": end,
                    "lanes": edge.lanes,
                })
        return {"nodes": nodes, "edges": edges}

@router.post("/reset")
async def reset_simulation():
    async with engine_lock:
        config.engine = SimulationEngine(config=SimulationConfig())
        return {
            "status": "ok",
            "time": config.engine.time_s,
            "intersection_count": len(config.engine.intersections),
        }

@ws_router.websocket("/ws/simulation/render")
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
