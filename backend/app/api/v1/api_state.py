from fastapi import APIRouter, HTTPException
from dataclasses import asdict
import api.v1.state as config
from api.v1.state import engine_lock
from services.simulation_service import calculate_intersection_reward
from models.simulation.engine import SimulationEngine
from core.config import SimulationConfig

router = APIRouter()

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

            # Tính toán các chỉ số tập trung ở BE
            queue_total = sum(float(d["queue_length"]) for d in state_data.values())
            density_total = sum(float(d["motorcycle_density"]) + float(d["car_density"]) for d in state_data.values())
            
            speeds = []
            for d in state_data.values():
                speeds.append((float(d["motorcycle_avg_speed"]) + float(d["car_avg_speed"])) / 2.0)
            speed_avg = sum(speeds) / len(speeds) if speeds else 0.0

            local_imbalance = current_engine.local_imbalance.get(intersection_id, 0.0)

            red_pressure = 0.0
            for incoming, color in light_states.items():
                if color.upper() == "RED":
                     red_pressure += state_data[incoming]["queue_length"]

            current_phase = 0
            for idx, incoming in enumerate(intersection.incoming_nodes):
                if light_states[incoming] in ("GREEN", "YELLOW"):
                    current_phase = idx
                    break

            reward_res = calculate_intersection_reward(current_engine, intersection_id, current_engine.time_s)

            result[intersection_id] = {
                "intersection_id": intersection_id,
                "time": current_engine.time_s,
                "incoming_nodes": intersection.incoming_nodes,
                "directions": state_data,
                "light_states": light_states,
                "timings": timing_snapshot,
                "local_imbalance": local_imbalance,
                "global_imbalance": current_engine.global_imbalance,
                "queue_total": queue_total,
                "density_total": density_total,
                "speed_avg": speed_avg,
                "red_pressure": red_pressure,
                "imbalance": local_imbalance,
                "current_phase": current_phase,
                "reward": reward_res["reward"],
                "reward_metrics": reward_res,
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

        # Tính toán các chỉ số tập trung ở BE
        queue_total = sum(float(d["queue_length"]) for d in state_data.values())
        density_total = sum(float(d["motorcycle_density"]) + float(d["car_density"]) for d in state_data.values())
        
        speeds = []
        for d in state_data.values():
            speeds.append((float(d["motorcycle_avg_speed"]) + float(d["car_avg_speed"])) / 2.0)
        speed_avg = sum(speeds) / len(speeds) if speeds else 0.0

        local_imbalance = current_engine.local_imbalance.get(intersection_id, 0.0)

        red_pressure = 0.0
        for incoming, color in light_states.items():
            if color.upper() == "RED":
                red_pressure += state_data[incoming]["queue_length"]

        current_phase = 0
        for idx, incoming in enumerate(intersection.incoming_nodes):
            if light_states[incoming] in ("GREEN", "YELLOW"):
                current_phase = idx
                break

        reward_res = calculate_intersection_reward(current_engine, intersection_id, current_engine.time_s)

        return {
            "intersection_id": intersection_id,
            "time": current_engine.time_s,
            "incoming_nodes": intersection.incoming_nodes,
            "directions": state_data,
            "light_states": light_states,
            "timings": timing_snapshot,
            "local_imbalance": local_imbalance,
            "global_imbalance": current_engine.global_imbalance,
            "queue_total": queue_total,
            "density_total": density_total,
            "speed_avg": speed_avg,
            "red_pressure": red_pressure,
            "imbalance": local_imbalance,
            "current_phase": current_phase,
            "reward": reward_res["reward"],
            "reward_metrics": reward_res,
        }

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
