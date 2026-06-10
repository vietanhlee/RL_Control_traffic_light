from fastapi import APIRouter, HTTPException
import api.v1.state as config
from api.v1.state import engine_lock
from services.simulation_service import calculate_intersection_reward

router = APIRouter()

@router.get("/reward_metrics/{intersection_id}")
async def get_reward_metrics(intersection_id: int):
    current_engine = config.engine
    async with engine_lock:
        if intersection_id not in current_engine.intersections:
            raise HTTPException(status_code=404, detail="Intersection not found")
        
        return calculate_intersection_reward(current_engine, intersection_id, current_engine.time_s)
