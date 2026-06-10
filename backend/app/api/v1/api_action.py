from fastapi import APIRouter, HTTPException
import api.v1.state as config
from api.v1.state import engine_lock
from schemas.simulation import ActionRequest, ActionsRequest

router = APIRouter()

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
