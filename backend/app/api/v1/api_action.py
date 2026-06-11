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

        # action sẽ là index của pha (0, 1, 2, 3)
        current_engine.set_active_phase(intersection_id, request.action)
        return {"status": "ok", "action_taken": request.action}


@router.post("/actions")
async def post_actions(request: ActionsRequest):
    current_engine = config.engine
    async with engine_lock:
        applied = 0
        for intersection_id, action in request.actions.items():
            if intersection_id not in current_engine.intersections:
                continue
            current_engine.set_active_phase(intersection_id, action)
            applied += 1
        return {"status": "ok", "applied": applied}
