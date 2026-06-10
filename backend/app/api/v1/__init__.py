from fastapi import APIRouter
from .api_state import router as state_router
from .api_action import router as action_router
from .api_reward import router as reward_router
from .api_render import router as render_router

router = APIRouter()

# Đăng ký HTTP endpoints của simulation với prefix /api/v1
router.include_router(state_router, prefix="/api/v1")
router.include_router(action_router, prefix="/api/v1")
router.include_router(reward_router, prefix="/api/v1")

# Đăng ký WebSocket endpoint ở root / để giữ nguyên path /ws/simulation/render
router.include_router(render_router)