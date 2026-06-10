from fastapi import APIRouter
from app.api.endpoints import simulation

api_router = APIRouter()

# Đăng ký các HTTP endpoints với prefix /api/v1
api_router.include_router(simulation.router, prefix="/api/v1")

# Đăng ký WebSocket endpoint ở root / để giữ nguyên path /ws/simulation/render
api_router.include_router(simulation.ws_router)
