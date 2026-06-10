import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.services.simulation_service import simulation_loop
from app.core.config import logger

app = FastAPI(title="Traffic Simulation API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Router chính
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Khởi chạy các background tasks...")
    asyncio.create_task(simulation_loop())
