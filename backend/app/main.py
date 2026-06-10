import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Thêm project root vào sys.path ở cuối để import được core/shared mà không đè CWD main.py
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.v1 import router as api_router
from services.simulation_service import simulation_loop
from api.v1.state import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Khởi chạy các background tasks...")
    task = asyncio.create_task(simulation_loop())
    yield
    # Hủy background task êm ái khi tắt server
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Traffic Simulation API", lifespan=lifespan)

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

if __name__ == "__main__":
    import uvicorn
    from core.config import PORT
    logger.info(f"Khởi chạy Uvicorn Server tại port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
