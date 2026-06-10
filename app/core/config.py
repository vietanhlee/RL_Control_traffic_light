import asyncio
import logging
from typing import Set
from fastapi import WebSocket
from simulation.engine import SimulationEngine
from config.constants import SimulationConfig

# Cấu hình logging ghi ra cả console và file simulation.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("simulation.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("app")

engine = SimulationEngine(config=SimulationConfig())
active_connections: Set[WebSocket] = set()
engine_lock = asyncio.Lock()
