from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.v1.state import active_connections, logger

router = APIRouter()

@router.websocket("/ws/simulation/render")
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
