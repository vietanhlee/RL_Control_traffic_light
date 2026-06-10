from pydantic import BaseModel
from typing import Dict

class ActionRequest(BaseModel):
    action: int  # 0 for Keep, 1 for Change

class ActionsRequest(BaseModel):
    actions: Dict[int, int]
