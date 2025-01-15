# Path: invoke\graph_builder\components\position.py
from pydantic import BaseModel


class Position(BaseModel):
    x: float
    y: float