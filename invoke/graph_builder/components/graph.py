# Path: invoke\graph_builder\components\graph.py
from typing import Dict, List, Any
from pydantic import BaseModel
from .edge import Edge


class Graph(BaseModel):
    id: str
    nodes: Dict[str, Any]
    edges: List[Edge]