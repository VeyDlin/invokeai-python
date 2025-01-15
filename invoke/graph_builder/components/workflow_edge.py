# Path: invoke\graph_builder\components\workflow_edge.py
from pydantic import BaseModel


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    source_handle: str
    target_handle: str