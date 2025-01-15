# Path: invoke\graph_builder\components\workflow_node.py
from pydantic import BaseModel
from .workflow_node_data import WorkflowNodeData
from .position import Position


class WorkflowNode(BaseModel):
    id: str
    type: str
    data: WorkflowNodeData
    position: Position