# Path: invoke\graph_builder\components\workflow_node_data.py
from typing import Dict
from pydantic import BaseModel
from .workflow_node_input import WorkflowNodeInput


class WorkflowNodeData(BaseModel):
    id: str
    version: str
    label: str
    notes: str
    type: str
    inputs: Dict[str, WorkflowNodeInput]
    is_open: bool
    is_intermediate: bool
    use_cache: bool