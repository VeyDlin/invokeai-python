# Path: invoke\graph_builder\components\workflow_node_input.py
from typing import Optional, Any
from pydantic import BaseModel


class WorkflowNodeInput(BaseModel):
    name: str
    label: str
    value: Optional[Any]
 
