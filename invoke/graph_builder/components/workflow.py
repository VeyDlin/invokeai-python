# Path: invoke\graph_builder\components\workflow.py
from typing import List, Optional, Any
from pydantic import BaseModel
from .workflow_node import WorkflowNode
from .workflow_edge import WorkflowEdge
from .exposed_field import ExposedField


class Workflow(BaseModel):
    name: str
    author: str
    description: str
    version: str
    contact: str
    tags: str
    notes: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge] 
    exposed_fields: Optional[List[ExposedField]] = None 
    meta: Optional[Any] = None


    def get_exposed_field(self, label_or_name: str) -> Optional[ExposedField]:
        if not label_or_name or not self.exposed_fields:
            return None

        for field in self.exposed_fields:
            matching_nodes = [
                node for node in self.nodes if node.id == field.node_id
            ]
            if not matching_nodes:
                continue

            inputs = matching_nodes[0].data.inputs
            any_input = [
                input_item
                for input_item in inputs.values()
                if (
                    input_item.label.lower() == label_or_name.lower()
                    or input_item.name.lower() == label_or_name.lower()
                    or input_item.name.lower().replace("_", " ") == label_or_name.lower()
                )
            ]
            if any_input and any_input[0].name == field.field_name:
                return field

        return None