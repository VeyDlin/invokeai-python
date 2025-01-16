# Path: invoke\graph_builder\components\batch.py
from typing import List, Dict, Optional, Callable, Any
from pydantic import BaseModel
from uuid import uuid4
from .graph import Graph
from .workflow import Workflow, WorkflowNode
from .edge import Edge, Source, Destination
from ..nodes import Node
from ...api.models import ModelRecord


class Batch(BaseModel):
    graph: Graph
    workflow: Optional[Workflow] = None
    runs: int = 1


    def add_node(self, node: Any) -> dict:
        if isinstance(node, Node):
            jnode = node.to_dict()
        elif isinstance(node, dict):
            jnode = node
        else:
            jnode = node.__dict__ if hasattr(node, "__dict__") else dict(node)

        self.graph.nodes[jnode["id"]] = jnode
        return jnode


    def find_node(self, predicate: Callable[[WorkflowNode], bool]) -> List[Dict]:
        ids = [node.id for node in self.workflow.nodes if predicate(node)]
        return [self.graph.nodes[node_id] for node_id in ids if node_id in self.graph.nodes]


    def delete_node(self, predicate: Callable[[Dict], bool]) -> None:
        delete_nodes = [
            node_id for node_id, node in self.graph.nodes.items() if predicate(node)
        ]
        self.graph.nodes = {
            node_id: node
            for node_id, node in self.graph.nodes.items()
            if node_id not in delete_nodes
        }
        self.graph.edges = [
            edge
            for edge in self.graph.edges
            if edge.source.node_id not in delete_nodes and edge.destination.node_id not in delete_nodes
        ]


    def find_edge(self, predicate: Callable[[Source, Destination], bool]) -> List[Edge]:
        return [edge for edge in self.graph.edges if predicate(edge.source, edge.destination)]


    def connect(self, source_node: Any, source_field: str, destination_node: Any, destination_field: str) -> None:
        self.connect_manual(source_node["id"], source_field, destination_node["id"], destination_field)


    def connect_manual(self, source_node_id: str, source_field: str, destination_node_id: str, destination_field: str) -> None:
        self.graph.edges.append(Edge(
            source=Source(node_id=source_node_id, field=source_field),
            destination=Destination(node_id=destination_node_id, field=destination_field)
        ))


    def set_graph_field_from_exposed(self, label_or_name: str, val: Any) -> None:
        try:
            field = self.workflow.get_exposed_field(label_or_name)
            node = self.graph.nodes[field.node_id]
            node[field.field_name] = val
        except Exception as ex:
            raise Exception(f"Error setting graph field '{label_or_name}': {ex}")


    @staticmethod
    def from_workflow(workflow: Workflow) -> "Batch":
        nodes = {
            node.id: {
                "id": node.id,
                "type": node.data.type,
                "is_intermediate": node.data.is_intermediate,
                "use_cache": node.data.use_cache,
                **{input.name: input.value for input in node.data.inputs.values() if input.value is not None}
            }
            for node in workflow.nodes
        }

        edges = [
            Edge(
                source=Source(node_id=edge.source, field=edge.source_handle),
                destination=Destination(node_id=edge.target, field=edge.target_handle)
            )
            for edge in workflow.edges
        ]

        return Batch(
            graph=Graph(
                id=str(uuid4()),
                nodes=nodes,
                edges=edges
            ),
            workflow=workflow
        )


    def update_models_hash(self, models: List[ModelRecord]) -> None:
        paths = {
            "main_model_loader": "model",
            "sdxl_model_loader": "model",
            "lora_selector": "lora",
            "controlnet": "control_model",
        }

        for node_id, node in self.graph.nodes.items():
            node_type = node["type"]
            if node_type in paths:
                path = paths[node_type]
                name = node[path]["name"]
                matching_models = [model for model in models if model.name == name]

                if matching_models:
                    model = matching_models[0]
                    node[path]["key"] = model["key"]
                    node[path]["hash"] = model["hash"]
                else:
                    raise Exception(f"Error updating hash for model '{name}': model not found")