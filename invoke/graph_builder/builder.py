# Path: invoke\graph_builder\builder.py
from typing import List, Tuple
from .components import BatchRoot
from .nodes.node import Node


class Builder:
    root: BatchRoot

    @property
    def runs(self) -> int:
        return self.root.batch.runs

    @runs.setter
    def runs(self, value: int):
        self.root.batch.runs = value

    @property
    def name(self) -> str:
        return self.root.batch.graph.id

    @name.setter
    def name(self, value: str):
        self.root.batch.graph.id = value


    def __init__(self, name: str, runs: int = 1):
        super().__init__(root=BatchRoot(
            prepend=False,
            batch={
                "graph": {
                    "id": name,
                    "nodes": {},
                    "edges": []
                },
                "runs": runs
            }
        ))


    def add_node(self, node: Node) -> Node:
        self.root.batch.add_node(node)
        return node


    def connect(self, source_node: Node, source_field: str, destination_node: Node, destination_field: str) -> None:
        self.root.batch.graph.edges.append({
            "source": {
                "node_id": source_node.id,
                "field": source_field
            },
            "destination": {
                "node_id": destination_node.id,
                "field": destination_field
            }
        })


    def connect_multiple(self, source_node: Node, source_field: str, destinations: List[Tuple[Node, str]]) -> None:
        for destination in destinations:
            self.connect(source_node, source_field, destination[0], destination[1])


    def connect_list(self, connections: List[Tuple[Node, str, Node, str]]) -> None:
        for connection in connections:
            self.connect(connection[0], connection[1], connection[2], connection[3])


    def build_json(self) -> str:
        return self.root.model_dump_json(indent=4)