# Path: graph_builder\nodes\string.py
from .node import Node


class String(Node):
    type: str = "string"
    value: str
    is_intermediate: bool = True
    use_cache: bool = True