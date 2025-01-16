# Path: invoke\graph_builder\nodes\prompt.py
from .node import Node


class Prompt(Node):
    type: str = "compel"
    prompt: str
    is_intermediate: bool = True
    use_cache: bool = True