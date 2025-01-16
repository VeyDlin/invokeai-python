# Path: invoke\graph_builder\nodes\noise.py
from .node import Node


class Noise(Node):
    type: str = "noise"
    seed: int
    width: int
    height: int
    use_cpu: bool = True
    is_intermediate: bool = True
    use_cache: bool = False