# Path: invoke\graph_builder\nodes\clip_skip.py
from .node import Node


class ClipSkip(Node):
    type: str = "clip_skip"
    skipped_layers: int
    is_intermediate: bool = True
    use_cache: bool = True