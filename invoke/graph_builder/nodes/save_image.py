# Path: invoke\graph_builder\nodes\save_image.py
from .node import Node


class SaveImage(Node):
    type: str = "save_image"
    is_intermediate: bool = True
    use_cache: bool = False