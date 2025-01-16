# Path: invoke\graph_builder\nodes\latents_to_image.py
from .node import Node


class LatentsToImage(Node):
    type: str = "l2i"
    fp32: bool
    is_intermediate: bool = True
    use_cache: bool = True