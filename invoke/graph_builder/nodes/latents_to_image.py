# Path: graph_builder\nodes\latents_to_image.py
from .node import Node


class LatentsToImage(Node):
    type: str = "l2i"
    tiled: bool = False
    tile_size: int = 0
    fp32: bool = False
    is_intermediate: bool = True
    use_cache: bool = True