# Path: graph_builder\nodes\sdxl_prompt.py
from .node import Node


class SDXLPrompt(Node):
    type: str = "sdxl_compel_prompt"
    prompt: str
    style: str
    original_width: int = 1024
    original_height: int = 1024
    crop_top: int = 0
    crop_left: int = 0
    target_width: int = 1024
    target_height: int = 1024
    is_intermediate: bool = True
    use_cache: bool = True