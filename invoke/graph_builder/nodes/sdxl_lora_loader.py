# Path: graph_builder\nodes\sdxl_lora_loader.py
from pydantic import BaseModel
from .node import Node


class SDXLLoraLoader(Node):
    type: str = "sdxl_lora_loader"
    is_intermediate: bool = True
    use_cache: bool = True
    lora: "Lora"
    weight: float

    class Lora(BaseModel):
        key: str = "-"
        hash: str = "-"
        name: str
        base: str = "sdxl"
        type: str = "lora"