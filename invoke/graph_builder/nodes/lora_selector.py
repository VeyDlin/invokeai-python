# Path: invoke\graph_builder\nodes\lora_selector.py
from pydantic import BaseModel
from .node import Node


class LoraSelector(Node):
    type: str = "lora_selector"
    is_intermediate: bool = True
    use_cache: bool = True
    lora: "Lora"
    weight: float

    class Lora(BaseModel):
        key: str = "-"
        hash: str = "-"
        name: str
        base: str = "sd-1"
        type: str = "lora"