# Path: invoke\graph_builder\nodes\lora_loader.py
from pydantic import BaseModel
from .node import Node


class LoraLoader(Node):
    type: str = "lora_loader"
    is_intermediate: bool = True
    use_cache: bool = True
    lora: "Lora"
    weight: float

    class Lora(BaseModel):
        model_name: str
        base_model: str