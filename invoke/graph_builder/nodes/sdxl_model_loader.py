# Path: graph_builder\nodes\sdxl_model_loader.py
from pydantic import BaseModel
from .node import Node


class SDXLModelLoader(Node):
    type: str = "sdxl_model_loader"
    is_intermediate: bool = True
    use_cache: bool = True
    model: "Model"

    class Model(BaseModel):
        key: str = "-"
        hash: str = "-"
        name: str
        base: str = "sdxl"
        type: str = "main"