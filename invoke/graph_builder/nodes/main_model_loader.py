# Path: invoke\graph_builder\nodes\main_model_loader.py
from pydantic import BaseModel
from .node import Node


class MainModelLoader(Node):
    type: str = "main_model_loader"
    is_intermediate: bool = True
    use_cache: bool = True
    model: "Model"

    class Model(BaseModel):
        key: str = "-"
        hash: str = "-"
        name: str
        base: str = "sd-1"
        type: str = "main"