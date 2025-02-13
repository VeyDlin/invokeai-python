# Path: graph_builder\nodes\vae_loader.py
from pydantic import BaseModel
from .node import Node


class VaeLoader(Node):
    type: str = "vae_loader"
    is_intermediate: bool = True
    use_cache: bool = True
    vae_model: "VaeModel"

    class VaeModel(BaseModel):
        key: str = "-"
        hash: str = "-"
        name: str
        base: str
        type: str = "vae"
        class Config:
            protected_namespaces = ()