# Path: invoke\graph_builder\batch_root.py
from pydantic import BaseModel
from .components.batch import Batch


class BatchRoot(BaseModel):
    prepend: bool = False
    batch: Batch