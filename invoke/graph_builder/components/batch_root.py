# Path: invoke\graph_builder\components\batch_root.py
from pydantic import BaseModel
from .batch import Batch


class BatchRoot(BaseModel):
    prepend: bool = False
    batch: Batch