# Path: invoke\graph_builder\components\source.py
from pydantic import BaseModel


class Source(BaseModel):
    node_id: str
    field: str