# Path: invoke\graph_builder\components\destination.py
from pydantic import BaseModel


class Destination(BaseModel):
    node_id: str
    field: str