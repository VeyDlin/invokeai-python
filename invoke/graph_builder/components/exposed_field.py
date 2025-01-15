# Path: invoke\graph_builder\components\exposed_field.py
from pydantic import BaseModel


class ExposedField(BaseModel):
    node_id: str
    field_name: str