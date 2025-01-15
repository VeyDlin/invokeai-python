# Path: invoke\api\models\schema.py
from pydantic import BaseModel
from typing import List, Optional


class Model(BaseModel):
    key: str
    hash: str
    path: str
    name: str
    base: str
    description: Optional[str]
    source: str
    source_type: str
    source_api_response: Optional[str]
    cover_image: Optional[str]
    type: str


class Models(BaseModel):
    models: List[Model]