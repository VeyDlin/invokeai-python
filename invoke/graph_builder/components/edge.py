# Path: invoke\graph_builder\components\edge.py
from pydantic import BaseModel
from .source import Source
from .destination import Destination


class Edge(BaseModel):
    source: Source
    destination: Destination
