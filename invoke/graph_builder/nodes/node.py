# Path: invoke\graph_builder\nodes\node.py
from abc import ABC
from typing import ClassVar
from pydantic import BaseModel
import itertools


class Node(BaseModel, ABC):
    id: str
    type: str 
    _id_counter: ClassVar[itertools.count] = itertools.count() 


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = self.create_id(self.type)


    @staticmethod
    def create_id(node_type: str) -> str:
        return f"{node_type}_{next(Node._id_counter)}"


    def to_dict(self) -> dict:
        return self.model_dump(exclude={"_id_counter"})


    class Config:
        arbitrary_types_allowed = True