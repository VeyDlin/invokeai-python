# Path: invoke\api\boards\schema.py
from pydantic import BaseModel
from typing import List, Optional


class BoardDTO(BaseModel):
    board_id: str
    board_name: str
    created_at: str
    updated_at: str
    deleted_at: Optional[str]
    cover_image_name: Optional[str]
    archived: bool
    is_private: Optional[bool]
    image_count: int


class BoardChanges(BaseModel):
    name: Optional[str] = None
    is_private: Optional[bool] = None


class DeleteBoardResult(BaseModel):
    board_id: str
    deleted_board_images: List[str]
    deleted_images: List[str]


class OffsetPaginatedResultsBoardDTO(BaseModel):
    results: List[BoardDTO]
    total: int
    offset: int
    limit: int