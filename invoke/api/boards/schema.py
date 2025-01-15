# Path: invoke\api\boards\schema.py
from pydantic import BaseModel
from typing import List, Optional


class BoardDTO(BaseModel):
    id: str
    name: str
    is_private: bool
    created_at: str
    updated_at: Optional[str] = None


class BoardChanges(BaseModel):
    name: Optional[str] = None
    is_private: Optional[bool] = None


class DeleteBoardResult(BaseModel):
    success: bool
    board_id: str
    images_deleted: Optional[int] = None


class AddImagesToBoardResult(BaseModel):
    added_images: List[str]
    board_id: str


class RemoveImagesFromBoardResult(BaseModel):
    removed_images: List[str]
    board_id: str


class OffsetPaginatedResultsBoardDTO(BaseModel):
    results: List[BoardDTO]
    total: int
    offset: int
    limit: int