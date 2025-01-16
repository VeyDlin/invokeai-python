# Path: invoke\api\images\schema.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ImageDto(BaseModel):
    image_name: str
    image_url: str
    thumbnail_url: str
    image_origin: str
    image_category: str
    width: int
    height: int
    created_at: datetime
    updated_at: datetime
    is_intermediate: bool
    session_id: str
    node_id: str
    starred: bool


class ImageUpload(BaseModel):
    image_name: str
    image_url: str
    thumbnail_url: str
    image_origin: str
    image_category: str
    width: int
    height: int
    created_at: Optional[str]
    updated_at: Optional[str]
    deleted_at: Optional[str]
    is_intermediate: bool
    session_id: Optional[str]
    node_id: Optional[str]
    starred: bool
    has_workflow: bool
    board_id: Optional[str]


class ListImageDtos(BaseModel):
    items: List[ImageDto]
    offset: int
    limit: int
    total: int


class ImageUrls(BaseModel):
    image_name: str
    image_url: str
    thumbnail_url: str


class DownloadImagesResponse(BaseModel):
    response: Optional[str]
    bulk_download_item_name: Optional[str]