# Path: invoke\api\images\images_api.py
import aiohttp
from typing import Optional, Dict, Union, Any, List
from ..api import Api, ResponseType
from .schema import *


class ImageOrigin:
    Internal = "internal"
    External = "external"


class Categories:
    General = "general"
    Mask = "mask"
    Control = "control"
    User = "user"
    Other = "other"


class ImagesApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def get_image_dto(self, image_name: str) -> ImageDto:
        json_data = await self.get_async(f"images/i/{image_name}", 1)
        return ImageDto(**json_data)


    async def list_image_dtos(
        self,
        offset: int = 0,
        limit: int = 10,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = None,
        image_origin: Optional[ImageOrigin] = None,
        categories: Optional[Categories] = None
    ) -> ListImageDtos:
        prams = [
            ("offset", str(offset)),
            ("limit", str(limit)),
            ("is_intermediate", str(is_intermediate).lower() if is_intermediate is not None else None),
            ("image_origin", image_origin if image_origin else None),
            ("categories", categories if categories else None),
            ("board_id", board_id if board_id else None),
        ]
        prams = [(key, value) for key, value in prams if value is not None]

        json_data = await self.get_async("images/", 1, prams)
        return ListImageDtos(**json_data)


    async def delete(self, image_name: str) -> None:
        await self.delete_async(f"images/i/{image_name}", 1)


    async def upload(
        self,
        image: bytes,
        category: Categories = Categories.General,
        is_intermediate: bool = False,
        board_id: Optional[str] = None,
        session_id: Optional[str] = None,
        crop_visible: bool = False
    ) -> ImageUpload:
        prams = [
            ("image_category", category),
            ("is_intermediate", str(is_intermediate).lower()),
            ("board_id", board_id if board_id else None),
            ("session_id", session_id if session_id else None),
            ("crop_visible", str(crop_visible).lower()),
        ]
        prams = [(key, value) for key, value in prams if value is not None]

        json_data = await self.upload_async("images/upload", 1, "file", image, prams)
        return ImageUpload(**json_data)
    

    async def get_full(self, image_name: str) -> bytes:
        response = await self.get_async(f"images/i/{image_name}/full", 1, type=ResponseType.RAW)
        return response.content


    async def get_thumbnail(self, image_name: str) -> bytes:
        response = await self.get_async(f"images/i/{image_name}/thumbnail", 1, type=ResponseType.RAW)
        return response.content


    async def get_urls(self, image_name: str) -> Dict[str, str]:
        json_data = await self.get_async(f"images/i/{image_name}/urls", 1)
        return json_data


    async def get_metadata(self, image_name: str) -> Dict[str, Union[str, int]]:
        json_data = await self.get_async(f"images/i/{image_name}/metadata", 1)
        return json_data


    async def get_workflow(self, image_name: str) -> Dict[str, Any]:
        json_data = await self.get_async(f"images/i/{image_name}/workflow", 1)
        return json_data


    async def delete_intermediates(self) -> None:
        await self.delete_async("images/intermediates", 1)


    async def get_intermediates_count(self) -> int:
        json_data = await self.get_async("images/intermediates", 1)
        return json_data["count"]


    async def delete_by_list(self, image_names: List[str]) -> None:
        await self.post_async("images/delete", 1, {"image_names": image_names})


    async def star(self, image_name: str) -> None:
        await self.post_async("images/star", 1, {"image_names": image_name})


    async def unstar(self, image_name: str) -> None:
        await self.post_async("images/unstar", 1, {"image_names": image_name})


    async def bulk_download(self, image_names: List[str]) -> bytes:
        response = await self.post_async("images/download", 1, {"image_names": image_names}, type=ResponseType.RAW)
        return response.content


    async def get_bulk_download_item(self, bulk_download_item_name: str) -> bytes:
        response = await self.get_async(f"images/download/{bulk_download_item_name}", 1, type=ResponseType.RAW)
        return response.content