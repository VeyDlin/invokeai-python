# Path: invoke\api\images\images_api.py
import aiohttp
from typing import Optional, Any, List
from ..api import Api, ResponseType, QueryParams
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


    async def upload(
        self,
        image: bytes,
        category: Categories = Categories.General,
        is_intermediate: bool = False,
        board_id: Optional[str] = None,
        session_id: Optional[str] = None,
        crop_visible: bool = False
    ) -> ImageUpload:
        prams: QueryParams = [
            ("image_category", category),
            ("is_intermediate", is_intermediate),
            ("board_id", board_id),
            ("session_id", session_id),
            ("crop_visible", crop_visible),
        ]
        prams = [(key, value) for key, value in prams if value is not None]

        json_data = await self.upload_async("images/upload", 1, "file", image, prams)
        return ImageUpload(**json_data)
    

    async def delete(self, image_name: str) -> None:
        await self.delete_async(f"images/i/{image_name}", 1)


    # TODO update


    async def get_image_dto(self, image_name: str) -> ImageDto:
        json_data = await self.get_async(f"images/i/{image_name}", 1)
        return ImageDto(**json_data)


    async def get_intermediates_count(self) -> int:
        text_data = await self.get_async("images/intermediates", 1, type=ResponseType.TEXT)
        return int(text_data)
    

    async def clear_intermediates(self) -> int:
        text_data = await self.delete_async("images/intermediates", 1, type=ResponseType.TEXT)
        return int(text_data)


    async def get_metadata(self, image_name: str) -> Any:
        json_data = await self.get_async(f"images/i/{image_name}/metadata", 1)
        return json_data


    async def get_workflow(self, image_name: str) -> Any:
        json_data = await self.get_async(f"images/i/{image_name}/workflow", 1)
        return json_data
    

    async def get_full(self, image_name: str) -> bytes:
        response = await self.get_async(f"images/i/{image_name}/full", 1, type=ResponseType.RAW)
        return response.content


    async def get_thumbnail(self, image_name: str) -> bytes:
        response = await self.get_async(f"images/i/{image_name}/thumbnail", 1, type=ResponseType.RAW)
        return response.content


    async def get_urls(self, image_name: str) -> ImageUrls:
        json_data = await self.get_async(f"images/i/{image_name}/urls", 1)
        return ImageUrls(**json_data)


    async def list_image_dtos(
        self,
        offset: int = 0,
        limit: int = 10,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = None,
        image_origin: Optional[ImageOrigin] = None,
        categories: Optional[Categories] = None
    ) -> ListImageDtos:
        prams: QueryParams  = [
            ("offset", offset),
            ("limit", limit),
            ("is_intermediate", is_intermediate),
            ("image_origin", image_origin),
            ("categories", categories),
            ("board_id", board_id),
        ]
        prams = [(key, value) for key, value in prams if value is not None]

        json_data = await self.get_async("images/", 1, prams)
        return ListImageDtos(**json_data)


    async def delete_by_list(self, image_names: List[str]) -> List[str]:
        json_data = await self.post_async("images/delete", 1, {"image_names": image_names})
        return json_data["deleted_images"]


    async def star_list(self, image_names: List[str]) -> List[str]:
        json_data = await self.post_async("images/star", 1, {"image_names": image_names})
        return json_data["updated_image_names"]


    async def unstar_list(self, image_names: List[str]) -> List[str]:
        json_data = await self.post_async("images/unstar", 1, {"image_names": image_names})
        return json_data["updated_image_names"]


    async def download_list(self, image_names: List[str], board_id: Optional[str] = None) -> DownloadImagesResponse:
        json_data = await self.post_async("images/download", 1, {"image_names": image_names, "board_id": board_id})
        return DownloadImagesResponse(**json_data)


    async def get_download_item(self, bulk_download_item_name: str) -> bytes:
        response = await self.get_async(f"images/download/{bulk_download_item_name}", 1, type=ResponseType.RAW)
        return response.content