# Path: invoke\api\boards\boards_api.py
import aiohttp
from ..api import Api, QueryParams
from .schema import *


class BoardsApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def create(self, board_name: str, is_private: bool = False) -> BoardDTO:
        prams: QueryParams = [
            ("board_name", board_name),
            ("is_private", is_private),
        ]
        json_data = await self.post_async("boards/", 1, prams=prams)
        return BoardDTO(**json_data)


    async def list(
        self,
        order_by: Optional[str] = "created_at",
        direction: Optional[str] = "ASC",
        all_boards: bool = False,
        offset: Optional[int] = 0,
        limit: Optional[int] = 10,
        include_archived: bool = False
    ) -> OffsetPaginatedResultsBoardDTO:
        prams: QueryParams = [
            ("order_by", order_by),
            ("direction", direction),
            ("all", all_boards),
            ("offset", offset),
            ("limit", limit),
            ("include_archived", include_archived),
        ]
        prams = [(key, value) for key, value in prams if value is not None]
        json_data = await self.get_async("boards/", 1, prams=prams)
        return OffsetPaginatedResultsBoardDTO(**json_data)


    async def get_board(self, board_id: str) -> BoardDTO:
        json_data = await self.get_async(f"boards/{board_id}", 1)
        return BoardDTO(**json_data)


    async def update_board(self, board_id: str, changes: BoardChanges) -> BoardDTO:
        json_data = await self.patch_async(f"boards/{board_id}", 1, data=changes.dict(exclude_none=True))
        return BoardDTO(**json_data)


    async def delete_board(self, board_id: str, include_images: bool = False) -> DeleteBoardResult:
        data = {
            "board_id": board_id, 
            "include_images": include_images
        }
        json_data = await self.delete_async(f"boards/{board_id}", 1, data)
        return DeleteBoardResult(**json_data)


    async def list_board_image_names(self, board_id: str) -> List[str]:
        json_data = await self.get_async(f"boards/{board_id}/image_names", 1)
        return json_data


    async def add_image_to_board(self, image_name: str, board_id: str) -> None:
        data = {"image_name": image_name, "board_id": board_id}
        await self.post_async("board_images/", 1, data=data)


    async def remove_image_from_board(self, image_name: str, board_id: str) -> None:
        data = {"image_name": image_name, "board_id": board_id}
        await self.delete_async("board_images/", 1, data=data)


    async def add_images_to_board(self, image_names: List[str], board_id: str) -> AddImagesToBoardResult:
        data = {"image_names": image_names, "board_id": board_id}
        json_data = await self.post_async("board_images/batch", 1, data=data)
        return AddImagesToBoardResult(**json_data)


    async def remove_images_from_board(self, image_names: List[str], board_id: str) -> RemoveImagesFromBoardResult:
        data = {"image_names": image_names, "board_id": board_id}
        json_data = await self.post_async("board_images/batch/delete", 1, data=data)
        return RemoveImagesFromBoardResult(**json_data)