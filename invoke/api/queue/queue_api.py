# Path: invoke\api\queue\queue_api.py
import aiohttp
from typing import Optional, Any
from ..api import Api
from .schema import *


class QueueApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str, queue_id: str = "default"):
        super().__init__(client, host)
        self.queue_id = queue_id


    async def enqueue_batch(self, data: Any) -> EnqueueBatch:
        json_data = await self.post_async(f"queue/{self.queue_id}/enqueue_batch", 1, data)
        return EnqueueBatch(**json_data)
    

    async def list(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        cursor: Optional[str] = None,
        priority: int = 0
    ) -> CursorPaginatedResults:
        prams = [
            ("limit", str(limit)),
            ("priority", str(priority)),
            ("status", status),
            ("cursor", cursor)
        ]
        prams = [(key, value) for key, value in prams if value is not None]

        json_data = await self.get_async(f"queue/{self.queue_id}/list", 1, prams)
        return CursorPaginatedResults(**json_data)


    async def resume(self) -> ProcessorResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/processor/resume", 1)
        return ProcessorResponse(**json_data)


    async def pause(self) -> ProcessorResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/processor/pause", 1)
        return ProcessorResponse(**json_data)


    async def cancel_by_batch_ids(self, batch_ids: List[str]) -> List[str]:
        data = {"batch_ids": batch_ids}
        json_data = await self.put_async(f"queue/{self.queue_id}/cancel_by_batch_ids", 1, data=data)
        return json_data["batch_ids"]


    async def cancel_by_destination(self, destination: str) -> int:
        prams = [("destination", destination)]
        json_data = await self.put_async(f"queue/{self.queue_id}/cancel_by_destination", 1, prams=prams)
        return int(json_data["canceled"])


    async def clear(self) -> int:
        json_data = await self.put_async(f"queue/{self.queue_id}/clear", 1)
        return int(json_data["deleted"])
    

    async def prune(self) -> int:
        json_data = await self.put_async(f"queue/{self.queue_id}/prune", 1)
        return int(json_data["deleted"])


    async def get_current_item(self) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/current", 1)
        return SessionQueueItem(**json_data)


    async def get_next_item(self) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/next", 1)
        return SessionQueueItem(**json_data)


    async def get_queue_status(self) -> QueueProcessorStatus:
        json_data = await self.get_async(f"queue/{self.queue_id}/status", 1)
        return QueueProcessorStatus(**json_data)
    

    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        json_data = await self.get_async(f"queue/{self.queue_id}/b/{batch_id}/status", 1)
        return BatchStatus(**json_data)


    async def get_queue_item(self, item_id: str) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/i/{item_id}", 1)
        return SessionQueueItem(**json_data)


    async def cancel_queue_item(self, item_id: str) -> SessionQueueItem:
        json_data = await self.put_async(f"queue/{self.queue_id}/i/{item_id}/cancel", 1)
        return SessionQueueItem(**json_data)


    async def counts_by_destination(self, destination: Optional[str] = None) -> BatchStatus:
        prams = [("destination", destination)] if destination else []
        json_data = await self.get_async(f"queue/{self.queue_id}/counts_by_destination", 1, prams=prams)
        return BatchStatus(**json_data)