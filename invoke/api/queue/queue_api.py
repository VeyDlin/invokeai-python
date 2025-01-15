# Path: invoke\api\queue\queue_api.py
import aiohttp
from typing import Optional, Any
from ..api import Api
from .schema import *


class QueueApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str, queue_id: str = "default"):
        super().__init__(client, host)
        self.queue_id = queue_id

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


    async def enqueue_batch(self, data: Any) -> EnqueueBatch:
        json_data = await self.post_async(f"queue/{self.queue_id}/enqueue_batch", 1, data)
        return EnqueueBatch(**json_data)


    async def get_batch_status(self, batch_id: str) -> BatchStatus:
        json_data = await self.get_async(f"queue/{self.queue_id}/b/{batch_id}/status", 1)
        return BatchStatus(**json_data)


    async def get_queue_item(self, item_id: str) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/i/{item_id}", 1)
        return SessionQueueItem(**json_data)


    async def clear(self) -> Clear:
        json_data = await self.put_async(f"queue/{self.queue_id}/clear", 1)
        return Clear(**json_data)
    

    async def prune(self) -> PruneResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/prune", 1)
        return PruneResponse(**json_data)


    async def get_status(self) -> QueueStatus:
        json_data = await self.get_async(f"queue/{self.queue_id}/status", 1)
        return QueueStatus(**json_data)


    async def get_current_item(self) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/current", 1)
        return SessionQueueItem(**json_data)


    async def get_next_item(self) -> SessionQueueItem:
        json_data = await self.get_async(f"queue/{self.queue_id}/next", 1)
        return SessionQueueItem(**json_data)


    async def resume_processor(self) -> ProcessorResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/processor/resume", 1)
        return ProcessorResponse(**json_data)


    async def pause_processor(self) -> ProcessorResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/processor/pause", 1)
        return ProcessorResponse(**json_data)


    async def cancel_queue_item(self, item_id: str) -> CancelQueueItemResponse:
        json_data = await self.put_async(f"queue/{self.queue_id}/i/{item_id}/cancel", 1)
        return CancelQueueItemResponse(**json_data)


    async def cancel_by_destination(self, destination: str) -> CancelResponse:
        prams = [("destination", destination)]
        json_data = await self.put_async(f"queue/{self.queue_id}/cancel_by_destination", 1, prams=prams)
        return CancelResponse(**json_data)


    async def cancel_by_batch_ids(self, batch_ids: List[str]) -> CancelByBatchIdsResponse:
        data = {"batch_ids": batch_ids}
        json_data = await self.put_async(f"queue/{self.queue_id}/cancel_by_batch_ids", 1, data=data)
        return CancelByBatchIdsResponse(**json_data)


    async def counts_by_destination(self, destination: Optional[str] = None) -> CountsByDestinationResponse:
        prams = [("destination", destination)] if destination else []
        json_data = await self.get_async(f"queue/{self.queue_id}/counts_by_destination", 1, prams=prams)
        return CountsByDestinationResponse(**json_data)
