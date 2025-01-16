# Path: invoke\api\download_queue\download_queue_api.py
import aiohttp
from typing import Optional, List
from ..api import Api
from .schema import *


class DownloadQueueApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def list(self) -> List[DownloadJob]:
        json_data = await self.get_async("download_queue/", 1)
        return [DownloadJob(**job) for job in json_data]


    async def prune(self) -> None:
        await self.path_async(f"download_queue/", 1)
    

    async def download(self, source: str, dest: str, priority: Optional[int] = None, access_token: Optional[str] = None) -> DownloadJob:
        data = {
            "source": source,
            "dest": dest,
            "priority": priority,
            "access_token": access_token,
        }
        json_data = await self.post_async("download_queue/i/", 1, data)
        return DownloadJob(**json_data)
    

    async def get_job(self, job_id: int) -> DownloadJob:
        json_data = await self.get_async(f"download_queue/i/{job_id}", 1)
        return DownloadJob(**json_data)
    

    async def cancel_job(self, job_id: int) -> None:
        await self.delete_async(f"download_queue/i/{job_id}", 1)


    async def cancel_all_jobs(self) -> None:
        await self.delete_async("download_queue/i", 1)