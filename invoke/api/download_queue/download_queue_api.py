# Path: invoke\api\download_queue\download_queue_api.py
import aiohttp
from typing import Optional, Dict, Any, List
from ..api import Api
from .schema import *


class DownloadQueueApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def submit_download_job(self, source_url: str, destination_path: str, metadata: Optional[Dict[str, Any]] = None) -> DownloadJob:
        data = {
            "source_url": source_url,
            "destination_path": destination_path,
            "metadata": metadata,
        }
        json_data = await self.post_async("download_queue", 1, data)
        return DownloadJob(**json_data)


    async def get_download_job(self, job_id: int) -> DownloadJob:
        json_data = await self.get_async(f"download_queue/i/{job_id}", 1)
        return DownloadJob(**json_data)


    async def cancel_download_job(self, job_id: int) -> None:
        await self.delete_async(f"download_queue/i/{job_id}", 1)


    async def cancel_all_download_jobs(self) -> None:
        await self.delete_async("download_queue/i", 1)


    async def list_all_download_jobs(self) -> List[DownloadJob]:
        json_data = await self.get_async("download_queue", 1)
        return [DownloadJob(**job) for job in json_data]


    async def get_download_job_status(self, job_id: int) -> str:
        json_data = await self.get_async(f"download_queue/status/{job_id}", 1)
        return json_data["status"]