# Path: invoke\api\download_queue\schema.py
from typing import Optional
from pydantic import BaseModel


class DownloadJob(BaseModel):
    id: int
    status: str
    progress: float 
    error: Optional[str]
    source_url: str
    destination_path: str