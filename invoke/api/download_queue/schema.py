# Path: invoke\api\download_queue\schema.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DownloadJobStatus(str, Enum):
    waiting = "waiting"
    running = "running"
    completed = "completed"
    cancelled = "cancelled"
    error = "error"


class DownloadJob(BaseModel):
    id: int
    dest: str
    download_path: str
    status: DownloadJobStatus
    bytes: int
    total_bytes: int
    error_type: Optional[str]
    error: Optional[str]
    source: str
    access_token: Optional[str]
    priority: int
    job_started: Optional[str]
    job_ended: Optional[str]
    content_type: Optional[str]