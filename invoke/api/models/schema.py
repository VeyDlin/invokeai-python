# Path: invoke\api\models\schema.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class Submodel(BaseModel):
    path_or_prefix: str
    model_type: str
    variant: str
    class Config:
        protected_namespaces = ()


class ModelRecord(BaseModel):
    source: str
    source_type: str
    name: str
    path: str
    description: Optional[str]
    base: str
    type: str
    key: str
    hash: str
    format: str


class ValidationErrorDetail(BaseModel):
    loc: List[str]
    msg: str
    type: str


class ScannedModel(BaseModel):
    path: str
    is_installed: bool


class HuggingFaceModelResponse(BaseModel):
    urls: List[str]
    is_diffusers: bool


class DownloadPart(BaseModel):
    id: int
    dest: str
    download_path: Optional[str]
    status: Optional[str]
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


class SourceMetadata(BaseModel):
    name: str
    type: str


class Config(BaseModel):
    source: Optional[str]
    source_type: Optional[str]
    name: Optional[str]
    path: Optional[str]
    description: Optional[str]
    base: Optional[str]
    type: Optional[str]
    key: Optional[str]
    hash: Optional[str]
    format: Optional[str]


class ModelInstallJobStatus(str, Enum):
    waiting = "waiting"
    downloading = "downloading"
    downloads_done = "downloads_done"
    running = "running"
    completed = "completed"
    cancelled = "cancelled"
    error = "error"


class ModelInstallJob(BaseModel):
    id: int
    status: ModelInstallJobStatus
    error_reason: Optional[str]
    config_in: Config
    config_out: Optional[Config]
    inplace: bool
    source: Any
    local_path: str
    bytes: int
    total_bytes: int
    source_metadata: Optional[SourceMetadata]
    download_parts: Optional[List[DownloadPart]]
    error: Optional[str]
    error_traceback: Optional[str]


class CachePerformanceStats(BaseModel):
    hits: int
    misses: int
    high_watermark: int
    in_cache: int
    cleared: int
    cache_size: int
    loaded_model_sizes: Dict[str, int]


class Dependency(BaseModel):
    description: str
    source: str
    name: str
    base: str
    type: str
    format: Optional[str]
    is_installed: bool
    previous_names: List[str]
    dependencies: Optional[List["Dependency"]]


class StarterModel(BaseModel):
    description: str
    source: str
    name: str
    base: str
    type: str
    format: Optional[str]
    is_installed: bool
    previous_names: List[str]
    dependencies: Optional[List[Dependency]]


class StarterModelsResponse(BaseModel):
    starter_models: List[StarterModel]
    starter_bundles: Dict[str, List[StarterModel]]


class HFTokenStatus(str, Enum):
    valid = "valid"
    invalid = "invalid"
    unknown = "unknown"