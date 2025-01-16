# Path: invoke\api\models\schema.py
from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class Submodel(BaseModel):
    path_or_prefix: str
    model_type: str
    variant: str


class DefaultSettings(BaseModel):
    vae: Optional[str]
    vae_precision: str
    scheduler: str
    steps: int
    cfg_scale: float
    cfg_rescale_multiplier: float
    width: int
    height: int
    guidance: float


class ModelRecord(BaseModel):
    key: str
    hash: str
    path: str
    name: str
    base: str
    description: str
    source: str
    source_type: str
    source_api_response: Optional[str]
    cover_image: Optional[str]
    submodels: Optional[Dict[str, Submodel]]
    type: str
    trigger_phrases: List[str]
    default_settings: DefaultSettings
    variant: str
    format: str
    repo_variant: str


class UpdateModelRecord(BaseModel):
    path: str
    name: str
    base: str
    type: str
    format: str
    config_path: str
    key: str
    hash: str
    description: str
    source: str
    converted_at: int
    variant: str
    prediction_type: str
    repo_variant: str
    upcast_attention: bool


class ValidationErrorDetail(BaseModel):
    loc: List[str]
    msg: str
    type: str


class DeleteModelResponse(BaseModel):
    detail: List[ValidationErrorDetail]


class ScannedModel(BaseModel):
    path: str
    is_installed: bool


class HuggingFaceModelResponse(BaseModel):
    urls: List[str]
    is_diffusers: bool


class ModelConfigIn(BaseModel):
    source: str
    source_type: str
    source_api_response: Optional[str]
    name: str
    path: str
    description: str
    base: str
    type: str
    key: str
    hash: str
    format: str
    trigger_phrases: List[str]
    default_settings: DefaultSettings
    variant: str
    prediction_type: str
    upcast_attention: bool
    config_path: str


class ModelConfigOut(BaseModel):
    key: str
    hash: str
    path: str
    name: str
    base: str
    description: str
    source: str
    source_type: str
    source_api_response: Optional[str]
    cover_image: Optional[str]
    submodels: Optional[Dict[str, Submodel]]
    type: str
    trigger_phrases: List[str]
    default_settings: DefaultSettings
    variant: str
    format: str
    repo_variant: str


class DownloadPart(BaseModel):
    id: int
    dest: str
    download_path: str
    status: str
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


class Source(BaseModel):
    path: str
    inplace: bool
    type: str


class InstallModelResponse(BaseModel):
    id: int
    status: str
    error_reason: Optional[str]
    config_in: ModelConfigIn
    config_out: Optional[ModelConfigOut]
    inplace: bool
    source: Source
    local_path: str
    bytes: int
    total_bytes: int
    source_metadata: Optional[SourceMetadata]
    download_parts: List[DownloadPart]
    error: Optional[str]
    error_traceback: Optional[str]


class ModelInstallJob(BaseModel):
    id: int
    status: str
    error_reason: Optional[str]
    config_in: ModelConfigIn
    config_out: Optional[ModelConfigOut]
    inplace: bool
    source: Source
    local_path: str
    bytes: int
    total_bytes: int
    source_metadata: Optional[SourceMetadata]
    download_parts: List[DownloadPart]
    error: Optional[str]
    error_traceback: Optional[str]


class ConvertedModel(BaseModel):
    path: str
    name: str
    base: str
    type: str
    format: str
    config_path: str
    key: str
    hash: str
    description: str
    source: str
    converted_at: int
    variant: str
    prediction_type: str
    repo_variant: str
    upcast_attention: bool


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
    format: str
    is_installed: bool
    previous_names: List[str]


class StarterModel(BaseModel):
    description: str
    source: str
    name: str
    base: str
    type: str
    format: str
    is_installed: bool
    previous_names: List[str]
    dependencies: List[Dependency]


class StarterModelsResponse(BaseModel):
    starter_models: List[StarterModel]
    starter_bundles: Dict[str, List[StarterModel]]


class HFTokenStatus(str, Enum):
    valid = "valid"
    invalid = "invalid"
    unknown = "unknown"