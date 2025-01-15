# Path: invoke\api\app\schema.py
from pydantic import BaseModel
from typing import Dict, Optional, Any


class AppVersion(BaseModel):
    version: str
    highlights: Optional[str]


class AppDeps(BaseModel):
    dependencies: Dict[str, str]


class AppConfig(BaseModel):
    infill_methods: Optional[Dict[str, Any]]
    upscaling_methods: Optional[Dict[str, Any]]
    nsfw_methods: Optional[Dict[str, Any]]
    other_settings: Optional[Dict[str, Any]] 


class LogLevel(BaseModel):
    log_level: str


class CacheStatus(BaseModel):
    enabled: bool
    cache_size: Optional[int]
    cache_hits: Optional[int]
    cache_misses: Optional[int]