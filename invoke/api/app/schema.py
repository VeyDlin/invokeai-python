# Path: invoke\api\app\schema.py
from pydantic import BaseModel
from typing import Optional, List


class AppVersion(BaseModel):
    version: str
    highlights: Optional[str]


class AppDeps(BaseModel):
    accelerate: str
    compel: str
    cuda: str
    diffusers: str
    numpy: str
    opencv: str
    onnx: str
    pillow: str
    python: str
    torch: str
    torchvision: str
    transformers: str
    xformers: str


class UpscalingMethod(BaseModel):
    upscaling_method: str
    upscaling_models: List[str]


class AppConfig(BaseModel):
    infill_methods: List[str]
    upscaling_methods: List[UpscalingMethod]
    nsfw_methods: List[str]
    watermarking_methods: List[str]


class CacheStatus(BaseModel):
    size: int
    hits: int
    misses: int
    enabled: bool
    max_size: int