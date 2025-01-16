# Path: invoke\api\models\models_api.py
import aiohttp
from typing import Optional, List
from ..api import Api, QueryParams, ResponseType
from .schema import *


class BaseModels:
    SD1 = "sd-1"
    SD2 = "sd-2"
    SD3 = "sd-3"
    SDXL = "sdxl"
    SDXLRefiner = "sdxl-refiner"
    Flux = "flux"
    Any = "any"


class ModelType:
    Main = "main"
    VAE = "vae"
    LoRA = "lora"
    Embedding = "embedding"
    Controlnet = "controlnet"
    T2IAdapter = "t2i_adapter"
    Onnx = "onnx"
    IPAdapter = "ip_adapter"
    ClipVision = "clip_vision"


class ModelsApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def list(
        self,
        base_models: Optional[List[BaseModels]] = None,
        name: Optional[List[str]] = None,
        model_type: Optional[List[ModelType]] = None,
        format: Optional[List[str]] = None
    ) -> List[ModelRecord]:
        prams: QueryParams = []

        self.add_params(prams, base_models, "base_models")
        self.add_params(prams, name, "model_name")
        self.add_params(prams, model_type, "model_type")
        self.add_params(prams, format, "model_format")
    
        json_data = await self.get_async("models/", 2, prams)
        return [ModelRecord(**job) for job in json_data["models"]]
    

    async def get_by_attributes(
        self,
        name: str,
        model_type: str,
        base: str
    ) -> ModelRecord:
        prams: QueryParams = [
            ("name", name),
            ("type", model_type),
            ("base", base)
        ]
        json_data = await self.get_async("models/get_by_attrs", 2, prams)
        return ModelRecord(**json_data)


    async def get_by_key(self, key: str) -> ModelRecord:
        json_data = await self.get_async(f"models/i/{key}", 2)
        return ModelRecord(**json_data)


    async def update(
        self, 
        key: str, 
        path: Optional[str] = None,
        name: Optional[str] = None,
        base: Optional[str] = None,
        model_type: Optional[str] = None,
        format: Optional[str] = None,
        config_path: Optional[str] = None,
        description: Optional[str] = None,
        variant: Optional[str] = None
    ) -> UpdateModelRecord:
        params: QueryParams = [
            ("path", path),
            ("name", name),
            ("base", base),
            ("type", model_type),
            ("format", format),
            ("config_path", config_path),
            ("description", description),
            ("variant", variant)
        ]
        json_data = await self.path_async(f"models/i/{key}", 2, params)
        return UpdateModelRecord(**json_data)
    

    async def delete(self, key: str) -> None:
        await self.delete_async(f"models/i/{key}", 2)


    async def scan_folder(self, scan_path: str) -> List[ScannedModel]:
        prams: QueryParams = [
            ("scan_path", scan_path)
        ]
        json_data = await self.get_async("models/scan_folder", 2, prams)
        return [ScannedModel(**item) for item in json_data]


    async def get_hugging_face_models(self, repo_name: str) -> HuggingFaceModelResponse:
        prams: QueryParams = [
            ("repo_name", repo_name)
        ]
        json_data = await self.get_async("models/hugging_face", 2, prams)
        return HuggingFaceModelResponse(**json_data)
    

    async def get_image(self, key: str) -> Optional[bytes]:
        response = await self.get_async(f"models/i/{key}/image", 2, type=ResponseType.RESPONSE)
        return response.content if response.status == 200 else None
    

    async def update_image(self, key: str, image_bytes: bytes) -> None:
        params: QueryParams = [
            ("image", image_bytes),
        ]
        await self.path_async(f"models/i/{key}/image", 2, params)


    async def delete_image(self, key: str) -> None:
        await self.delete_async(f"models/i/{key}/image", 2)


    async def install(
        self,
        source: str,
        access_token: Optional[str] = None,
        inplace: bool = False
    ) -> InstallModelResponse:
        prams: QueryParams = [
            ("source", source),
            ("access_token", access_token),
            ("inplace", str(inplace).lower())
        ]
        json_data = await self.post_async("models/install", 2, prams=prams)
        return InstallModelResponse(**json_data)


    async def list_install_jobs(self) -> List[ModelInstallJob]:
        json_data = await self.get_async("models/install", 2)
        return [ModelInstallJob(**item) for item in json_data]


    async def prune_completed_jobs(self) -> None:
        await self.delete_async("models/install", 2)


    async def install_huggingface(self, source: str) -> None:
        prams: QueryParams = [
            ("source", source),
        ]
        await self.get_async("models/install/huggingface", 2, prams=prams)


    async def get_install_job(self, job_id: str) -> ModelInstallJob:
        json_data = await self.get_async(f"models/install/{job_id}", 2)
        return ModelInstallJob(**json_data)


    async def cancel_install_job(self, job_id: str) -> None:
        await self.delete_async(f"models/install/{job_id}", 2)


    async def convert(self, key: str) -> ConvertedModel:
        json_data = await self.put_async(f"models/convert/{key}", 2)
        return ConvertedModel(**json_data)


    async def get_starter_models(self) -> StarterModelsResponse:
        json_data = await self.get_async("models/starter_models", 2)
        return StarterModelsResponse(**json_data)


    async def cache_performance_statistics(self) -> CachePerformanceStats:
        json_data = await self.get_async("models/stats", 2)
        return CachePerformanceStats(**json_data)
    

    async def get_huggingface_login_status(self) -> HFTokenStatus:
        text_data = await self.get_async("models/hf_login", 2, type=ResponseType.TEXT)
        return HFTokenStatus(text_data)


    async def do_huggingface_login(self, token: str) -> HFTokenStatus:
        prams: QueryParams = [
            ("token", token),
        ]
        text_data = await self.post_async("models/hf_login", 2, prams=prams, type=ResponseType.TEXT)
        return HFTokenStatus(text_data)