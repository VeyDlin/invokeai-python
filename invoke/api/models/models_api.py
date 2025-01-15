# Path: invoke\api\models\models_api.py
import aiohttp
from typing import Optional, List, Tuple, Dict, Any
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
    ) -> Models:
        prams: QueryParams = []

        self.add_params(prams, base_models, "base_models")
        self.add_params(prams, name, "model_name")
        self.add_params(prams, model_type, "model_type")
        self.add_params(prams, format, "model_format")
    
        json_data = await self.get_async("models/", 2, prams)
        return Models(**json_data)
    

    async def get_by_attributes(
        self,
        name: str,
        model_type: str,
        base: str
    ) -> Model:
        prams: List[Tuple[str, str]] = [
            ("name", name),
            ("type", model_type),
            ("base", base)
        ]
        json_data = await self.get_async("models/get_by_attrs", 2, prams)
        return Model(**json_data)


    async def get_by_key(self, key: str) -> Model:
        json_data = await self.get_async(f"models/i/{key}", 2)
        return Model(**json_data)


    async def patch_by_key(self, key: str, updates: Dict[str, Any]) -> Model:
        json_data = await self.put_async(f"models/i/{key}", 2, updates)
        return Model(**json_data)


    async def delete_by_key(self, key: str) -> None:
        await self.delete_async(f"models/i/{key}", 2)


    async def scan_folder(self, scan_path: Optional[str] = None) -> List[str]:
        prams: QueryParams = []
        if scan_path:
            prams.append(("scan_path", scan_path))
        json_data = await self.get_async("models/scan_folder", 2, prams)
        return json_data


    async def install(
        self,
        source: str,
        access_token: Optional[str] = None,
        inplace: bool = False
    ) -> Dict[str, Any]:
        prams: QueryParams = [
            ("source", source),
            ("access_token", access_token),
            ("inplace", str(inplace).lower())
        ]
        json_data = await self.post_async("models/install", 2, prams=prams)
        return json_data


    async def list_install_jobs(self) -> List[Dict[str, Any]]:
        json_data = await self.get_async("models/install", 2)
        return json_data


    async def get_install_job(self, job_id: str) -> Dict[str, Any]:
        json_data = await self.get_async(f"models/install/{job_id}", 2)
        return json_data


    async def cancel_install_job(self, job_id: str) -> None:
        await self.delete_async(f"models/install/{job_id}", 2)


    async def prune_completed_jobs(self) -> None:
        await self.delete_async("models/install", 2)


    async def convert(self, key: str) -> None:
        await self.post_async(f"models/convert/{key}", 2)


    async def get_image(self, key: str) -> Optional[bytes]:
        response = await self.get_async(f"models/i/{key}/image", 2, type=ResponseType.RESPONSE)
        return response.content if response.status == 200 else None


    async def update_image(self, key: str, image_bytes: bytes) -> None:
        await self.upload_async(f"models/i/{key}/image", 2, "image", image_bytes)


    async def delete_image(self, key: str) -> None:
        await self.delete_async(f"models/i/{key}/image", 2)


    async def get_stats(self) -> Dict[str, Any]: # TODO
        json_data = await self.get_async("models/stats", 2)
        return json_data