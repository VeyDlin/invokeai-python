# Path: invoke\api\app\app_api.py
import aiohttp
from .schema import *
from ..api import Api, ResponseType


class AppApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)

    async def version(self) -> AppVersion:
        json_data = await self.get_async("app/version", 1)
        return AppVersion(**json_data)


    async def app_deps(self) -> AppDeps:
        json_data = await self.get_async("app/app_deps", 1)
        return AppDeps(**json_data)


    async def config(self) -> AppConfig:
        json_data = await self.get_async("app/config", 1)
        return AppConfig(**json_data)


    async def get_log_level(self) -> int:
        text_data = await self.get_async("app/logging", 1, type=ResponseType.TEXT)
        return int(text_data)


    async def set_log_level(self, log_level: str) -> int:
        data = {"log_level": log_level}
        text_data = await self.post_async("app/logging", 1, data=data)
        return int(text_data)


    async def clear_invocation_cache(self) -> None:
        await self.delete_async("app/invocation_cache", 1)


    async def enable_invocation_cache(self) -> None:
        await self.put_async("app/invocation_cache/enable", 1)


    async def disable_invocation_cache(self) -> None:
        await self.put_async("app/invocation_cache/disable", 1)


    async def get_invocation_cache_status(self) -> CacheStatus:
        json_data = await self.get_async("app/invocation_cache/status", 1)
        return CacheStatus(**json_data)
