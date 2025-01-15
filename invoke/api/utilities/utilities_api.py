# Path: invoke\api\utilities\utilities_api.py
import aiohttp
from ..api import Api
from .schema import *


class UtilitiesApi(Api):
    def __init__(self, client: aiohttp.ClientSession, host: str):
        super().__init__(client, host)


    async def parse_dynamic_prompts(self, prompt: str, max_prompts: int = 1000, combinatorial: bool = True) -> DynamicPromptsResponse:
        data = {
            "prompt": prompt,
            "max_prompts": max_prompts,
            "combinatorial": combinatorial,
        }
        json_data = await self.post_async("utilities/dynamicprompts", 1, data=data)
        return DynamicPromptsResponse(**json_data)