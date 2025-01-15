# Path: invoke\api\utilities\schema.py
from pydantic import BaseModel
from typing import List, Optional


class DynamicPromptsResponse(BaseModel):
    prompts: List[str]
    error: Optional[str]