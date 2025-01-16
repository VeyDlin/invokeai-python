# Path: invoke\graph_builder\nodes\denoise_latents.py
from .node import Node


class DenoiseLatents(Node):
    type: str = "denoise_latents"
    cfg_scale: float
    cfg_rescale_multiplier: float = 0
    scheduler: str
    steps: int
    denoising_start: float = 0
    denoising_end: float = 1
    is_intermediate: bool = True
    use_cache: bool = True