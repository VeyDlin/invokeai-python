# Path: presets\text_to_image_sdxl.py
import random
import sys
from typing import List, Optional, Tuple
from uuid import uuid4
from ...invoke import Invoke
from ..api.queue import EnqueueBatch
from ..graph_builder import Schedulers
from ..graph_builder.components import BatchRoot
from ..graph_builder.builder import Builder
from ..graph_builder.nodes import (
    SDXLModelLoader, SDXLPrompt, Noise, DenoiseLatents,
    LatentsToImage, SaveImage, VaeLoader, SDXLLoraLoader
)


class TextToImageSDXL:
    @staticmethod
    async def run_batch(
        invoke: Invoke, 
        model: str,
        positive: str,
        positive_style: Optional[str] = None,
        negative: str = "",
        negative_style: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
        cfg: float = 7.5,
        scheduler: str = Schedulers.euler_a,
        steps: int = 30,
        fp32: bool = True,
        vae: Optional[str] = None,
        loras: Optional[List[Tuple[str, float]]] = None,
        wait_batch: bool = False
    ) -> EnqueueBatch:
        batch_root = TextToImageSDXL.build_batch(
            model=model,
            model_key="-",
            model_hash="-",
            positive=positive,
            positive_style=positive_style,
            negative=negative,
            negative_style=negative_style,
            height=height,
            width=width,
            seed=seed,
            cfg=cfg,
            scheduler=scheduler,
            steps=steps,
            fp32=fp32,
            vae=vae,
            loras=loras
        )

        all_models = await invoke.models.list()
        batch_root.batch.update_models_hash(all_models)

        enqueue_batch = await invoke.queue.enqueue_batch(batch_root.model_dump_json())
        if wait_batch:
            await invoke.wait_batch(enqueue_batch)
        return enqueue_batch


    @staticmethod
    def build_batch(
        model: str,
        model_key: str,
        model_hash: str,
        positive: str,
        positive_style: Optional[str] = None,
        negative: str = "",
        negative_style: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
        cfg: float = 7.5,
        scheduler: str = Schedulers.euler_a,
        steps: int = 30,
        fp32: bool = True,
        vae: Optional[str] = None,
        loras: Optional[List[Tuple[str, float]]] = None
    ) -> BatchRoot:
        builder = Builder(str(uuid4()))

        # SDXL Model Loader
        sdxl_model_loader = builder.add_node(SDXLModelLoader(
            model={
                "key": model_key,
                "hash": model_hash,
                "base": "sdxl",
                "name": model
            }
        ))

        # Prompts
        positive_prompt = builder.add_node(SDXLPrompt(
            prompt=positive, 
            style=(positive_style if positive_style is not None else positive)
        ))
        negative_prompt = builder.add_node(SDXLPrompt(
            prompt=negative, 
            style=(negative_style if negative_style is not None else negative)
        ))

        # Noise
        noise = builder.add_node(Noise(
            height=height,
            width=width,
            seed=(seed if seed is not None else random.randint(0, sys.maxsize))
        ))

        # Denoise Latents
        denoise_latents = builder.add_node(DenoiseLatents(
            cfg_scale=cfg,
            scheduler=scheduler.lower(),
            steps=steps
        ))

        # Latents To Image
        latents_to_image = builder.add_node(LatentsToImage(fp32=fp32))

        # Save Image
        save_image = builder.add_node(SaveImage(is_intermediate=False))

        # VAE
        if vae:
            vae_loader = builder.add_node(VaeLoader(
                vae_model={
                    "name": vae,
                    "base": "sdxl"
                }
            ))
            builder.connect(vae_loader, "vae", latents_to_image, "vae")
        else:
            builder.connect(sdxl_model_loader, "vae", latents_to_image, "vae")

        # Loras
        last_connection = sdxl_model_loader
        if loras:
            for lora_name, lora_weight in loras:
                lora_loader = builder.add_node(SDXLLoraLoader(
                    lora={
                        "name": lora_name,
                        "base": "sdxl"
                    },
                    weight=lora_weight
                ))
                builder.connect(last_connection, "unet", lora_loader, "unet")
                builder.connect(last_connection, "clip", lora_loader, "clip")
                builder.connect(last_connection, "clip2", lora_loader, "clip2")
                last_connection = lora_loader

        builder.connect(last_connection, "unet", denoise_latents, "unet")

        builder.connect(last_connection, "clip", positive_prompt, "clip")
        builder.connect(last_connection, "clip2", positive_prompt, "clip2")

        builder.connect(last_connection, "clip", negative_prompt, "clip")
        builder.connect(last_connection, "clip2", negative_prompt, "clip2")

        # Prompts
        builder.connect(positive_prompt, "conditioning", denoise_latents, "positive_conditioning")
        builder.connect(negative_prompt, "conditioning", denoise_latents, "negative_conditioning")

        # Noise
        builder.connect(noise, "noise", denoise_latents, "noise")

        # Denoise Latents
        builder.connect(denoise_latents, "latents", latents_to_image, "latents")

        # Latents To Image
        builder.connect(latents_to_image, "image", save_image, "image")

        return builder.root