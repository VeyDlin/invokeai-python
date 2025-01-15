# Path: invoke\presets\text_to_image_sd1.py
from typing import List, Optional, Tuple
from uuid import uuid4
from ..graph_builder.builder import Builder
from ..graph_builder.nodes import (
    MainModelLoader, Prompt, Noise, DenoiseLatents,
    LatentsToImage, SaveImage, VaeLoader, LoraLoader
)


class TextToImageSD1:
    model: str
    model_key: str
    model_hash: str
    positive_prompt: str
    negative_prompt: str
    height: int
    width: int
    seed: int
    cfg: float
    scheduler: str
    steps: int
    fp32: bool
    vae: Optional[str] = None
    loras: Optional[List[Tuple[str, float]]] = None


    def build_json(self) -> str:
        builder = Builder(str(uuid4()))

        # Main Model Loader
        main_model_loader = builder.add_node(MainModelLoader(
            model={
                "key": self.model_key,
                "hash": self.model_hash,
                "base": "sd-1",
                "name": self.model
            }
        ))

        # Prompts
        positive_prompt = builder.add_node(Prompt(prompt=self.positive_prompt))
        negative_prompt = builder.add_node(Prompt(prompt=self.negative_prompt))

        # Noise
        noise = builder.add_node(Noise(
            height=self.height,
            width=self.width,
            seed=self.seed
        ))

        # Denoise Latents
        denoise_latents = builder.add_node(DenoiseLatents(
            cfg_scale=self.cfg,
            scheduler=self.scheduler.lower(),
            steps=self.steps
        ))

        # Latents To Image
        latents_to_image = builder.add_node(LatentsToImage(fp32=self.fp32))

        # Save Image
        save_image = builder.add_node(SaveImage(is_intermediate=False))

        # VAE
        if self.vae:
            vae_loader = builder.add_node(VaeLoader(
                vae_model={
                    "base_model": "sd-1",
                    "model_name": self.vae
                }
            ))
            builder.connect(vae_loader, "vae", latents_to_image, "vae")
        else:
            builder.connect(main_model_loader, "vae", latents_to_image, "vae")

        # Loras
        last_connection = main_model_loader
        if self.loras:
            for lora_name, lora_weight in self.loras:
                lora_loader = builder.add_node(LoraLoader(
                    lora={
                        "base_model": "sd-1",
                        "model_name": lora_name
                    },
                    weight=lora_weight
                ))
                builder.connect(last_connection, "unet", lora_loader, "unet")
                builder.connect(last_connection, "clip", lora_loader, "clip")
                last_connection = lora_loader

        builder.connect(last_connection, "unet", denoise_latents, "unet")
        builder.connect(last_connection, "clip", positive_prompt, "clip")
        builder.connect(last_connection, "clip", negative_prompt, "clip")

        # Prompts
        builder.connect(positive_prompt, "conditioning", denoise_latents, "positive_conditioning")
        builder.connect(negative_prompt, "conditioning", denoise_latents, "negative_conditioning")

        # Noise
        builder.connect(noise, "noise", denoise_latents, "noise")

        # Denoise Latents
        builder.connect(denoise_latents, "latents", latents_to_image, "latents")

        # Latents To Image
        builder.connect(latents_to_image, "image", save_image, "image")

        return builder.build_json()