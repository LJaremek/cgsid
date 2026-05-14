from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import torch
from diffusers import Flux2Pipeline
from diffusers.utils.logging import set_verbosity_error as set_diffusers_verbosity_error
from PIL import Image
from transformers.utils.logging import set_verbosity_error as set_transformers_verbosity_error

from cgsid.core.config import Settings

from .base import BaseImageModel


class FluxModel(BaseImageModel):
    """Wrapper around the FLUX.2 diffusers pipeline."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        settings = Settings()
        set_diffusers_verbosity_error()
        set_transformers_verbosity_error()

        self.model_name = model_name or settings.model_name
        self.cache_dir = cache_dir or str(settings.weights_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = Flux2Pipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
        )
        self.pipe = self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

        if self.device == "cuda":
            if hasattr(self.pipe, "upcast_vae"):
                self.pipe.upcast_vae()
            elif hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                self.pipe.vae.to(dtype=torch.float32)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing("max")
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = " ",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 28,
        true_cfg_scale: float = 4.0,
        max_sequence_length: int = 512,
        seed: Optional[int] = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator_device = "cuda" if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)

        call_signature = inspect.signature(self.pipe.__call__)
        raw_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "guidance_scale": true_cfg_scale,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
        }
        call_kwargs = {
            key: value
            for key, value in raw_kwargs.items()
            if key in call_signature.parameters and value is not None
        }

        result = self.pipe(**call_kwargs)
        return result.images[0]

    @staticmethod
    def save(image: Image.Image, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
