from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseImageModel(ABC):
    """Abstract interface for text-to-image models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = " ",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> Any:
        """Generate image for a prompt."""

    @staticmethod
    @abstractmethod
    def save(image: Any, output_path: str) -> None:
        """Save image to disk."""
