from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
from transformers import CLIPModel, CLIPProcessor
import torch


@dataclass(frozen=True)
class ClipConfig:
    model: str
    processor: str
    tokenizer: str | None = None
    device: str = "cpu"


class CLIPS:
    laion5b_roberta = ClipConfig(
        "calpt/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "xlm-roberta-base",
    )
    openai_clip = ClipConfig(
        "openai/clip-vit-base-patch32",
        "flash_attention_2",
    )
    fashion_clip = ClipConfig(
        "patrickjohncyh/fashion-clip",
        "openai/clip-vit-base-patch32",
        "bert-base-uncased",
    )


def get_clip(model: str, processor: str, tokenizer: str, device: str) -> tuple[Any, Any, Any]:
    model = AutoModel.from_pretrained(
        model,
        trust_remote_code=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoFeatureExtractor.from_pretrained(processor)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return model, processor, tokenizer


def get_clip_with_processor(model_config: ClipConfig) -> tuple[CLIPModel, CLIPProcessor]:
    if model_config.device == "cuda":
        torch_dtype = torch.float16
        model = CLIPModel.from_pretrained(
            model_config.model,
            attn_implementation=model_config.processor,
            device_map=model_config.device,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    else:
        torch_dtype = torch.float32
        model = CLIPModel.from_pretrained(
            model_config.model,
            device_map={"": model_config.device},
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    processor = CLIPProcessor.from_pretrained(model_config.model)
    return model, processor


def get_clip_from_clip_config(config: ClipConfig) -> tuple[Any, Any, Any]:
    model = AutoModel.from_pretrained(
        config.model,
        trust_remote_code=True,
        use_safetensors=True,
    ).to(config.device)
    processor = AutoFeatureExtractor.from_pretrained(config.processor)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    return model, processor, tokenizer


def query_clip(image, texts: list[str], model: Any, processor: Any, tokenizer: Any):
    device = model.device

    image_inputs = processor(image, return_tensors="pt").to(device)
    text_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    text_inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**image_inputs, **text_inputs)
        return outputs.logits_per_image.softmax(dim=-1)


def query_clip_with_processor(image, texts: list[str], model: CLIPModel, processor: CLIPProcessor):
    device = model.device
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits_per_image.softmax(dim=-1)
