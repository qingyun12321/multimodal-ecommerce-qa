from __future__ import annotations

import gc
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    ChineseCLIPModel,
    ChineseCLIPProcessor,
    CLIPModel,
    CLIPProcessor,
)

from ecom_rag.gme_retriever import GMERetriever


def choose_torch_dtype(device: str, torch_dtype: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    return getattr(torch, torch_dtype)


def batched(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class BaseRetriever:
    name: str
    model_id: str
    device: str

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def cleanup(self) -> None:
        raise NotImplementedError


class HFDualEncoderRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        processor_loader,
        model_loader,
        device: str,
        torch_dtype: str,
        text_padding: str | bool = True,
        text_max_length: int | None = None,
        force_lowercase_text: bool = False,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device = device
        self.text_padding = text_padding
        self.text_max_length = text_max_length
        self.force_lowercase_text = force_lowercase_text
        dtype = choose_torch_dtype(device, torch_dtype)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        local_model_dir = snapshot_download(repo_id=model_id)
        self.processor = processor_loader.from_pretrained(local_model_dir)
        self.model = model_loader.from_pretrained(local_model_dir, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        prepared_texts = [
            text.lower() if self.force_lowercase_text else text
            for text in texts
        ]
        model_max_length = getattr(self.processor.tokenizer, "model_max_length", None)
        use_truncation = (
            isinstance(model_max_length, int)
            and model_max_length > 0
            and model_max_length < 100_000
        )
        for batch in tqdm(
            list(batched(list(prepared_texts), batch_size)),
            desc=f"{self.name}: texts",
            leave=False,
        ):
            processor_kwargs = {
                "text": batch,
                "return_tensors": "pt",
                "padding": self.text_padding,
            }
            if self.text_max_length is not None:
                processor_kwargs["max_length"] = self.text_max_length
            if use_truncation:
                processor_kwargs["truncation"] = True
            inputs = self.processor(**processor_kwargs)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(**inputs)
            embeddings.append(self._to_numpy(features))
        return np.concatenate(embeddings, axis=0)

    @torch.inference_mode()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for batch_paths in tqdm(
            list(batched(list(image_paths), batch_size)),
            desc=f"{self.name}: images",
            leave=False,
        ):
            images = [load_rgb_image(path) for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)
            embeddings.append(self._to_numpy(features))
        return np.concatenate(embeddings, axis=0)

    def _unwrap_features(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(output, attr):
                value = getattr(output, attr)
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(f"Unsupported feature output type: {type(output)!r}")

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        tensor = self._unwrap_features(tensor)
        normalized = F.normalize(tensor.float(), p=2, dim=-1)
        array = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return array

    def cleanup(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


class CLIPRetriever(HFDualEncoderRetriever):
    def __init__(self, *, device: str, torch_dtype: str) -> None:
        super().__init__(
            name="clip",
            model_id="openai/clip-vit-base-patch32",
            processor_loader=CLIPProcessor,
            model_loader=CLIPModel,
            device=device,
            torch_dtype=torch_dtype,
        )


class ChineseCLIPRetriever(HFDualEncoderRetriever):
    def __init__(self, *, device: str, torch_dtype: str) -> None:
        super().__init__(
            name="chinese_clip",
            model_id="OFA-Sys/chinese-clip-vit-base-patch16",
            processor_loader=ChineseCLIPProcessor,
            model_loader=ChineseCLIPModel,
            device=device,
            torch_dtype=torch_dtype,
        )


class SigLIP2Retriever(HFDualEncoderRetriever):
    def __init__(self, *, device: str, torch_dtype: str) -> None:
        super().__init__(
            name="siglip2",
            model_id="google/siglip2-base-patch16-224",
            processor_loader=AutoProcessor,
            model_loader=AutoModel,
            device=device,
            torch_dtype=torch_dtype,
            text_padding="max_length",
            text_max_length=64,
            force_lowercase_text=True,
        )


@dataclass(slots=True)
class ModelRunConfig:
    name: str
    image_batch_size: int
    text_batch_size: int
    torch_dtype: str
    max_image_tokens: int | None = None


def build_retriever(config: ModelRunConfig, device: str) -> BaseRetriever:
    if config.name == "clip":
        return CLIPRetriever(device=device, torch_dtype=config.torch_dtype)
    if config.name == "chinese_clip":
        return ChineseCLIPRetriever(device=device, torch_dtype=config.torch_dtype)
    if config.name == "siglip2":
        return SigLIP2Retriever(device=device, torch_dtype=config.torch_dtype)
    if config.name == "gme_2b":
        return GMERetriever(
            device=device,
            torch_dtype=config.torch_dtype,
            max_image_tokens=config.max_image_tokens or 64,
        )
    raise ValueError(f"Unsupported model name: {config.name}")
