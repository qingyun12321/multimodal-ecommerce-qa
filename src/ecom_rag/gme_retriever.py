from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Iterable

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel


DEFAULT_T2I_PROMPT = "Find an image that matches the given text."


class GMERetriever:
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str | None = None,
        torch_dtype: str = "float16",
        min_image_tokens: int = 64,
        max_image_tokens: int = 256,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = getattr(torch, torch_dtype)
        min_image_tokens = min(min_image_tokens, max_image_tokens)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        local_model_dir = snapshot_download(repo_id=model_name)
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "min_image_tokens": min_image_tokens,
            "max_image_tokens": max_image_tokens,
        }
        if device.startswith("cuda"):
            load_kwargs["device_map"] = device
        self.model = AutoModel.from_pretrained(local_model_dir, **load_kwargs)
        if not device.startswith("cuda"):
            self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Iterable[str],
        instruction: str = DEFAULT_T2I_PROMPT,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        embeddings = self.model.get_text_embeddings(
            texts=list(texts),
            instruction=instruction,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return self._to_numpy(embeddings)

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: Iterable[Path],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        embeddings = self.model.get_image_embeddings(
            images=[str(path) for path in image_paths],
            is_query=False,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return self._to_numpy(embeddings)

    def _to_numpy(self, embeddings: torch.Tensor) -> np.ndarray:
        normalized = F.normalize(embeddings.float(), p=2, dim=-1)
        array = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return array

    def cleanup(self) -> None:
        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
