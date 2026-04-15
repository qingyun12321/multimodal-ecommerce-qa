from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


SERPAPI_BASE_URL = "https://serpapi.com/search"
JINA_READER_PREFIX = "https://r.jina.ai/"
PDF_SUMMARY_SYSTEM = (
    "You are a helpful assistant. Your task is to summarize the main content of the given web "
    "page in no more than five sentences. Your summary should cover the overall key points of "
    "the page, not just parts related to the user's question."
)
PDF_SUMMARY_PROMPT = (
    "If any part of the content is helpful for answering the user's question, be sure to include it clearly "
    "in the summary. Do not ignore relevant information, but also make sure the general structure and main ideas "
    "of the page are preserved. Your summary should be concise, factual, and informative.\n\n"
    "Webpage Content (first 30000 characters) is:\n{webpage_content}\n\n"
    "Question: {question}"
)


def ensure_workspace_env() -> dict[str, str]:
    root = Path.cwd()
    hf_home = root / "artifacts" / "hf_home"
    env_updates = {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    return env_updates


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sequential_repo_download(repo_id: str) -> Path:
    target_dir = Path.cwd() / "artifacts" / "hf_home" / "sequential_repos" / repo_id.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for filename in api.list_repo_files(repo_id):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
        )
    return target_dir


@dataclass(slots=True)
class SerpAPISearchConfig:
    api_key: str | None
    hl: str = "zh-cn"
    gl: str = "cn"
    location: str = "Beijing, China"
    safe: str = "active"
    output: str = "json"


class SerpAPIClient:
    def __init__(self, config: SerpAPISearchConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is missing. The SerpAPI code path is implemented, but live search "
                "cannot run without an API key."
            )
        payload = {
            "api_key": self.config.api_key,
            "hl": self.config.hl,
            "gl": self.config.gl,
            "location": self.config.location,
            "output": self.config.output,
            **params,
        }
        response = self.session.get(SERPAPI_BASE_URL, params=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def text_search(
        self,
        *,
        query: str,
        start: int = 0,
        num: int = 10,
    ) -> dict[str, Any]:
        return self._request(
            {
                "engine": "google",
                "q": query,
                "start": start,
                "num": num,
            }
        )

    def image_search(
        self,
        *,
        query: str,
        ijn: int = 0,
    ) -> dict[str, Any]:
        return self._request(
            {
                "engine": "google_images",
                "q": query,
                "ijn": ijn,
                "safe": self.config.safe,
            }
        )


class JinaReaderClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.session = requests.Session()

    def read(
        self,
        *,
        url: str,
        instruction: str | None = None,
        respond_with: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if instruction:
            headers["x-instruction"] = instruction
        if respond_with:
            headers["x-respond-with"] = respond_with
        if json_schema:
            headers["x-json-schema"] = json.dumps(json_schema, ensure_ascii=False)

        response = self.session.get(f"{JINA_READER_PREFIX}{url}", headers=headers, timeout=120)
        response.raise_for_status()
        return {
            "url": url,
            "reader_url": f"{JINA_READER_PREFIX}{url}",
            "content": response.text,
            "content_length": len(response.text),
        }


class Qwen3_5TextGenerator:
    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> None:
        ensure_workspace_env()
        self.model_id = model_id
        self.device = device
        local_dir = sequential_repo_download(model_id) if "9B" in model_id else model_id
        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        if "9B" in model_id and device.startswith("cuda"):
            self.input_device = torch.device("cuda:0")
            load_kwargs = {
                "device_map": "cuda",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, torch_dtype),
                ),
            }
            dtype = getattr(torch, torch_dtype)
        elif device.startswith("cuda"):
            self.input_device = torch.device("cuda:0")
            load_kwargs = {
                "device_map": "auto",
                "max_memory": {0: "26GiB", "cpu": "128GiB"},
                "offload_folder": str(Path.cwd() / "artifacts" / "offload" / model_id.replace("/", "__")),
            }
            dtype = getattr(torch, torch_dtype)
        else:
            self.input_device = torch.device("cpu")
            load_kwargs = {"device_map": "cpu"}
            dtype = torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            local_dir,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        self.model.eval()

    def _generate(
        self,
        *,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 256,
    ) -> str:
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.input_device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def summarize_webpage(self, *, webpage_content: str, question: str) -> str:
        prompt = PDF_SUMMARY_PROMPT.format(
            webpage_content=webpage_content[:30000],
            question=question,
        )
        return self._generate(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": PDF_SUMMARY_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            max_new_tokens=256,
        )

    def image_to_search_query(self, *, image_path: Path) -> str:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {
                        "type": "text",
                        "text": (
                            "Describe this image as a short web search query for finding the same product or very similar products. "
                            "Return one concise line only."
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[prompt],
            images=[rgb],
            return_tensors="pt",
        ).to(self.input_device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def cleanup(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
