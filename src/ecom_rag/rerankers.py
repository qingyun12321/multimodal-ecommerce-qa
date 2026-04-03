from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ecom_rag.data import ProductRecord, build_query_text
from ecom_rag.model_retrievers import choose_torch_dtype


def _chunked(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass(slots=True)
class RankedDocument:
    index: int
    product_id: str
    image_path: str
    document: str
    score: float


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


class BGEReranker:
    def __init__(self, *, device: str, torch_dtype: str = "float16") -> None:
        self.model_id = "BAAI/bge-reranker-v2-m3"
        local_dir = snapshot_download(self.model_id)
        dtype = choose_torch_dtype(device, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            torch_dtype=dtype,
        )
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def rank(
        self,
        *,
        query: str,
        documents: list[tuple[int, ProductRecord]],
        batch_size: int = 8,
    ) -> list[RankedDocument]:
        pairs = [[query, build_query_text(product, mode="compact_rerank_zh")] for _, product in documents]
        scores: list[float] = []
        for batch in _chunked(pairs, batch_size):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits.view(-1).float()
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            scores.extend(float(score) for score in probs)

        ranked: list[RankedDocument] = []
        for (index, product), score in zip(documents, scores, strict=True):
            ranked.append(
                RankedDocument(
                    index=index,
                    product_id=product.id,
                    image_path=product.image_path,
                    document=build_query_text(product, mode="compact_rerank_zh"),
                    score=score,
                )
            )
        ranked.sort(key=lambda row: row.score, reverse=True)
        return ranked

    def cleanup(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


class QwenFilter:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        local_dir = sequential_repo_download(model_id) if "9B" in model_id else snapshot_download(model_id)
        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        offload_dir = Path.cwd() / "artifacts" / "offload" / model_id.replace("/", "__")
        offload_dir.mkdir(parents=True, exist_ok=True)
        if "9B" in model_id and device.startswith("cuda"):
            load_kwargs = {
                "device_map": "cuda",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, torch_dtype),
                ),
            }
            self.input_device = torch.device("cuda:0")
        elif device.startswith("cuda"):
            load_kwargs = {
                "device_map": "auto",
                "max_memory": {0: "26GiB", "cpu": "128GiB"},
                "offload_folder": str(offload_dir),
            }
            self.input_device = torch.device("cuda:0")
        else:
            load_kwargs = {"device_map": "cpu"}
            self.input_device = torch.device("cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            local_dir,
            dtype=getattr(torch, torch_dtype) if device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        self.model.eval()

    def _build_prompt(
        self,
        *,
        query: str,
        candidates: list[tuple[int, ProductRecord]],
    ) -> str:
        doc_blocks = []
        for rank, (_, product) in enumerate(candidates, start=1):
            doc_blocks.append(
                "\n".join(
                    [
                        f"DocID: {rank}",
                        build_query_text(product, mode="compact_rerank_zh"),
                    ]
                )
            )
        return "\n\n".join(
            [
                "You are a careful ecommerce retrieval judge.",
                "Select the 5 most relevant products for the query.",
                "Prefer products that truly belong to the requested subcategory and clearly filter out near-miss or unrelated items.",
                "Return strict JSON only.",
                'JSON schema: {"ranked_doc_ids":[int,int,int,int,int],"filtered_doc_ids":[int,...]}',
                "Do not output explanations or markdown.",
                f"User query:\n{query}",
                "Candidate documents:",
                "\n\n".join(doc_blocks),
            ]
        )

    def _decode_json(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))

        ranked_match = re.search(r'"ranked_doc_ids"\s*:\s*\[([^\]]+)\]', text)
        filtered_match = re.search(r'"filtered_doc_ids"\s*:\s*\[([^\]]+)\]', text)
        if ranked_match and filtered_match:
            def parse_ints(raw: str) -> list[int]:
                return [int(token) for token in re.findall(r"\d+", raw)]

            return {
                "ranked_doc_ids": parse_ints(ranked_match.group(1)),
                "filtered_doc_ids": parse_ints(filtered_match.group(1)),
            }
        raise ValueError(f"No JSON block found in generation: {text!r}")

    @torch.inference_mode()
    def rank(
        self,
        *,
        query: str,
        candidates: list[tuple[int, ProductRecord]],
        max_new_tokens: int = 256,
    ) -> tuple[list[RankedDocument], dict]:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(query=query, candidates=candidates)}]}
        ]
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
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        payload = self._decode_json(text)

        by_doc_id = {
            row_id: RankedDocument(
                index=index,
                product_id=product.id,
                image_path=product.image_path,
                document=build_query_text(product, mode="compact_rerank_zh"),
                score=float(6 - row_id),
            )
            for row_id, (index, product) in enumerate(candidates, start=1)
        }
        ranked = [by_doc_id[int(doc_id)] for doc_id in payload["ranked_doc_ids"] if int(doc_id) in by_doc_id]
        return ranked, payload

    def cleanup(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
