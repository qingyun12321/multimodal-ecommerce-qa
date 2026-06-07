from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecom_qa.common.model_client import (
    HOST,
    SERVER_BINARY,
    ServerConfig,
    image_path_to_data_url,
    post_json,
)
from ecom_qa.tool_calls.generation import (
    build_first_round_prompt,
    parse_tool_call_output,
)


DEFAULT_PORT = 8014
DEFAULT_CTX_SIZE = 16384
DEFAULT_THREADS = 8
DEFAULT_PARALLEL = 2
DEFAULT_BATCH_SIZE = 1024
DEFAULT_UBATCH_SIZE = 256
DEFAULT_MAX_TOKENS = 1024

TEMPERATURE = 0.2
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 0.2


@dataclass(frozen=True, slots=True)
class ToolCallModelSpec:
    key: str
    alias: str
    model_path: Path
    mmproj_path: Path


MODEL_SPECS: dict[str, ToolCallModelSpec] = {
    "qwen3-vl-4b": ToolCallModelSpec(
        key="qwen3-vl-4b",
        alias="qwen3-vl-4b-instruct",
        model_path=Path(
            "/home/qingyun/models/unsloth/Qwen3-VL-4B-Instruct-GGUF/"
            "Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf"
        ),
        mmproj_path=Path("/home/qingyun/models/unsloth/Qwen3-VL-4B-Instruct-GGUF/mmproj-F16.gguf"),
    ),
    "qwen3-vl-8b": ToolCallModelSpec(
        key="qwen3-vl-8b",
        alias="qwen3-vl-8b-instruct",
        model_path=Path(
            "/home/qingyun/models/unsloth/Qwen3-VL-8B-Instruct-GGUF/"
            "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
        ),
        mmproj_path=Path("/home/qingyun/models/unsloth/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf"),
    ),
}


def make_server_config(
    *,
    model_key: str,
    server_binary: Path = SERVER_BINARY,
    host: str = HOST,
    port: int = DEFAULT_PORT,
    threads: int = DEFAULT_THREADS,
    ctx_size: int = DEFAULT_CTX_SIZE,
    parallel: int = DEFAULT_PARALLEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ubatch_size: int = DEFAULT_UBATCH_SIZE,
) -> ServerConfig:
    spec = MODEL_SPECS[model_key]
    return ServerConfig(
        server_binary=server_binary,
        model_path=spec.model_path,
        mmproj_path=spec.mmproj_path,
        host=host,
        port=port,
        ctx_size=ctx_size,
        threads=threads,
        parallel=parallel,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        alias=spec.alias,
    )


def qa_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    value = str(metadata.get("qa_type") or "").strip().lower()
    if value in {"text_only", "multimodal"}:
        return value
    return "multimodal"


def has_image_input(record: dict[str, Any]) -> bool:
    return qa_type(record) != "text_only"


def build_system_prompt(record: dict[str, Any]) -> str:
    return build_first_round_prompt(
        has_image_input=has_image_input(record),
        domain=str(record["domain"]),
    )


def build_user_content(record: dict[str, Any]) -> list[dict[str, Any]] | str:
    query = str(record["query"]).strip()
    if not has_image_input(record):
        return f"这里是问题，没有图片输入。\n问题：{query}"
    return [
        {"type": "text", "text": f"这里是图片和问题：<image>\n问题：{query}"},
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(Path(record["image_path"]))}},
    ]


def infer_tool_call(
    record: dict[str, Any],
    config: ServerConfig,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": config.alias,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "presence_penalty": PRESENCE_PENALTY,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": build_system_prompt(record)},
            {"role": "user", "content": build_user_content(record)},
        ],
    }
    started_at = time.perf_counter()
    response = post_json(f"{config.base_url}/v1/chat/completions", payload, timeout=600.0)
    elapsed_seconds = time.perf_counter() - started_at
    raw_output = str(response["choices"][0]["message"]["content"]).strip()
    return parse_tool_call_output(raw_output), elapsed_seconds
