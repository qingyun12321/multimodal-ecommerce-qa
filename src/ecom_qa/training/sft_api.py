from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import fcntl
from openai import OpenAI


TAG_PATTERN = re.compile(
    r"</?(?:Think|Answer|RAG_search|Web_search|Grounding|information)\b[^>]*>",
    re.IGNORECASE,
)


DIRECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["first_think", "final_answer"],
    "properties": {
        "first_think": {"type": "string", "minLength": 1},
        "final_answer": {"type": "string", "minLength": 1},
    },
}


RAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["first_think", "after_tool_think", "next_action", "web_search_query", "final_answer"],
    "properties": {
        "first_think": {"type": "string", "minLength": 1},
        "after_tool_think": {"type": "string", "minLength": 1},
        "next_action": {"type": "string", "enum": ["Answer", "Web_search"]},
        "web_search_query": {"type": "string"},
        "final_answer": {"type": "string"},
    },
}


def web_schema(*, max_items: int, include_first_think: bool = True) -> dict[str, Any]:
    required = ["information_items", "after_tool_think", "final_answer"]
    properties: dict[str, Any] = {
        "information_items": {
            "type": "array",
            "minItems": 1,
            "maxItems": max(1, int(max_items)),
            "items": {"type": "string", "minLength": 1},
        },
        "after_tool_think": {"type": "string", "minLength": 1},
        "final_answer": {"type": "string", "minLength": 1},
    }
    if include_first_think:
        required.insert(0, "first_think")
        properties["first_think"] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


WEB_SCHEMA: dict[str, Any] = web_schema(max_items=5, include_first_think=True)
WEB_AFTER_RAG_SCHEMA: dict[str, Any] = web_schema(max_items=5, include_first_think=False)


class ApiGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class ApiClient:
    repo_dir: Path
    cache_dir: Path
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "xiaomi/mimo-v2.5"
    reasoning_effort: str = "none"
    reasoning_exclude: bool = True
    timeout_seconds: int = 900
    max_retries: int = 2
    temperature: float = 0.1
    top_p: float | None = None
    max_tokens: int = 1024
    use_cache: bool = True
    keep_debug: bool = False
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _schema_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _metadata_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _cache_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _call_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ApiGenerationError("OPENROUTER_API_KEY is required for SFT API generation.")
        self.api_key = api_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index.update(load_cache_index(self.cache_dir / "records.jsonl"))
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=float(self.timeout_seconds))

    def generate_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        cache_key: str,
        image_paths: list[Path],
        use_search: bool = False,
    ) -> dict[str, Any]:
        if self.use_cache:
            with self._cache_lock:
                cached = self._cache_index.get(cache_key)
            if cached and cached.get("status") == "success" and isinstance(cached.get("response"), dict):
                payload = dict(cached["response"])
                validate_no_tags(payload)
                validate_payload_schema(payload, schema)
                self._record_call_metadata(
                    cache_key,
                    {
                        "cache_hit": True,
                        "schema_mode": cached.get("schema_mode") or "unknown",
                        "attempts": int(cached.get("attempts") or 0),
                        "usage": cached.get("usage") or {},
                        "api_seconds": 0.0,
                        "parse_seconds": 0.0,
                        "elapsed_seconds": 0.0,
                        "prompt_tokens": token_value(cached.get("usage"), "prompt_tokens"),
                        "completion_tokens": token_value(cached.get("usage"), "completion_tokens"),
                        "total_tokens": token_value(cached.get("usage"), "total_tokens"),
                    },
                )
                return payload

        schema_path = self._write_schema(schema_name=schema_name, schema=schema)
        request_payload: dict[str, Any] = {
            "cache_key": cache_key,
            "schema_name": schema_name,
            "model": self.model,
            "base_url": self.base_url,
            "reasoning": self.reasoning_config(),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "use_search": use_search,
            "image_paths": [str(path) for path in image_paths],
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "schema_path": str(schema_path),
        }

        last_error = ""
        last_category = ""
        attempts = self.max_retries + 1
        schema_mode = "json_schema"
        started_at = time.perf_counter()
        last_usage: dict[str, Any] = {}
        for attempt in range(1, attempts + 1):
            api_started = time.perf_counter()
            try:
                response = self._create_completion(
                    prompt=prompt,
                    schema_name=schema_name,
                    schema=schema,
                    image_paths=image_paths,
                    schema_mode=schema_mode,
                )
            except Exception as exc:  # noqa: BLE001 - OpenRouter/OpenAI SDK exceptions vary by version
                last_error = repr(exc)
                last_category = "api_error"
                if schema_mode == "json_schema" and is_schema_rejection(last_error):
                    schema_mode = "json_object"
                    last_category = "schema_fallback"
                if self.keep_debug:
                    self._write_debug(
                        cache_key=cache_key,
                        attempt=attempt,
                        prompt=prompt,
                        request_payload={**request_payload, "schema_mode": schema_mode},
                        raw_response="",
                        error=last_error,
                    )
                continue
            api_seconds = time.perf_counter() - api_started
            parse_started = time.perf_counter()
            raw_text = response_message_content(response)
            last_usage = response_usage_dict(response)
            try:
                payload = parse_json_output(raw_text)
                validate_no_tags(payload)
                validate_payload_schema(payload, schema)
            except Exception as exc:  # noqa: BLE001 - preserve parse/schema reason for retry
                last_error = repr(exc)
                last_category = "parse_or_schema_error"
                if self.keep_debug:
                    self._write_debug(
                        cache_key=cache_key,
                        attempt=attempt,
                        prompt=prompt,
                        request_payload={**request_payload, "schema_mode": schema_mode},
                        raw_response=raw_text,
                        error=last_error,
                    )
                continue

            elapsed_seconds = time.perf_counter() - started_at
            metadata = {
                "cache_hit": False,
                "schema_mode": schema_mode,
                "attempts": attempt,
                "usage": last_usage,
                "api_seconds": round(api_seconds, 3),
                "parse_seconds": round(time.perf_counter() - parse_started, 3),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "prompt_tokens": token_value(last_usage, "prompt_tokens"),
                "completion_tokens": token_value(last_usage, "completion_tokens"),
                "total_tokens": token_value(last_usage, "total_tokens"),
                "response_id": str(getattr(response, "id", "") or ""),
                "response_model": str(getattr(response, "model", "") or ""),
            }
            self._record_cache(
                {
                    **request_payload,
                    "status": "success",
                    "attempts": attempt,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "api_seconds": metadata["api_seconds"],
                    "parse_seconds": metadata["parse_seconds"],
                    "schema_mode": schema_mode,
                    "usage": last_usage,
                    "response": payload,
                }
            )
            self._record_call_metadata(cache_key, metadata)
            return payload

        elapsed_seconds = time.perf_counter() - started_at
        failure_row = {
            **request_payload,
            "status": "failure",
            "attempts": attempts,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "schema_mode": schema_mode,
            "usage": last_usage,
            "error": last_error,
            "error_category": last_category,
        }
        self._record_cache(failure_row)
        self._record_call_metadata(
            cache_key,
            {
                "cache_hit": False,
                "schema_mode": schema_mode,
                "attempts": attempts,
                "usage": last_usage,
                "api_seconds": round(elapsed_seconds, 3),
                "parse_seconds": 0.0,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "error_category": last_category,
                "prompt_tokens": token_value(last_usage, "prompt_tokens"),
                "completion_tokens": token_value(last_usage, "completion_tokens"),
                "total_tokens": token_value(last_usage, "total_tokens"),
            },
        )
        raise ApiGenerationError(f"API generation failed for {cache_key}: {last_error}")

    def consume_call_metadata(self, cache_key: str) -> dict[str, Any]:
        with self._metadata_lock:
            return dict(self._call_metadata.pop(cache_key, {}))

    def reasoning_config(self) -> dict[str, Any]:
        return {"effort": self.reasoning_effort, "exclude": self.reasoning_exclude}

    def _create_completion(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        image_paths: list[Path],
        schema_mode: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content(prompt=prompt, image_paths=image_paths)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": response_format(schema_name=schema_name, schema=schema, schema_mode=schema_mode),
            "extra_body": {"reasoning": self.reasoning_config()},
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return self._client.chat.completions.create(**kwargs)

    def _write_schema(self, *, schema_name: str, schema: dict[str, Any]) -> Path:
        schema_path = self.cache_dir / "schemas" / f"{safe_name(schema_name)}.schema.json"
        with self._schema_lock:
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = schema_path.with_suffix(f".{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, schema_path)
        return schema_path

    def _record_cache(self, row: dict[str, Any]) -> None:
        if not self.use_cache:
            return
        row = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
        with self._cache_lock:
            self._cache_index[str(row["cache_key"])] = row
            append_jsonl_locked(self.cache_dir / "records.jsonl", row)

    def _record_call_metadata(self, cache_key: str, metadata: dict[str, Any]) -> None:
        with self._metadata_lock:
            self._call_metadata[cache_key] = metadata

    def _write_debug(
        self,
        *,
        cache_key: str,
        attempt: int,
        prompt: str,
        request_payload: dict[str, Any],
        raw_response: str,
        error: str,
    ) -> None:
        debug_dir = self.cache_dir / "debug" / cache_key
        debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            **request_payload,
            "prompt": prompt,
            "attempt": attempt,
            "raw_response": raw_response[-12000:],
            "error": error,
        }
        (debug_dir / f"api_attempt_{attempt}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def response_format(*, schema_name: str, schema: dict[str, Any], schema_mode: str) -> dict[str, Any]:
    if schema_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": safe_name(schema_name),
            "strict": True,
            "schema": schema,
        },
    }


def message_content(*, prompt: str, image_paths: list[Path]) -> str | list[dict[str, Any]]:
    if not image_paths:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        if image_path.is_file():
            parts.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}})
    return parts


def image_path_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def make_cache_key(*, record_id: str, kind: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{kind}\n{record_id}\n{prompt}".encode("utf-8")).hexdigest()[:20]
    safe_record = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("_") or "record"
    return f"{safe_record}.{kind}.{digest}"


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return safe[:64] or "sft_schema"


def load_cache_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache_key = str(row.get("cache_key") or "")
            if cache_key:
                rows[cache_key] = row
    return rows


def append_jsonl_locked(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def parse_json_output(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("API output must be a JSON object.")
    return payload


def validate_no_tags(payload: Any) -> None:
    if isinstance(payload, str):
        if TAG_PATTERN.search(payload):
            raise ValueError(f"API field contains reserved SFT tag: {payload[:80]!r}")
        return
    if isinstance(payload, list):
        for item in payload:
            validate_no_tags(item)
        return
    if isinstance(payload, dict):
        for value in payload.values():
            validate_no_tags(value)


def validate_payload_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    validate_schema_value(payload, schema, "$")


def validate_schema_value(value: Any, schema: dict[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be object")
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f"{path} has extra properties: {extras}")
        for key, property_schema in properties.items():
            if key in value:
                validate_schema_value(value[key], property_schema, f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(f"{path} needs at least {min_items} items")
        if isinstance(max_items, int) and len(value) > max_items:
            raise ValueError(f"{path} allows at most {max_items} items")
        item_schema = schema.get("items") or {}
        for index, item in enumerate(value):
            validate_schema_value(item, item_schema, f"{path}[{index}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be string")
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value.strip()) < min_length:
            raise ValueError(f"{path} must have length >= {min_length}")
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            raise ValueError(f"{path} must be one of {enum}")


def is_schema_rejection(message: str) -> bool:
    lowered = message.lower()
    return "response_format" in lowered and any(token in lowered for token in ("json_schema", "schema", "strict"))


def response_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("API response has no choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or getattr(item, "content", "") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def response_usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if isinstance(usage, dict):
        return dict(usage)
    return {key: getattr(usage, key) for key in dir(usage) if key.endswith("_tokens") and getattr(usage, key) is not None}


def token_value(usage: Any, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
