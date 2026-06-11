from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import fcntl


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


class CodexGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class CodexClient:
    repo_dir: Path
    cache_dir: Path
    codex_bin: str = "codex"
    model: str = "gpt-5.3-codex-spark"
    reasoning_effort: str = "low"
    timeout_seconds: int = 900
    max_retries: int = 2
    use_cache: bool = True
    keep_debug: bool = False
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _schema_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _cache_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index.update(load_cache_index(self.cache_dir / "records.jsonl"))

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
                return payload

        schema_path = self._write_schema(schema_name=schema_name, schema=schema)
        request_payload: dict[str, Any] = {
            "cache_key": cache_key,
            "schema_name": schema_name,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "use_search": use_search,
            "image_paths": [str(path) for path in image_paths],
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

        last_error = ""
        attempts = self.max_retries + 1
        started_at = time.perf_counter()
        for attempt in range(1, attempts + 1):
            with tempfile.TemporaryDirectory(prefix=f"{cache_key}.", dir=self._tmp_dir()) as tmp:
                tmp_dir = Path(tmp)
                debug_dir = self.cache_dir / "debug" / cache_key if self.keep_debug else tmp_dir
                debug_dir.mkdir(parents=True, exist_ok=True)
                output_path = debug_dir / f"last_message_attempt_{attempt}.json"
                command = self._command(
                    schema_path=schema_path,
                    output_path=output_path,
                    image_paths=image_paths,
                    use_search=use_search,
                )
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        cwd=self.repo_dir,
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    last_error = f"codex timed out after {self.timeout_seconds}s"
                    if self.keep_debug:
                        self._write_debug(
                            cache_key=cache_key,
                            attempt=attempt,
                            command=command,
                            prompt=prompt,
                            request_payload=request_payload,
                            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                            output_path=output_path,
                            error=last_error,
                        )
                    continue
                if self.keep_debug:
                    self._write_debug(
                        cache_key=cache_key,
                        attempt=attempt,
                        command=command,
                        prompt=prompt,
                        request_payload=request_payload,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        output_path=output_path,
                        returncode=completed.returncode,
                    )
                if completed.returncode != 0:
                    last_error = f"codex exited {completed.returncode}: {completed.stderr[-1000:]}"
                    continue
                try:
                    payload = parse_json_output(output_path.read_text(encoding="utf-8"))
                    validate_no_tags(payload)
                except Exception as exc:  # noqa: BLE001 - preserve the parse reason for retries
                    last_error = repr(exc)
                    continue
                self._record_cache(
                    {
                        **request_payload,
                        "status": "success",
                        "attempts": attempt,
                        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                        "response": payload,
                    }
                )
                return payload

        self._record_cache(
            {
                **request_payload,
                "status": "failure",
                "attempts": attempts,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "error": last_error,
            }
        )
        raise CodexGenerationError(f"Codex generation failed for {cache_key}: {last_error}")

    def _tmp_dir(self) -> str:
        tmp_dir = self.cache_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return str(tmp_dir)

    def _write_schema(self, *, schema_name: str, schema: dict[str, Any]) -> Path:
        schema_path = self.cache_dir / "schemas" / f"{schema_name}.schema.json"
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

    def _write_debug(
        self,
        *,
        cache_key: str,
        attempt: int,
        command: list[str],
        prompt: str,
        request_payload: dict[str, Any],
        stdout: str,
        stderr: str,
        output_path: Path,
        returncode: int | None = None,
        error: str = "",
    ) -> None:
        debug_dir = self.cache_dir / "debug" / cache_key
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "request.json").write_text(
            json.dumps({**request_payload, "prompt": prompt}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        run_log = {
            "attempt": attempt,
            "command": command,
            "returncode": returncode,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "output_path": str(output_path),
            "error": error,
        }
        (debug_dir / f"codex_run_attempt_{attempt}.json").write_text(
            json.dumps(run_log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _command(
        self,
        *,
        schema_path: Path,
        output_path: Path,
        image_paths: list[Path],
        use_search: bool,
    ) -> list[str]:
        command = [self.codex_bin]
        if use_search:
            command.append("--search")
        command.extend(
            [
                "exec",
                "-m",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-C",
                str(self.repo_dir),
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
            ]
        )
        for image_path in image_paths:
            command.extend(["-i", str(image_path)])
        command.append("-")
        return command


def make_cache_key(*, record_id: str, kind: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{kind}\n{record_id}\n{prompt}".encode("utf-8")).hexdigest()[:20]
    safe_record = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("_") or "record"
    return f"{safe_record}.{kind}.{digest}"


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
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Codex output must be a JSON object.")
    return payload


def validate_no_tags(payload: Any) -> None:
    if isinstance(payload, str):
        if TAG_PATTERN.search(payload):
            raise ValueError(f"Codex field contains reserved SFT tag: {payload[:80]!r}")
        return
    if isinstance(payload, list):
        for item in payload:
            validate_no_tags(item)
        return
    if isinstance(payload, dict):
        for value in payload.values():
            validate_no_tags(value)
