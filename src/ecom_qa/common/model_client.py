from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


SERVER_BINARY = Path("/home/qingyun/llama.cpp/build/bin/llama-server")
MODEL_PATH = Path("/home/qingyun/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
MMPROJ_PATH = Path("/home/qingyun/models/unsloth/Qwen3.5-9B-GGUF/mmproj-F16.gguf")
MODEL_ALIAS = "qwen35-preview"
HOST = "127.0.0.1"
PORT = 8012
CTX_SIZE = 16384
THREADS = 8
GPU_LAYERS = "all"
FLASH_ATTN = "on"
REASONING = "off"


@dataclass(frozen=True, slots=True)
class ServerConfig:
    server_binary: Path = SERVER_BINARY
    model_path: Path = MODEL_PATH
    mmproj_path: Path = MMPROJ_PATH
    host: str = HOST
    port: int = PORT
    ctx_size: int = CTX_SIZE
    threads: int = THREADS
    parallel: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_layers: str = GPU_LAYERS
    flash_attn: str = FLASH_ATTN
    reasoning: str = REASONING
    alias: str = MODEL_ALIAS

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def image_path_to_data_url(image_path: Path) -> str:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 300.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def wait_for_server(base_url: str, *, timeout_seconds: float = 300.0) -> float:
    start = time.perf_counter()
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5.0) as response:
                if response.status == 200:
                    return time.perf_counter() - start
        except Exception:
            pass
        if time.perf_counter() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for llama.cpp server at {base_url}.")
        time.sleep(1.0)


@contextmanager
def run_llama_server(config: ServerConfig, *, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        env = os.environ.copy()
        lib_dir = str(config.server_binary.parent)
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir if not current_ld_path else f"{lib_dir}:{current_ld_path}"
        args = [
            str(config.server_binary),
            "-m",
            str(config.model_path),
            "--mmproj",
            str(config.mmproj_path),
            "--gpu-layers",
            config.gpu_layers,
            "--flash-attn",
            config.flash_attn,
            "--threads",
            str(config.threads),
            "--reasoning",
            config.reasoning,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--ctx-size",
            str(config.ctx_size),
        ]
        if config.parallel is not None:
            args.extend(["--parallel", str(config.parallel)])
        if config.batch_size is not None:
            args.extend(["--batch-size", str(config.batch_size)])
        if config.ubatch_size is not None:
            args.extend(["--ubatch-size", str(config.ubatch_size)])
        args.extend(["--alias", config.alias])

        process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        try:
            startup_seconds = wait_for_server(config.base_url)
            yield startup_seconds
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
