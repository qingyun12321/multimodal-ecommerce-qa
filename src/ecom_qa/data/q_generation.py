from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from random import Random
from typing import Any

from PIL import Image

from ecom_qa.data.catalog import ProductRecord


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
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5


@dataclass(frozen=True, slots=True)
class PreviewRequest:
    qa_type: str
    product: ProductRecord
    image_path: Path
    source_row_index: int


@dataclass(frozen=True, slots=True)
class ServerConfig:
    server_binary: Path = SERVER_BINARY
    model_path: Path = MODEL_PATH
    mmproj_path: Path = MMPROJ_PATH
    host: str = HOST
    port: int = PORT
    ctx_size: int = CTX_SIZE
    threads: int = THREADS

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def build_preview_requests(
    products: list[ProductRecord],
    *,
    dataset_dir: Path,
    multimodal_count: int,
    text_count: int,
    seed: int,
) -> list[PreviewRequest]:
    total = multimodal_count + text_count
    if total <= 0:
        raise ValueError("Preview count must be greater than 0.")
    if total > len(products):
        raise ValueError("Preview count exceeds product count.")

    chosen = Random(seed).sample(range(len(products)), total)
    requests: list[PreviewRequest] = []
    for order, row_index in enumerate(chosen):
        product = products[row_index]
        qa_type = "multimodal" if order < multimodal_count else "text_only"
        requests.append(
            PreviewRequest(
                qa_type=qa_type,
                product=product,
                image_path=dataset_dir / product.image_path,
                source_row_index=row_index,
            )
        )
    return requests


def build_system_prompt(qa_type: str) -> str:
    prompt = (
        "你要根据给定的商品信息，必要时结合图片，生成 1 条中文 QA 样本。"
        "只输出一个 JSON 对象，字段只能是 query 和 answer。"
        "问题要像真实用户会问的。"
        "只能生成两类问题："
        "第一类是开放性问题，也就是通常需要商品背景知识、品牌知识、产地来源、发展历史或其他外部知识才能回答的问题；"
        "第二类是观察型问题，也就是可以通过观察图片直接得到，或主要通过观察图片得到的问题。"
        "这两类都不要理解得过窄，给出的例子只是示例，不是限制。"
        "不要生成那些答案可以直接从商品标题、商品参数、价格、颜色、规格、尺码、评分、售后、商品描述等输入字段中直接抄出的传统电商属性问题。"
        "开放性问题优先选择客观事实型问法，不要优先生成通常、一般、适合、表达什么情感、设计灵感来自哪里这类容易变成主观概括、文化联想或营销话术的问题。"
        "答案要简洁直接，但必须是自然、完整的中文句子。"
        "不要使用像从图片来看、从图片细节来看、图中显示这类说明式措辞。"
        "不要在问题里出现商品ID、一级类目、二级类目、source_row_index 等内部字段名或其取值。"
        "如果问题需要外部知识，answer 也照常给出一个自然回答，不要求一定能被输入严格验证。"
    )
    if qa_type == "text_only":
        return (
            prompt
            + "这是纯文本输入。"
            + "问题必须把商品指清楚。"
            + "不要使用这个、这款、这件、这台、这种、它之类脱离上下文的指代词，即使后面跟了商品名也不要用。"
            + "也不要写某品牌的这款某商品、某品牌的这件某商品这类表达。"
            + "请直接使用自然的商品名称、品牌名、品类名或它们的组合来指代商品。"
            + "纯文本输入时，只能生成开放性问题。"
        )
    return (
        prompt
        + "这是图文输入。"
        + "只有在图文输入时，问题才可以使用这个商品、这款、这台、这个、这件之类的自然指代。"
        + "图文输入时，既可以生成观察型问题，也可以生成以图片中商品为对象的开放性问题。"
        + "图片主要用于帮助理解用户指的是哪个商品，不必强行把问题限制成只能询问视觉细节。"
        + "不要生成主观推断题，比如适合什么家居环境、给人什么感觉、适合什么气质。"
    )


def build_user_content(request: PreviewRequest) -> list[dict[str, Any]] | str:
    product = request.product
    product_block = "\n".join(
        [
            f"商品ID: {product.id}",
            f"一级类目: {product.category}",
            f"二级类目: {product.subcategory}",
            f"商品标题: {product.title}",
            f"品牌: {product.brand}",
            f"价格: {product.price:.2f} 元",
            f"颜色: {'、'.join(product.colors) if product.colors else '未知'}",
            f"尺码/规格: {'、'.join(product.sizes) if product.sizes else '未知'}",
            f"评分: {product.rating}",
            f"店铺: {product.shop_name}",
            f"售后: {json.dumps(product.after_sales, ensure_ascii=False)}",
            f"商品参数: {json.dumps(product.parameters, ensure_ascii=False)}",
            f"商品描述: {product.description}",
        ]
    )
    instruction_lines = [
        "请生成 1 条中文 QA。\n"
        "要求：\n"
        "1. 只生成 1 个问题和 1 个答案。\n"
        "2. 只能生成开放性问题，或观察型问题。\n"
        "3. 开放性问题是指通常需要背景知识、品牌知识、产地来源、发展历史或其他外部知识才能回答的问题。\n"
        "4. 观察型问题是指可以通过观察图片直接得到，或主要通过观察图片得到的问题；范围不要局限于询问商品名称。\n"
        "5. 上面两类问题都不要理解得过窄，示例只是示例，不是限制。\n"
        "6. 可以把这个商品是什么、这是什么类型的商品当作观察型问题的例子，但不要只围绕这类问法生成；凡是通过图片观察商品本身可以得到的信息，都可以形成观察型问题。\n"
        "7. 开放性问题也不要只局限于是谁发明的、产自哪里这几个问法，这些也只是例子。\n"
        "8. 开放性问题优先写成客观事实型问法，不要优先生成通常、一般、适合什么场合、表达什么情感、设计灵感来自哪里这类主观、泛化或营销化的问法。\n"
        "9. 不要生成传统电商属性问题，尤其不要生成那些答案可以直接从商品标题、商品参数、价格、颜色、规格、尺码、评分、售后、商品描述中直接抄出的题目。\n"
        "10. 输出必须是合法 JSON。\n"
        "11. 问题和答案都要自然，不要写成标注说明或数据字段复述。\n"
        "12. 答案必须写成完整句子，不能只回答一个短语、数字或词语。\n"
        "13. 如果问题需要外部知识，answer 也正常填写为自然句子，不要求一定能被输入严格验证。\n"
    ]
    if request.qa_type == "text_only":
        instruction_lines.append(
            "14. 这是纯文本输入，问题必须明确指代商品对象，不能只写这个、这款、这件、这台、它，也不要出现商品ID之类内部标识。\n"
        )
        instruction_lines.append(
            "15. 即使后面跟了商品名，也不要写这款男士衬衫、这个手机、某品牌的这款包这类说法；应直接写成男士衬衫、某品牌手机、某品牌手提包这类自然表达。\n"
        )
        instruction_lines.append(
            "16. 纯文本输入时，只能生成开放性问题，不要生成观察型问题，也不要生成传统电商属性问题。\n"
        )
    else:
        instruction_lines.append(
            "14. 这是图文输入，问题可以使用这个商品、这款、这台、这个、这件之类的自然指代。\n"
        )
        instruction_lines.append(
            "15. 图文输入时，优先生成观察型问题，或以图片中的商品为对象的开放性问题，不要生成传统电商属性问题。\n"
        )
        instruction_lines.append(
            "16. 观察型问题可以询问图片中直接可见或主要可见的信息，例如商品的外观特征、结构组成、设计元素、可见部件、图案样式、文字标识、包装形式、界面布局等，但不要把范围收得过窄，也不要只重复少数固定模板。\n"
        )
        instruction_lines.append(
            "17. 如果一个问题的答案主要是从标题、参数或描述里直接抄出来，而看不看图片差别不大，就不要生成成图文问题。\n"
        )
        instruction_lines.append(
            "18. 问题不要写成从图片来看、从图片细节来看这类措辞；直接像用户提问一样表达即可。\n"
        )
        instruction_lines.append(
            "19. 不要生成主观联想或场景延伸类问题，例如适合什么风格、给人什么感觉、适合什么家居环境，除非输入信息里明确给出。\n"
        )
    instruction = "".join(instruction_lines) + "\n" + product_block
    if request.qa_type == "text_only":
        return instruction
    return [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(request.image_path)}},
    ]


def image_path_to_data_url(image_path: Path) -> str:
    try:
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


def qa_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 4},
                    "answer": {"type": "string", "minLength": 4},
                },
                "required": ["query", "answer"],
                "additionalProperties": False,
            }
        },
    }


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


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
        args = [
            str(config.server_binary),
            "-m",
            str(config.model_path),
            "--mmproj",
            str(config.mmproj_path),
            "--gpu-layers",
            GPU_LAYERS,
            "--flash-attn",
            FLASH_ATTN,
            "--threads",
            str(config.threads),
            "--reasoning",
            REASONING,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--ctx-size",
            str(config.ctx_size),
            "--alias",
            MODEL_ALIAS,
        ]
        process = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
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


def generate_one_qa(request: PreviewRequest, config: ServerConfig) -> tuple[dict[str, str], float]:
    payload = {
        "model": MODEL_ALIAS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "presence_penalty": PRESENCE_PENALTY,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": qa_response_format(),
        "messages": [
            {"role": "system", "content": build_system_prompt(request.qa_type)},
            {"role": "user", "content": build_user_content(request)},
        ],
    }
    started_at = time.perf_counter()
    response = post_json(f"{config.base_url}/v1/chat/completions", payload)
    elapsed_seconds = time.perf_counter() - started_at
    message = response["choices"][0]["message"]["content"]
    qa = extract_json_object(message)
    return {
        "query": str(qa["query"]).strip(),
        "answer": str(qa["answer"]).strip(),
    }, elapsed_seconds


def build_result_record(request: PreviewRequest, qa: dict[str, str], elapsed_seconds: float) -> dict[str, Any]:
    return {
        "query": qa["query"],
        "answer": qa["answer"],
        "qa_type": request.qa_type,
        "source_row_index": request.source_row_index,
        "product_id": request.product.id,
        "image_path": request.product.image_path,
        "source_title": request.product.title,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def build_run_summary(
    records: list[dict[str, Any]],
    *,
    multimodal_count: int,
    text_count: int,
    startup_seconds: float,
    total_wall_seconds: float,
) -> dict[str, Any]:
    generation_seconds_total = sum(float(record["elapsed_seconds"]) for record in records)
    return {
        "requested_counts": {"multimodal": multimodal_count, "text_only": text_count},
        "generated_count": len(records),
        "server_startup_seconds": round(startup_seconds, 3),
        "generation_seconds_total": round(generation_seconds_total, 3),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "average_elapsed_seconds": round(generation_seconds_total / len(records), 3) if records else 0.0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
