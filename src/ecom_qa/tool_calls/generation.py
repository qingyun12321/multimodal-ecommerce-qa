from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Iterable

from ecom_qa.common.paths import data_path
from ecom_qa.common.model_client import (
    ServerConfig,
    image_path_to_data_url,
    post_json,
    run_llama_server,
    write_json,
)


UNIFIED_VQA_DIR = data_path("unified_vqa")
DEFAULT_SOURCE_COUNTS = {
    "ecom_qa_pairs": 2500,
    "ecom_qa_pairs_open_question": 2500,
    "ecom_qa_pairs_supplement": 2000,
    "infoseek_sample": 8000,
}
TOOL_CALL_CTX_SIZE = 16384
TOOL_CALL_PARALLEL = 2
TAG_NAMES = ("Think", "Answer", "RAG_search", "Web_search", "Grounding")
TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in TAG_NAMES
}
NO_GROUNDING_VALUES = {"", "no", "none", "false", "不需要", "无需", "否", "不用"}


@dataclass(frozen=True, slots=True)
class ToolCallSource:
    name: str
    path: Path
    domain: str
    language: str


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    source_name: str
    row_index: int
    row: dict[str, Any]

    @property
    def query(self) -> str:
        return str(self.row["question"]).strip()

    @property
    def image_path(self) -> Path:
        return Path(str(self.row["image_path"]))

    @property
    def image_rel_path(self) -> str:
        return str(self.row["image_rel_path"])

    @property
    def record_id(self) -> str:
        return str(self.row["record_id"])

    @property
    def domain(self) -> str:
        return str(self.row["domain"])

    @property
    def language(self) -> str:
        return str(self.row["language"])

    @property
    def qa_type(self) -> str:
        metadata = self.row.get("metadata") or {}
        value = str(metadata.get("qa_type") or "").strip().lower()
        if value in {"text_only", "multimodal"}:
            return value
        return "multimodal"

    @property
    def has_image_input(self) -> bool:
        return self.qa_type != "text_only"


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    per_source: dict[str, int]
    total: int


SOURCES: dict[str, ToolCallSource] = {
    "ecom_qa_pairs": ToolCallSource(
        name="ecom_qa_pairs",
        path=UNIFIED_VQA_DIR / "ecom_qa_pairs.jsonl",
        domain="in_domain",
        language="zh",
    ),
    "ecom_qa_pairs_open_question": ToolCallSource(
        name="ecom_qa_pairs_open_question",
        path=UNIFIED_VQA_DIR / "ecom_qa_pairs_open_question.jsonl",
        domain="in_domain",
        language="zh",
    ),
    "ecom_qa_pairs_supplement": ToolCallSource(
        name="ecom_qa_pairs_supplement",
        path=UNIFIED_VQA_DIR / "ecom_qa_pairs_supplement.jsonl",
        domain="in_domain",
        language="zh",
    ),
    "infoseek_sample": ToolCallSource(
        name="infoseek_sample",
        path=UNIFIED_VQA_DIR / "infoseek_sample.jsonl",
        domain="out_of_domain",
        language="en",
    ),
}


def default_tool_call_server_config() -> ServerConfig:
    return ServerConfig(ctx_size=TOOL_CALL_CTX_SIZE, parallel=TOOL_CALL_PARALLEL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_rows(source_names: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    return {name: load_jsonl(SOURCES[name].path) for name in source_names}


def compute_balanced_counts(total: int, source_names: list[str]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("Total sample count must be greater than 0.")
    base = total // len(source_names)
    remainder = total % len(source_names)
    return {
        source_name: base + (1 if index < remainder else 0)
        for index, source_name in enumerate(source_names)
    }


def make_sampling_plan(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    count: int | None,
    per_source_limit: int | None,
    source_counts: dict[str, int] | None = None,
    seed: int,
) -> SamplingPlan:
    del seed
    configured_modes = sum(value is not None for value in (count, per_source_limit, source_counts))
    if configured_modes > 1:
        raise ValueError("Use only one of count, per_source_limit, or source_counts.")

    if count is not None:
        per_source = compute_balanced_counts(count, list(source_rows))
    elif per_source_limit is not None:
        per_source = {name: per_source_limit for name in source_rows}
    elif source_counts is not None:
        missing = sorted(set(source_rows) - set(source_counts))
        if missing:
            raise ValueError(f"Missing --source-count value(s): {', '.join(missing)}")
        per_source = {name: source_counts[name] for name in source_rows}
    else:
        per_source = {
            name: min(DEFAULT_SOURCE_COUNTS.get(name, len(rows)), len(rows))
            for name, rows in source_rows.items()
        }

    for name, wanted in per_source.items():
        if wanted <= 0:
            raise ValueError(f"{name} sample count must be greater than 0.")
        available = len(source_rows[name])
        if wanted > available:
            raise ValueError(f"{name} only has {available} rows, cannot sample {wanted}.")

    return SamplingPlan(per_source=per_source, total=sum(per_source.values()))


def sample_requests(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    plan: SamplingPlan,
    seed: int,
) -> list[ToolCallRequest]:
    rng = Random(seed)
    requests: list[ToolCallRequest] = []
    for source_name, rows in source_rows.items():
        chosen_indices = sorted(rng.sample(range(len(rows)), plan.per_source[source_name]))
        for row_index in chosen_indices:
            requests.append(
                ToolCallRequest(
                    source_name=source_name,
                    row_index=row_index,
                    row=rows[row_index],
                )
            )
    rng.shuffle(requests)
    return requests


def build_first_round_prompt(*, has_image_input: bool, domain: str) -> str:
    image_note = (
        "当前输入包含图片。"
        if has_image_input
        else "当前输入没有图片，因此不能使用图像线索，也不能真正对图片做 Grounding；如果调用搜索工具，必须输出 <Grounding>No</Grounding>。"
    )
    domain_note = (
        "当前样本属于电商域内 VQA：如果问题需要商品页或本地商品库中的价格、规格、参数、售后、店铺、评分、材质、成分等信息，应优先使用 RAG_search。"
        if domain == "in_domain"
        else "当前样本属于域外开源 VQA，不属于本地电商商品库：不要使用 RAG_search；如果图片和常识不足以直接回答，应使用 Web_search。"
    )
    return (
        "你是一名专业的视觉助手。你的任务是基于给定图片回答用户问题，但在第一轮你必须先判断应该直接回答还是调用工具。\n"
        f"{image_note}\n"
        f"{domain_note}\n\n"
        "第 1 步：分析图片。\n"
        "仔细检查图片和用户问题，识别所有可见实体、物体、文字、标志、局部细节以及其他视觉线索。\n\n"
        "第 2 步：规划动作。\n"
        "根据你的分析，你必须执行以下三个动作中的一个。选择动作之前，必须先把思考过程写在 <Think>...</Think> 标签内。"
        "思考过程必须简洁，不要超过 100 个汉字或 60 个英文词，不要分条列举。\n\n"
        "Action 1：直接回答。\n"
        "如果你能够根据图片中的视觉元素、可读文字、标志、结构、布局、常识或自身知识，确信已经有足够事实回答问题，"
        "就直接给出简洁答案，并把答案写在 <Answer>...</Answer> 标签内。直接回答时不要写“无需调用工具”，而是写真实答案。\n"
        "如果选择 Action 1，只能输出 <Think>...</Think> 和 <Answer>...</Answer>，禁止输出 <Grounding>。\n"
        "输出格式：\n"
        "<Think>推理过程</Think>\n"
        "<Answer>最终答案</Answer>\n\n"
        "Action 2：使用 RAG_search。\n"
        "如果图片或问题明显关于购物、电商商品、商品价格或商品属性，并且答案通常需要从电商网站、商品页或本地商品库获得，"
        "就使用 RAG_search。必须把简洁的检索对象或类别名写在 <RAG_search>...</RAG_search> 标签内，例如 <RAG_search>iPhone</RAG_search>。\n\n"
        "Action 3：使用 Web_search。\n"
        "如果问题是通用问题，或图片信息以及 Action 2 的电商检索信息不足以回答，需要更具体的互联网外部信息，"
        "就使用 Web_search。必须把简洁的检索对象或查询词写在 <Web_search>...</Web_search> 标签内，例如 <Web_search>大熊猫</Web_search>。\n\n"
        "如果使用 Action 2 或 Action 3，还必须决定是否使用图像处理工具：\n"
        "Grounding Tool：如果问题明确关于某个具体视觉元素，例如物体、人物、动物、植物、飞行器、商品、包装、标签、局部文字、局部图案或局部区域，"
        "或者背景与问题无关，就使用 Grounding Tool，并把简洁目标写在 <Grounding>...</Grounding> 标签内，例如 <Grounding>大熊猫</Grounding>。\n"
        "No Grounding Tool：只有当问题关于整个场景、位置、地点归属、整体环境或整体上下文时，才不使用 Grounding Tool。此时只输出 <Grounding>No</Grounding>。\n\n"
        "请记住：搜索结果会在后续轮次提供给你。所有搜索结果会放在 <information>...</information> 标签内返回。"
        "当你准备最终回答时，才把最终答案写在 <Answer>...</Answer> 标签内。\n\n"
        "本轮只输出第一轮动作所需的标签，不要输出 Markdown，不要输出 JSON，不要输出额外解释。"
        "所有标签必须完整闭合。"
        "如果选择 RAG_search 或 Web_search，本轮不要同时输出 <Answer>。\n\n"
        "输出示例 1：\n"
        "<Think>推理过程</Think>\n"
        "<Answer>...</Answer>\n\n"
        "输出示例 2：\n"
        "<Think>推理过程</Think>\n"
        "<RAG_search>iPhone</RAG_search>\n"
        "<Grounding>iPhone</Grounding>\n\n"
        "输出示例 3：\n"
        "<Think>推理过程</Think>\n"
        "<Web_search>大熊猫</Web_search>\n"
        "<Grounding>大熊猫</Grounding>\n\n"
        "输出示例 4：\n"
        "<Think>推理过程</Think>\n"
        "<Web_search>埃菲尔铁塔</Web_search>\n"
        "<Grounding>No</Grounding>"
    )


def build_system_prompt(request: ToolCallRequest) -> str:
    return build_first_round_prompt(
        has_image_input=request.has_image_input,
        domain=request.domain,
    )


def build_user_content(request: ToolCallRequest) -> list[dict[str, Any]] | str:
    text = f"这里是图片和问题：<image>\n问题：{request.query}"
    if not request.has_image_input:
        return f"这里是问题，没有图片输入。\n问题：{request.query}"
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_path_to_data_url(request.image_path)}},
    ]


def extract_tags(text: str) -> dict[str, list[str]]:
    return {
        tag: [match.group(1).strip() for match in pattern.finditer(text)]
        for tag, pattern in TAG_PATTERNS.items()
    }


def parse_tool_call_output(raw_output: str) -> dict[str, Any]:
    tags = extract_tags(raw_output)

    def latest(tag: str) -> str:
        values = tags.get(tag) or []
        return values[-1] if values else ""

    warnings: list[str] = []
    think = latest("Think")
    direct_answer = latest("Answer")
    rag_input = latest("RAG_search")
    web_input = latest("Web_search")
    grounding_input = latest("Grounding")

    if not think:
        warnings.append("missing_think")
    if rag_input and web_input:
        warnings.append("multiple_search_tools")

    search_tool = "none"
    search_input = ""
    if rag_input:
        search_tool = "RAG_search"
        search_input = rag_input
    elif web_input:
        search_tool = "Web_search"
        search_input = web_input

    if search_tool == "none":
        if not direct_answer:
            warnings.append("missing_action")
        if grounding_input:
            warnings.append("grounding_without_search")
        use_grounding = False
        grounding_input = ""
    else:
        if direct_answer:
            warnings.append("answer_with_search")
        if not grounding_input:
            warnings.append("missing_grounding")
        use_grounding = grounding_input.strip().lower() not in NO_GROUNDING_VALUES

    tool_calls: list[str] = []
    if search_tool != "none":
        tool_calls.append(search_tool)
        if use_grounding:
            tool_calls.append("图像裁剪")

    if search_tool == "none":
        decision_type = "direct_answer"
        answer = direct_answer
    else:
        decision_type = "tool_call"
        answer = f"调用工具【{'，'.join(tool_calls)}】"

    return {
        "raw_output": raw_output.strip(),
        "think": think,
        "answer": answer,
        "direct_answer": direct_answer,
        "decision_type": decision_type,
        "search_tool": search_tool,
        "search_input": search_input,
        "use_grounding": use_grounding,
        "grounding_input": grounding_input if search_tool != "none" else "",
        "tool_calls": tool_calls,
        "parsed_tags": tags,
        "parse_warnings": warnings,
    }


def generate_one_tool_call(
    request: ToolCallRequest,
    config: ServerConfig,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": config.alias,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.2,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": build_system_prompt(request)},
            {"role": "user", "content": build_user_content(request)},
        ],
    }
    started_at = time.perf_counter()
    response = post_json(f"{config.base_url}/v1/chat/completions", payload)
    elapsed_seconds = time.perf_counter() - started_at
    raw_output = str(response["choices"][0]["message"]["content"]).strip()
    return parse_tool_call_output(raw_output), elapsed_seconds


def detect_image_format(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def build_result_record(
    request: ToolCallRequest,
    parsed: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    row = request.row
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "tool_plan": {
                "decision_type": parsed["decision_type"],
                "search_tool": parsed["search_tool"],
                "search_input": parsed["search_input"],
                "use_grounding": parsed["use_grounding"],
                "grounding_input": parsed["grounding_input"],
                "tool_calls": parsed["tool_calls"],
            },
            "source_record_id": row.get("record_id"),
            "source_task_type": row.get("task_type"),
            "source_primary_answer": row.get("primary_answer"),
            "source_answers": row.get("answers"),
            "generation_row_index": request.row_index,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "parse_warnings": parsed["parse_warnings"],
        }
    )
    return {
        "record_id": f"tool_call:{request.source_name}:{request.row_index:06d}",
        "task_type": "single_image_tool_call",
        "domain": request.domain,
        "source_dataset": request.source_name,
        "source_file": str(SOURCES[request.source_name].path.resolve()),
        "source_record_key": request.record_id,
        "language": request.language,
        "query": request.query,
        "raw_output": parsed["raw_output"],
        "think": parsed["think"],
        "answer": parsed["answer"],
        "direct_answer": parsed["direct_answer"],
        "decision_type": parsed["decision_type"],
        "search_tool": parsed["search_tool"],
        "search_input": parsed["search_input"],
        "use_grounding": parsed["use_grounding"],
        "grounding_input": parsed["grounding_input"],
        "tool_calls": parsed["tool_calls"],
        "parsed_tags": parsed["parsed_tags"],
        "parse_warnings": parsed["parse_warnings"],
        "image_path": str(request.image_path.resolve()),
        "image_rel_path": request.image_rel_path,
        "image_id": row.get("image_id"),
        "image_format": detect_image_format(request.image_path),
        "metadata": metadata,
    }


def render_progress(current: int, total: int, succeeded: int, failed: int) -> None:
    width = 30
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total * 100) if total else 100.0
    print(
        f"\r[{bar}] {current}/{total} ({percent:5.1f}%) success={succeeded} failed={failed}",
        end="",
        flush=True,
    )


def split_records_by_source(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_dataset"])].append(record)
    return dict(grouped)


def build_tool_ratio_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(records)
    counts = Counter()
    counts["直接回答"] = sum(1 for record in records if record["search_tool"] == "none")
    counts["RAG_search"] = sum(1 for record in records if record["search_tool"] == "RAG_search")
    counts["Web_search"] = sum(1 for record in records if record["search_tool"] == "Web_search")
    counts["图像裁剪"] = sum(1 for record in records if record["use_grounding"])
    return [
        {
            "tool_class": name,
            "count": count,
            "percent": round(count / total * 100, 2) if total else 0.0,
        }
        for name, count in counts.items()
    ]


@dataclass(slots=True)
class ToolCallSummaryAccumulator:
    generated_count: int = 0
    generation_seconds_total: float = 0.0
    retried_record_count: int = 0
    retry_attempts_total: int = 0
    search_tool_distribution: Counter[str] = field(default_factory=Counter)
    action_distribution: Counter[str] = field(default_factory=Counter)
    grounding_distribution: Counter[bool] = field(default_factory=Counter)
    warning_distribution: Counter[str] = field(default_factory=Counter)
    per_source_counts: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        self.generated_count += 1
        self.generation_seconds_total += float(record["metadata"]["elapsed_seconds"])
        retry_count = int(record.get("retry_count") or 0)
        if retry_count:
            self.retried_record_count += 1
            self.retry_attempts_total += retry_count
        self.search_tool_distribution[str(record["search_tool"])] += 1
        self.action_distribution[str(record["answer"])] += 1
        self.grounding_distribution[bool(record["use_grounding"])] += 1
        self.warning_distribution.update(record.get("parse_warnings") or [])
        self.per_source_counts[str(record["source_dataset"])] += 1

    def tool_ratio_table(self) -> list[dict[str, Any]]:
        total = self.generated_count
        rows = [
            ("直接回答", self.search_tool_distribution["none"]),
            ("RAG_search", self.search_tool_distribution["RAG_search"]),
            ("Web_search", self.search_tool_distribution["Web_search"]),
            ("图像裁剪", self.grounding_distribution[True]),
        ]
        return [
            {
                "tool_class": name,
                "count": count,
                "percent": round(count / total * 100, 2) if total else 0.0,
            }
            for name, count in rows
        ]

    def to_summary(
        self,
        *,
        requested_counts: dict[str, int],
        startup_seconds: float,
        total_wall_seconds: float,
    ) -> dict[str, Any]:
        return {
            "requested_counts": requested_counts,
            "generated_count": self.generated_count,
            "server_startup_seconds": round(startup_seconds, 3),
            "generation_seconds_total": round(self.generation_seconds_total, 3),
            "total_wall_seconds": round(total_wall_seconds, 3),
            "average_elapsed_seconds": (
                round(self.generation_seconds_total / self.generated_count, 3)
                if self.generated_count
                else 0.0
            ),
            "retried_record_count": self.retried_record_count,
            "retry_attempts_total": self.retry_attempts_total,
            "tool_ratio_table": self.tool_ratio_table(),
            "search_tool_distribution": dict(self.search_tool_distribution),
            "action_distribution": dict(self.action_distribution),
            "use_grounding_distribution": {
                str(key).lower(): value
                for key, value in self.grounding_distribution.items()
            },
            "parse_warning_distribution": dict(self.warning_distribution),
            "per_source_counts": dict(self.per_source_counts),
        }


def build_run_summary(
    records: list[dict[str, Any]],
    *,
    requested_counts: dict[str, int],
    startup_seconds: float,
    total_wall_seconds: float,
) -> dict[str, Any]:
    generation_seconds_total = sum(float(record["metadata"]["elapsed_seconds"]) for record in records)
    per_source = split_records_by_source(records)
    search_tool_distribution = Counter(record["search_tool"] for record in records)
    action_distribution = Counter(record["answer"] for record in records)
    grounding_distribution = Counter(bool(record["use_grounding"]) for record in records)
    warning_distribution: Counter[str] = Counter()
    for record in records:
        warning_distribution.update(record.get("parse_warnings") or [])
    return {
        "requested_counts": requested_counts,
        "generated_count": len(records),
        "server_startup_seconds": round(startup_seconds, 3),
        "generation_seconds_total": round(generation_seconds_total, 3),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "average_elapsed_seconds": round(generation_seconds_total / len(records), 3) if records else 0.0,
        "tool_ratio_table": build_tool_ratio_table(records),
        "search_tool_distribution": dict(search_tool_distribution),
        "action_distribution": dict(action_distribution),
        "use_grounding_distribution": {str(key).lower(): value for key, value in grounding_distribution.items()},
        "parse_warning_distribution": dict(warning_distribution),
        "per_source_counts": {name: len(rows) for name, rows in per_source.items()},
    }


def run_generation(
    requests: list[ToolCallRequest],
    *,
    config: ServerConfig,
    run_dir: Path,
    max_tokens: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    combined_path = run_dir / "tool_call_records.jsonl"
    summary_path = run_dir / "run_summary.json"
    failed_path = run_dir / "failed_rows.jsonl"
    server_log_path = run_dir / "llama_server.log"
    prompt_path = run_dir / "first_round_prompt.md"
    per_source_dir = run_dir / "per_source"

    started_at = time.perf_counter()
    requested_counts = dict(Counter(request.source_name for request in requests))
    summary_accumulator = ToolCallSummaryAccumulator()
    failed_count = 0

    prompt_path.write_text(
        "# 第一轮工具调用 Prompt\n\n"
        "## 图文电商域内\n\n"
        + build_first_round_prompt(has_image_input=True, domain="in_domain")
        + "\n\n## 纯文本电商域内\n\n"
        + build_first_round_prompt(has_image_input=False, domain="in_domain")
        + "\n\n## 域外图文\n\n"
        + build_first_round_prompt(has_image_input=True, domain="out_of_domain")
        + "\n",
        encoding="utf-8",
    )

    def generate_request(
        order: int,
        request: ToolCallRequest,
    ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        last_error = ""
        last_parsed: dict[str, Any] | None = None
        total_elapsed_seconds = 0.0
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                parsed, elapsed_seconds = generate_one_tool_call(
                    request,
                    config,
                    max_tokens=max_tokens,
                )
                total_elapsed_seconds += elapsed_seconds
                last_parsed = parsed
            except Exception as exc:
                last_error = str(exc)
                if attempt <= max_retries:
                    continue
                return (
                    order,
                    None,
                    {
                        "source_dataset": request.source_name,
                        "source_record_key": request.record_id,
                        "row_index": request.row_index,
                        "query": request.query,
                        "image_path": str(request.image_path),
                        "attempts": attempt,
                        "error": last_error,
                    },
                )

            parse_warnings = list(parsed.get("parse_warnings") or [])
            if not parse_warnings:
                record = build_result_record(request, parsed, total_elapsed_seconds)
                retry_count = attempt - 1
                record["retry_count"] = retry_count
                record["attempts"] = attempt
                record["metadata"]["retry_count"] = retry_count
                record["metadata"]["attempts"] = attempt
                return order, record, None

            last_error = f"parse_warnings: {', '.join(parse_warnings)}"
            if attempt <= max_retries:
                continue

        assert last_parsed is not None
        return (
            order,
            None,
            {
                "source_dataset": request.source_name,
                "source_record_key": request.record_id,
                "row_index": request.row_index,
                "query": request.query,
                "image_path": str(request.image_path),
                "attempts": attempts,
                "error": last_error,
                "parse_warnings": last_parsed.get("parse_warnings") or [],
                "raw_output": last_parsed.get("raw_output") or "",
            },
        )

    max_workers = max(1, int(config.parallel or 1))
    next_request_index = 0
    next_write_order = 1
    completed_count = 0
    pending_records: dict[int, dict[str, Any]] = {}
    pending_failed_rows: dict[int, dict[str, Any]] = {}

    def submit_next(
        executor: ThreadPoolExecutor,
        in_flight: dict[Future[tuple[int, dict[str, Any] | None, dict[str, Any] | None]], int],
    ) -> None:
        nonlocal next_request_index
        if next_request_index >= len(requests):
            return
        order = next_request_index + 1
        future = executor.submit(generate_request, order, requests[next_request_index])
        in_flight[future] = order
        next_request_index += 1

    def flush_ready_rows() -> None:
        nonlocal next_write_order, failed_count
        while True:
            record = pending_records.pop(next_write_order, None)
            if record is not None:
                append_jsonl(combined_path, record)
                append_jsonl(per_source_dir / f"{record['source_dataset']}.jsonl", record)
                summary_accumulator.add(record)
                next_write_order += 1
                continue

            failed_row = pending_failed_rows.pop(next_write_order, None)
            if failed_row is not None:
                append_jsonl(failed_path, failed_row)
                failed_count += 1
                next_write_order += 1
                continue

            break

    per_source_dir.mkdir(parents=True, exist_ok=True)

    render_progress(0, len(requests), 0, 0)
    with run_llama_server(config, log_path=server_log_path) as startup_seconds:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[Future[tuple[int, dict[str, Any] | None, dict[str, Any] | None]], int] = {}
            for _ in range(min(max_workers, len(requests))):
                submit_next(executor, in_flight)

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    completed_count += 1
                    order, record, failed_row = future.result()
                    if failed_row is not None:
                        pending_failed_rows[order] = failed_row
                    else:
                        assert record is not None
                        pending_records[order] = record
                    flush_ready_rows()
                    submit_next(executor, in_flight)

                displayed_failed_count = failed_count + len(pending_failed_rows)
                render_progress(
                    completed_count,
                    len(requests),
                    summary_accumulator.generated_count + len(pending_records),
                    displayed_failed_count,
                )
    print()
    flush_ready_rows()

    total_wall_seconds = time.perf_counter() - started_at
    summary = summary_accumulator.to_summary(
        requested_counts=requested_counts,
        startup_seconds=startup_seconds,
        total_wall_seconds=total_wall_seconds,
    )
    summary["combined_path"] = str(combined_path)
    summary["server_log_path"] = str(server_log_path)
    summary["prompt_path"] = str(prompt_path)
    summary["server_config"] = {
        "ctx_size": config.ctx_size,
        "parallel": config.parallel,
        "request_workers": max_workers,
        "threads": config.threads,
        "flash_attn": config.flash_attn,
    }
    summary["failed_count"] = failed_count
    summary["max_retries"] = max_retries
    if failed_count:
        summary["failed_rows_path"] = str(failed_path)

    write_json(summary_path, summary)
    return [], [], summary
