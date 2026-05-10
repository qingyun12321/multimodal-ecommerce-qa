from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable

from ecom_qa.data.qa_generation import (
    MODEL_ALIAS,
    ServerConfig,
    extract_json_object,
    image_path_to_data_url,
    post_json,
    run_llama_server,
    write_json,
    write_jsonl,
)


UNIFIED_VQA_DIR = Path("dataset/unified_vqa")


@dataclass(frozen=True, slots=True)
class ToolCallSource:
    name: str
    path: Path
    domain: str
    language: str
    prompt_style: str


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    source_name: str
    prompt_style: str
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
        prompt_style="in_domain",
    ),
    "ecom_qa_pairs_open_question": ToolCallSource(
        name="ecom_qa_pairs_open_question",
        path=UNIFIED_VQA_DIR / "ecom_qa_pairs_open_question.jsonl",
        domain="in_domain",
        language="zh",
        prompt_style="in_domain",
    ),
    "infoseek_sample": ToolCallSource(
        name="infoseek_sample",
        path=UNIFIED_VQA_DIR / "infoseek_sample.jsonl",
        domain="out_of_domain",
        language="en",
        prompt_style="out_of_domain",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_source_rows(source_names: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in source_names:
        source = SOURCES[name]
        loaded[name] = load_jsonl(source.path)
    return loaded


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
    seed: int,
) -> SamplingPlan:
    del seed
    if count is not None and per_source_limit is not None:
        raise ValueError("Use either count or per_source_limit, not both.")
    if count is None and per_source_limit is None:
        per_source = {name: len(rows) for name, rows in source_rows.items()}
    elif count is not None:
        per_source = compute_balanced_counts(count, list(source_rows))
    else:
        assert per_source_limit is not None
        per_source = {name: per_source_limit for name in source_rows}

    for name, wanted in per_source.items():
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
        source = SOURCES[source_name]
        for row_index in chosen_indices:
            requests.append(
                ToolCallRequest(
                    source_name=source_name,
                    prompt_style=source.prompt_style,
                    row_index=row_index,
                    row=rows[row_index],
                )
            )
    rng.shuffle(requests)
    return requests


def build_system_prompt(request: ToolCallRequest) -> str:
    common = (
        "你是一个多模态工具调用数据标注助手。"
        "你的任务不是回答用户问题，而是判断为了回答该问题，模型应该直接回答还是调用工具。"
        "可用能力只有四类："
        "1. 直接回答：当图片本身加上常识已经足够作答时使用。"
        "2. RAG_search：仅用于图片或问题明显围绕某个电商商品，且问题所求信息通常需要从电商网站或商品页检索才能获得时使用，例如价格、品牌、颜色、规格、参数、售后、商品属性等。"
        "3. Web_search：用于通用世界知识、品牌历史、人物、动物、地点、时间、发明者、百科信息等网络检索。"
        "4. 图像裁剪：当问题只关注图片中的某个具体实体，且背景大多无关时，需要同时调用图像裁剪；如果问题关心整个场景、地点或整体环境，则不要调用图像裁剪。"
        "只输出 1 个 JSON 对象，不要输出 Markdown，不要回答原问题。"
        "JSON 字段只能包含 think、search_tool、use_grounding。"
        "其中 search_tool 只能是 none、RAG_search、Web_search 之一。"
        "如果 search_tool 为 none，则 use_grounding 必须为 false。"
        "think 要用自然中文简洁说明判断依据，必须明确提到为什么是直接回答、RAG_search 或 Web_search，以及是否需要图像裁剪。"
        "请严格按第一轮工具规划来判断：你当前只有图片和问题，还没有任何检索返回的 information。"
        "如果图片本身已经足够支持答案，或者答案可以直接通过看图识别、读图中文字、识别品牌标识、观察外观结构、部件类型或布局得到，优先选择直接回答，不要为了保守而额外调用搜索工具。"
        "只有当图片不足以支撑答案时，才调用 RAG_search 或 Web_search。"
        "如果问题明显是具体商品属性或商品页事实，优先考虑先调用 RAG_search；如果后续 RAG 结果仍不足，再在多轮流程里升级到 Web_search。"
        "如果问题明显是品牌历史、人物、动物、地点、发明者、通用常识、命名来源、时间背景等超出商品页的信息，直接选择 Web_search。"
        "对于域外或通用问题，如果图片本身已经足以支持一个高层次、常识性的答案，就优先直接回答；不要因为问题听起来像百科题，就默认调用 Web_search。"
        "如果图片只能支持你识别出一个通用类别、职业、食物类型、可见结构、部件类型、服务场景或明显的用途，而这个层次已经足够回答问题，就直接回答。"
        "如果图片并不能唯一识别出某个具体命名实体，就不要假设你已经知道它是谁、是哪座建筑、哪种材料或哪个品牌系列，再据此调用 Web_search。"
        "图像裁剪不是默认附加项。只有当问题确实需要定位图片中的某个具体对象、局部区域、局部文字、侧面、背面、标签、控制面板等细节，并且背景会干扰判断时，才使用图像裁剪。"
        "只有在以下情况之一满足时，图像裁剪才有价值：1. 图片里有多个候选目标，必须先定位到底问的是哪一个；2. 关键证据是局部小文字、局部按钮、局部标签、局部结构，整图观察不够稳定。"
        "如果搜索本身回答的是抽象属性、产品规格、售后政策、原产地、品牌历史、分类学、发明者、地点归属等信息，而图像裁剪并不能帮助你在第一轮确定工具类别，就不要使用图像裁剪。"
        "如果问题讨论的是整个商品的整体属性、可选颜色、抽象比较、历史背景、命名来源、技术原理、地点归属或整体场景，不要因为它是单个实体就机械加入图像裁剪。"
        "要按问题字面含义判断，不要把模糊词自动扩展成别的意思；例如 source 可能是来源、产生者或出处，不一定是产地。"
        "如果问题需要的答案是图中可见的部件、结构类型、文字、标志、界面布局、局部配置，即使措辞看起来专业，也优先直接回答。"
        "不要假设你已经看过任何商品详情、catalog 行或检索结果。"
        "对于材质、成分、纤维、鞋面材质、包材质、面料成分这类问题，除非图片中有明确可读文字直接写出材质，或视觉证据几乎无歧义，否则默认不要直接回答，优先选择 RAG_search。"
        "如果问题混合了可见信息和不可见商品属性，只要关键答案的一部分不能从图片稳定得到，就不要直接回答。"
    )
    if request.qa_type == "text_only":
        return (
            common
            + "当前样本是纯文本问题，没有图片输入。"
            + "因此你不能依赖任何视觉线索，也不能使用图像裁剪。"
            + "纯文本样本中，只有在常识足以直接回答时才可以直接回答；如果是电商商品事实问题，优先选择 RAG_search；如果是品牌历史、发明者、通用世界知识，则选择 Web_search。"
        )
    return common


def build_user_content(request: ToolCallRequest) -> list[dict[str, Any]]:
    lines = [
        f"问题: {request.query}",
        (
            "请仅根据这张图片和这个问题，判断第一轮应当直接回答还是调用工具。"
            if request.has_image_input
            else "当前只有这个问题文本，没有图片。请据此判断第一轮应当直接回答还是调用工具。"
        ),
    ]
    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    if request.has_image_input:
        content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(request.image_path)}})
    return content


def tool_call_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "think": {"type": "string", "minLength": 6},
                    "search_tool": {
                        "type": "string",
                        "enum": ["none", "RAG_search", "Web_search"],
                    },
                    "use_grounding": {"type": "boolean"},
                },
                "required": ["think", "search_tool", "use_grounding"],
                "additionalProperties": False,
            }
        },
    }


def query_disallows_grounding(query: str) -> bool:
    text = query.strip().lower()
    disallow_markers = [
        "这个街道",
        "这条街",
        "这个场景",
        "这个地方",
        "这片区域",
        "这个环境",
        "背景",
        "where is",
        "where was",
        "what country",
        "what city",
        "which city",
        "which country",
        "what place",
        "street",
        "scene",
        "landscape",
        "environment",
        "background",
        "located",
        "颜色可选",
        "有哪些颜色",
        "屏幕尺寸",
        "为什么",
        "区别",
        "原理",
        "named after",
        "what country",
        "owner",
        "weigh",
        "historic period",
        "gestation",
        "taxonomy",
        "discoverer",
        "inventor",
        "manufacturer",
        "do for a living",
        "原产",
        "原产于",
        "能效",
        "刷新率",
        "退货",
        "售后",
        "功能",
        "主要功能",
        "通常",
        "一般",
        "历史",
        "哪一年",
        "什么时候",
        "why",
        "compare",
        "difference",
    ]
    return any(marker in text for marker in disallow_markers)


def think_has_negated_tool_request(think: str) -> bool:
    text = think.lower()
    markers = [
        "无需调用",
        "不需要调用",
        "无需使用",
        "不需要使用",
        "不应调用",
        "不用调用",
        "无需 r",
        "无需 w",
        "cannot use",
        "no need",
    ]
    return any(marker in text for marker in markers)


def think_requests_rag(think: str) -> bool:
    text = think.lower()
    markers = [
        "应调用rag_search",
        "需要调用rag_search",
        "应使用rag_search",
        "需要使用rag_search",
        "选择rag_search",
        "改为使用 rag_search",
    ]
    return any(marker in text for marker in markers)


def think_requests_web(think: str) -> bool:
    text = think.lower()
    markers = [
        "应调用web_search",
        "需要调用web_search",
        "应使用web_search",
        "需要使用web_search",
        "选择web_search",
        "改为使用 web_search",
        "需要网络检索",
    ]
    return any(marker in text for marker in markers)


def think_requests_grounding(think: str) -> bool:
    text = think.lower()
    markers = [
        "需要图像裁剪",
        "需要裁剪",
        "过滤无关背景",
        "需要 grounding",
        "需要grounding",
        "需要定位局部",
        "需要定位具体对象",
        "多个候选目标",
        "局部文字",
        "局部标签",
    ]
    return any(marker in text for marker in markers)


def normalize_tool_plan(request: ToolCallRequest, raw_plan: dict[str, Any]) -> dict[str, Any]:
    search_tool = str(raw_plan["search_tool"]).strip()
    use_grounding = bool(raw_plan["use_grounding"])
    think = str(raw_plan["think"]).strip()

    if request.prompt_style == "out_of_domain" and search_tool == "RAG_search":
        search_tool = "Web_search"
        if "RAG_search" not in think:
            think = think + " 该样本属于域外问题，因此改为使用 Web_search。"

    if search_tool == "none" and not think_has_negated_tool_request(think):
        if request.prompt_style == "out_of_domain" and think_requests_web(think):
            search_tool = "Web_search"
            think = think + " 由于图片不足以直接回答，该样本改为使用 Web_search。"
        elif request.prompt_style == "in_domain" and think_requests_rag(think):
            search_tool = "RAG_search"
            think = think + " 由于问题依赖本地商品信息，该样本改为使用 RAG_search。"

    if not request.has_image_input:
        use_grounding = False
        think = think.replace("需要图像裁剪", "不需要图像裁剪")
        think = think.replace("需要裁剪", "不需要裁剪")

    if search_tool == "none":
        use_grounding = False
    elif query_disallows_grounding(request.query):
        use_grounding = False
        think = think.replace("需要图像裁剪", "不需要图像裁剪")
        think = think.replace("需要裁剪", "不需要裁剪")
    elif not use_grounding and think_requests_grounding(think):
        use_grounding = True
        think = think.replace("不需要图像裁剪", "需要图像裁剪")
        think = think.replace("无需图像裁剪", "需要图像裁剪")
        think = think.replace("也不需要图像裁剪", "并且需要图像裁剪")
        if "图像裁剪" not in think:
            think = think + " 该问题需要定位局部目标或局部文字，因此补充图像裁剪。"

    tool_calls: list[str] = []
    if search_tool != "none":
        tool_calls.append(search_tool)
        if use_grounding:
            tool_calls.append("图像裁剪")

    if not tool_calls:
        answer = "直接回答，无需调用工具"
        decision_type = "direct_answer"
    else:
        answer = f"调用工具【{'，'.join(tool_calls)}】"
        decision_type = "tool_call"

    return {
        "query": request.query,
        "think": think,
        "answer": answer,
        "decision_type": decision_type,
        "search_tool": search_tool,
        "use_grounding": use_grounding,
        "tool_calls": tool_calls,
    }


def generate_one_tool_call(request: ToolCallRequest, config: ServerConfig) -> tuple[dict[str, Any], float]:
    payload = {
        "model": MODEL_ALIAS,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.2,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": tool_call_response_format(),
        "messages": [
            {"role": "system", "content": build_system_prompt(request)},
            {"role": "user", "content": build_user_content(request)},
        ],
    }
    started_at = time.perf_counter()
    response = post_json(f"{config.base_url}/v1/chat/completions", payload)
    elapsed_seconds = time.perf_counter() - started_at
    message = response["choices"][0]["message"]["content"]
    raw_plan = extract_json_object(message)
    return normalize_tool_plan(request, raw_plan), elapsed_seconds


def detect_image_format(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return suffix[1:] if suffix.startswith(".") else suffix


def build_result_record(
    request: ToolCallRequest,
    tool_plan: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    row = request.row
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "tool_plan": {
                "decision_type": tool_plan["decision_type"],
                "search_tool": tool_plan["search_tool"],
                "use_grounding": tool_plan["use_grounding"],
                "tool_calls": tool_plan["tool_calls"],
            },
            "source_record_id": row.get("record_id"),
            "source_task_type": row.get("task_type"),
            "source_primary_answer": row.get("primary_answer"),
            "source_answers": row.get("answers"),
            "generation_row_index": request.row_index,
            "elapsed_seconds": round(elapsed_seconds, 3),
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
        "query": tool_plan["query"],
        "think": tool_plan["think"],
        "answer": tool_plan["answer"],
        "decision_type": tool_plan["decision_type"],
        "search_tool": tool_plan["search_tool"],
        "use_grounding": tool_plan["use_grounding"],
        "tool_calls": tool_plan["tool_calls"],
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


def build_run_summary(
    records: list[dict[str, Any]],
    *,
    requested_counts: dict[str, int],
    startup_seconds: float,
    total_wall_seconds: float,
) -> dict[str, Any]:
    generation_seconds_total = sum(float(record["metadata"]["elapsed_seconds"]) for record in records)
    tool_distribution = Counter(record["answer"] for record in records)
    grounding_distribution = Counter(bool(record["use_grounding"]) for record in records)
    per_source = split_records_by_source(records)
    return {
        "requested_counts": requested_counts,
        "generated_count": len(records),
        "server_startup_seconds": round(startup_seconds, 3),
        "generation_seconds_total": round(generation_seconds_total, 3),
        "total_wall_seconds": round(total_wall_seconds, 3),
        "average_elapsed_seconds": round(generation_seconds_total / len(records), 3) if records else 0.0,
        "tool_answer_distribution": dict(tool_distribution),
        "use_grounding_distribution": {str(key).lower(): value for key, value in grounding_distribution.items()},
        "per_source_counts": {name: len(rows) for name, rows in per_source.items()},
    }


def run_generation(
    requests: list[ToolCallRequest],
    *,
    config: ServerConfig,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    combined_path = run_dir / "tool_call_records.jsonl"
    summary_path = run_dir / "run_summary.json"
    failed_path = run_dir / "failed_rows.jsonl"
    server_log_path = run_dir / "llama_server.log"
    per_source_dir = run_dir / "per_source"

    started_at = time.perf_counter()
    records: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    requested_counts = dict(Counter(request.source_name for request in requests))
    render_progress(0, len(requests), 0, 0)
    with run_llama_server(config, log_path=server_log_path) as startup_seconds:
        for index, request in enumerate(requests, start=1):
            try:
                tool_plan, elapsed_seconds = generate_one_tool_call(request, config)
            except Exception as exc:
                failed_rows.append(
                    {
                        "source_dataset": request.source_name,
                        "source_record_key": request.record_id,
                        "row_index": request.row_index,
                        "query": request.query,
                        "image_path": str(request.image_path),
                        "error": str(exc),
                    }
                )
                render_progress(index, len(requests), len(records), len(failed_rows))
                continue
            records.append(build_result_record(request, tool_plan, elapsed_seconds))
            render_progress(index, len(requests), len(records), len(failed_rows))
    print()

    total_wall_seconds = time.perf_counter() - started_at
    summary = build_run_summary(
        records,
        requested_counts=requested_counts,
        startup_seconds=startup_seconds,
        total_wall_seconds=total_wall_seconds,
    )
    summary["combined_path"] = str(combined_path)
    summary["server_log_path"] = str(server_log_path)
    if failed_rows:
        summary["failed_rows_path"] = str(failed_path)

    write_jsonl(combined_path, records)
    per_source_dir.mkdir(parents=True, exist_ok=True)
    for source_name, rows in split_records_by_source(records).items():
        write_jsonl(per_source_dir / f"{source_name}.jsonl", rows)
    if failed_rows:
        write_jsonl(failed_path, failed_rows)
    write_json(summary_path, summary)
    return records, failed_rows, summary
