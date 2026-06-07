#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecom_qa.common.paths import repo_root, resolve_data_root
from ecom_qa.tool_calls.multiturn import (  # noqa: E402
    LocalRAGSearch,
    UNABLE_TO_ANSWER,
    format_information,
)
from ecom_qa.retrieval.web import (  # noqa: E402
    LlamaCppSummarizer,
    SearXNGClient,
    format_web_information,
    normalize_training_text,
    summarize_search_results,
)


TAG_NAMES = ("Think", "Answer", "RAG_search", "Web_search", "Grounding", "information")
TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in TAG_NAMES
}
LOCAL_ROOT = "/home/qingyun/projects/multimodal-ecommerce-qa"
DEFAULT_DATA_DIR = Path("/workspace/data/ecom-qa-tool-call-sft-v2")
DEFAULT_BLOCKLIST = {
    "tool_call:infoseek_sample:006131",
    "tool_call:infoseek_sample:012005",
    "tool_call:ecom_qa_pairs_supplement:002220",
    "tool_call:infoseek_sample:005339",
    "tool_call:ecom_qa_pairs_supplement:003194",
    "tool_call:infoseek_sample:009079",
    "tool_call:ecom_qa_pairs_supplement:000571",
    "tool_call:ecom_qa_pairs_supplement:002372",
    "tool_call:infoseek_sample:008326",
    "tool_call:ecom_qa_pairs_open_question:003305",
    "tool_call:infoseek_sample:002058",
    "tool_call:infoseek_sample:014780",
    "tool_call:infoseek_sample:000667",
    "tool_call:ecom_qa_pairs_open_question:004611",
    "tool_call:infoseek_sample:005592",
    "tool_call:ecom_qa_pairs_supplement:000404",
    "tool_call:ecom_qa_pairs_open_question:004760",
    "tool_call:infoseek_sample:007557",
    "tool_call:ecom_qa_pairs_supplement:002380",
    "tool_call:ecom_qa_pairs_supplement:004746",
    "tool_call:infoseek_sample:014708",
    "tool_call:infoseek_sample:010013",
    "tool_call:infoseek_sample:002641",
    "tool_call:ecom_qa_pairs_supplement:001661",
    "tool_call:ecom_qa_pairs_open_question:003524",
    "tool_call:infoseek_sample:014676",
    "tool_call:infoseek_sample:004388",
    "tool_call:infoseek_sample:013291",
    "tool_call:ecom_qa_pairs_supplement:002750",
    "tool_call:infoseek_sample:004165",
    "tool_call:ecom_qa_pairs_supplement:001843",
    "tool_call:ecom_qa_pairs_open_question:002698",
    "tool_call:infoseek_sample:000507",
    "tool_call:infoseek_sample:010617",
    "tool_call:infoseek_sample:006105",
    "tool_call:infoseek_sample:008070",
    "tool_call:ecom_qa_pairs_open_question:004146",
    "tool_call:infoseek_sample:013673",
    "tool_call:infoseek_sample:007490",
    "tool_call:ecom_qa_pairs_supplement:000889",
    "tool_call:ecom_qa_pairs_supplement:003917",
    "tool_call:infoseek_sample:002337",
    "tool_call:ecom_qa_pairs_supplement:000291",
    "tool_call:ecom_qa_pairs_open_question:004089",
    "tool_call:infoseek_sample:008470",
    "tool_call:infoseek_sample:001103",
    "tool_call:ecom_qa_pairs_supplement:002761",
}


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
        "你是一名专业的视觉助手。你的任务是基于给定图片回答用户问题，并在需要时规划工具调用。\n"
        f"{image_note}\n{domain_note}\n\n"
        "第一轮必须先在 <Think>...</Think> 中写出简洁判断，然后选择一个动作：\n"
        "Action 1：如果视觉信息、可读文字、常识或自身知识已经足够回答，输出 <Answer>最终答案</Answer>。\n"
        "Action 2：如果需要电商商品库、商品页或本地商品属性信息，输出 <RAG_search>检索词</RAG_search>。\n"
        "Action 3：如果需要互联网外部信息，输出 <Web_search>检索词</Web_search>。\n"
        "选择 RAG_search 或 Web_search 时必须同时输出 <Grounding>目标</Grounding>；不需要裁剪时输出 <Grounding>No</Grounding>。"
        "Grounding 是模型应判断和输出的文本标签，本数据流程不实际调用裁剪工具。\n\n"
        "工具返回会在后续轮次以 <information>...</information> 放入用户消息。"
        "后续轮次如果能够回答，必须输出 <Answer>最终答案</Answer>；仍缺少信息时输出固定答案："
        f"<Answer>{UNABLE_TO_ANSWER}</Answer>。\n"
        "不要输出 Markdown、JSON 或额外解释，所有标签必须完整闭合。"
    )


def build_after_tool_think(tool: str, answer: str) -> str:
    if answer == UNABLE_TO_ANSWER:
        return "工具返回的信息仍不足以支持可靠回答，因此给出固定无法回答结果。"
    if tool == "RAG_search":
        return "已根据电商检索信息定位到相关商品属性，可以给出答案。"
    return "已根据网页摘要补充外部事实，可以给出答案。"


def clean_tag_literals(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(
        r"</?(Think|Answer|RAG_search|Web_search|Grounding|information)>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:[-*]|\d+[.)])\s*", "", text)
    return text.strip()


def extract_tags(text: str) -> dict[str, list[str]]:
    return {tag: [m.group(1).strip() for m in pattern.finditer(text)] for tag, pattern in TAG_PATTERNS.items()}


def validate_assistant_output(raw: str, *, round_name: str) -> list[str]:
    tags = extract_tags(raw)
    counts = {tag: len(tags[tag]) for tag in TAG_NAMES}
    warnings: list[str] = []
    if counts["Think"] != 1:
        warnings.append(f"{round_name}:think_count={counts['Think']}")
    if counts["information"]:
        warnings.append(f"{round_name}:assistant_contains_information")
    if round_name == "first":
        search_tags = [tag for tag in ("RAG_search", "Web_search") if counts[tag] > 0]
        if counts["Answer"] and search_tags:
            warnings.append("first:answer_with_search")
        if len(search_tags) > 1:
            warnings.append("first:multiple_search_tags")
        if counts["Answer"] == 1 and not search_tags and counts["Grounding"]:
            warnings.append("first:direct_answer_with_grounding")
        if len(search_tags) == 1 and counts["Grounding"] != 1:
            warnings.append("first:tool_call_missing_grounding")
        if not counts["Answer"] and not search_tags:
            warnings.append("first:missing_action")
    else:
        if counts["Answer"] != 1:
            warnings.append(f"{round_name}:answer_count={counts['Answer']}")
        if counts["RAG_search"] or counts["Grounding"]:
            warnings.append(f"{round_name}:unexpected_tool_tag")
    for tag in TAG_NAMES:
        opens = len(re.findall(rf"<{tag}>", raw, flags=re.IGNORECASE))
        closes = len(re.findall(rf"</{tag}>", raw, flags=re.IGNORECASE))
        if opens != closes:
            warnings.append(f"{round_name}:unclosed_{tag}")
    return warnings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_blocklist(path: Path | None) -> set[str]:
    blocklist = set(DEFAULT_BLOCKLIST)
    if path is None or not path.exists():
        return blocklist
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("issues") or payload.get("record_ids") or []
        else:
            rows = payload
        for row in rows:
            if isinstance(row, str):
                blocklist.add(row)
            elif isinstance(row, dict) and row.get("record_id"):
                blocklist.add(str(row["record_id"]))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                blocklist.add(line)
    return blocklist


def has_image(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return str(metadata.get("qa_type") or "multimodal").lower() != "text_only"


def qa_type(row: dict[str, Any]) -> str:
    return "multimodal" if has_image(row) else "text_only"


def remote_image_path(row: dict[str, Any], repo_dir: Path) -> Path:
    raw = str(row.get("image_path") or "")
    if raw.startswith(LOCAL_ROOT):
        return repo_dir / raw[len(LOCAL_ROOT) :].lstrip("/")
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    rel = str(row.get("image_rel_path") or "").lstrip("/")
    project_data_dir = resolve_data_root(repo_dir)
    candidates = [project_data_dir / rel, project_data_dir / "infoseek_sample" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_dir / raw.lstrip("/")


def source_entity(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    for key in ("entity_text", "source_title", "product_id"):
        value = clean_tag_literals(metadata.get(key) or "")
        if value:
            return value
    return ""


def web_search_query(row: dict[str, Any], question: str | None = None) -> str:
    return clean_tag_literals(question or row.get("query") or "")


def canonical_first_output(row: dict[str, Any]) -> str:
    decision_type = str(row.get("decision_type") or "")
    if decision_type == "direct_answer":
        think = "根据图片、题目和已有标注信息，可以直接给出简洁答案。"
        answer = normalize_training_text(gold_answer(row))
        return f"<Think>{think}</Think>\n<Answer>{answer}</Answer>"
    think = clean_tag_literals(row.get("think") or "先判断问题是否能直接回答，以及是否需要检索工具。")
    tool = str(row.get("search_tool") or "").strip()
    if tool not in {"RAG_search", "Web_search"}:
        tool = "Web_search"
    if tool == "Web_search":
        search_input = web_search_query(row)
    else:
        search_input = clean_tag_literals(row.get("search_input") or row.get("query") or "")
    grounding = clean_tag_literals(row.get("grounding_input") or "No") or "No"
    return f"<Think>{think}</Think>\n<{tool}>{search_input}</{tool}>\n<Grounding>{grounding}</Grounding>"


def gold_answer(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    candidates = [
        metadata.get("source_primary_answer"),
        row.get("direct_answer"),
        row.get("answer") if str(row.get("decision_type")) == "direct_answer" else "",
    ]
    source_answers = metadata.get("source_answers")
    if isinstance(source_answers, list):
        candidates.extend(source_answers)
    for candidate in candidates:
        text = clean_tag_literals(candidate or "")
        if text and not text.startswith("调用工具"):
            return text
    return UNABLE_TO_ANSWER


def expected_answers(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    answers: list[str] = []
    for candidate in (metadata.get("source_primary_answer"),):
        text = normalize_training_text(clean_tag_literals(candidate or ""))
        if text and text != UNABLE_TO_ANSWER and not text.startswith("调用工具"):
            answers.append(text)
    source_answers = metadata.get("source_answers")
    if isinstance(source_answers, list):
        for candidate in source_answers:
            text = normalize_training_text(clean_tag_literals(candidate or ""))
            if text and text != UNABLE_TO_ANSWER and not text.startswith("调用工具"):
                answers.append(text)
    return list(dict.fromkeys(answers))


def _compact_for_match(text: str) -> str:
    text = normalize_training_text(clean_tag_literals(text)).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def has_required_label(row: dict[str, Any]) -> bool:
    if str(row.get("decision_type")) == "tool_call":
        if str(row.get("search_tool")) == "Web_search" and not source_entity(row):
            return False
        return bool(expected_answers(row))
    return bool(expected_answers(row))


def support_terms(text: str) -> set[str]:
    normalized = normalize_training_text(clean_tag_literals(text)).lower()
    terms: set[str] = set()
    stop_terms = {
        "这款",
        "商品",
        "采用",
        "材质",
        "版型",
        "设计",
        "支持",
        "主要",
        "结合",
        "功能",
        "答案",
        "男士",
        "女士",
    }
    for match in re.findall(r"[a-z]*\d+(?:\.\d+)?[a-z%\u4e00-\u9fff]*|[a-z]{2,}", normalized):
        key = _compact_for_match(match)
        if key and key not in stop_terms:
            terms.add(key)
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        compact = _compact_for_match(chunk)
        if len(compact) <= 8 and compact not in stop_terms:
            terms.add(compact)
        for size in (2, 3, 4):
            for index in range(0, max(0, len(compact) - size + 1)):
                term = compact[index : index + size]
                if term and term not in stop_terms:
                    terms.add(term)
    return terms


def information_supports_expected(information: str, expected: list[str]) -> bool:
    if not expected:
        return False
    info_key = _compact_for_match(information)
    if not info_key:
        return False
    for candidate in expected:
        expected_key = _compact_for_match(candidate)
        if expected_key and (expected_key in info_key or info_key in expected_key):
            return True
        terms = support_terms(candidate)
        if not terms:
            continue
        hits = [term for term in terms if term in info_key]
        strong_hits = [term for term in hits if re.search(r"\d|[a-z]", term) or len(term) >= 2]
        required_hits = 1 if len(terms) <= 2 else 2
        if len(strong_hits) >= required_hits:
            return True
    return False


def requires_explanatory_evidence(question: str) -> bool:
    normalized = normalize_training_text(question).lower()
    markers = (
        "含义",
        "意思",
        "意味",
        "代表",
        "来源",
        "源自",
        "起源",
        "由来",
        "灵感",
        "创立",
        "发明",
        "发现",
        "哪个国家",
        "哪国",
        "哪里",
        "why",
        "meaning",
        "origin",
        "inspiration",
        "invent",
        "discover",
        "founded",
    )
    return any(marker in normalized for marker in markers)


def rag_information_supports_expected(information: str, question: str, expected: list[str]) -> bool:
    if not information_supports_expected(information, expected):
        return False
    if not requires_explanatory_evidence(question):
        return True
    info_key = _compact_for_match(information)
    explanation_markers = (
        "含义",
        "意思",
        "意味",
        "代表",
        "来源",
        "源自",
        "起源",
        "由来",
        "灵感",
        "创立",
        "发明",
        "发现",
        "国家",
        "品牌介绍",
        "品牌故事",
    )
    if any(_compact_for_match(marker) in info_key for marker in explanation_markers):
        return True
    return any(_compact_for_match(candidate) in info_key for candidate in expected)


def web_information_supports_expected(information: str, row: dict[str, Any], expected: list[str]) -> bool:
    if not information_supports_expected(information, expected):
        return False
    entity = source_entity(row)
    if not entity:
        return False
    info_key = _compact_for_match(information)
    entity_terms = support_terms(entity)
    if not entity_terms:
        return False
    return any(term in info_key for term in entity_terms)


def final_answer_output(*, tool: str, answer: str) -> str:
    answer = normalize_training_text(clean_tag_literals(answer)) or UNABLE_TO_ANSWER
    think = build_after_tool_think(tool, answer)
    return f"<Think>{think}</Think>\n<Answer>{answer}</Answer>"


def user_question_content(row: dict[str, Any]) -> str:
    query = str(row.get("query") or "").strip()
    if has_image(row):
        return f"<image>\n{query}"
    return query


def enhanced_rag_query(row: dict[str, Any], question: str, answer: str) -> str:
    parts = [
        row.get("search_input"),
        question,
        answer if answer != UNABLE_TO_ANSWER else "",
    ]
    metadata = row.get("metadata") or {}
    for key in ("source_title", "source_primary_answer", "source_record_id"):
        parts.append(metadata.get(key))
    source_answers = metadata.get("source_answers")
    if isinstance(source_answers, list):
        parts.extend(str(item) for item in source_answers[:3])
    return " ".join(clean_tag_literals(str(part)) for part in parts if part).strip()


def select_smoke_rows(
    rows: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    blocklist: set[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = [row for row in rows if str(row.get("record_id")) not in blocklist and has_required_label(row)]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                qa_type(row),
                str(row.get("decision_type") or ""),
                str(row.get("search_tool") or ""),
                f"ground={bool(row.get('use_grounding'))}",
                str(row.get("domain") or ""),
            ]
        )
        buckets[key].append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    required = [
        lambda r: qa_type(r) == "text_only",
        lambda r: qa_type(r) == "multimodal",
        lambda r: r.get("search_tool") == "Web_search",
        lambda r: r.get("search_tool") == "RAG_search",
        lambda r: r.get("decision_type") == "direct_answer",
        lambda r: bool(r.get("use_grounding")),
    ]
    for predicate in required:
        candidates = [row for row in rows if predicate(row) and str(row.get("record_id")) not in used]
        if candidates:
            row = rng.choice(candidates)
            selected.append(row)
            used.add(str(row.get("record_id")))
    for key in sorted(buckets):
        if len(selected) >= size:
            break
        candidates = [row for row in buckets[key] if str(row.get("record_id")) not in used]
        if not candidates:
            continue
        row = rng.choice(candidates)
        selected.append(row)
        used.add(str(row.get("record_id")))
    remaining = [row for row in rows if str(row.get("record_id")) not in used]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    rng.shuffle(selected)
    return selected[:size]


def build_item(
    row: dict[str, Any],
    *,
    repo_dir: Path,
    rag: LocalRAGSearch,
    searxng: SearXNGClient | None,
    summarizer: LlamaCppSummarizer | None,
    live_web_budget: list[int],
    web_top_k: int,
    summarize_pages: int,
    web_trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    question = str(row.get("query") or "").strip()
    image_input = has_image(row)
    image_path = remote_image_path(row, repo_dir)
    first_output = canonical_first_output(row)
    messages = [
        {"role": "system", "content": build_first_round_prompt(has_image_input=image_input, domain=str(row.get("domain") or ""))},
        {"role": "user", "content": user_question_content(row)},
        {"role": "assistant", "content": first_output},
    ]
    warnings = validate_assistant_output(first_output, round_name="first")
    answer = gold_answer(row)
    expected = expected_answers(row)
    answer_source = "gold" if answer != UNABLE_TO_ANSWER else "missing"
    tool = str(row.get("search_tool") or "none")
    information = ""
    live_web_used = False

    if str(row.get("decision_type")) == "tool_call" and tool == "RAG_search":
        items = rag.search(enhanced_rag_query(row, question, answer))
        information = format_information(items)
        if rag_information_supports_expected(information, question, expected):
            answer = normalize_training_text(gold_answer(row))
            answer_source = "gold_supported_by_rag"
        else:
            answer = UNABLE_TO_ANSWER
            answer_source = "rag_unsupported"
        messages.extend(
            [
                {"role": "user", "content": information},
                {"role": "assistant", "content": final_answer_output(tool=tool, answer=answer)},
            ]
        )
        warnings.extend(validate_assistant_output(messages[-1]["content"], round_name="after_rag"))

    elif str(row.get("decision_type")) == "tool_call" and tool == "Web_search":
        search_input = web_search_query(row, question)
        trace: dict[str, Any] = {
            "record_id": row.get("record_id"),
            "query": question,
            "search_input": search_input,
            "mode": "offline_reference",
        }
        if searxng is not None and live_web_budget[0] > 0:
            started_at = time.perf_counter()
            live_web_budget[0] -= 1
            live_web_used = True
            trace["mode"] = "searxng_llama_summary"
            try:
                results = searxng.search(search_input, num_results=web_top_k)
                summaries = summarize_search_results(
                    results=results,
                    question=question,
                    summarizer=summarizer,
                    max_pages=summarize_pages,
                    allow_fallback=True,
                )
                information = format_web_information(summaries)
                trace["result_count"] = len(results)
                trace["summary_count"] = len(summaries)
                trace["summaries"] = [asdict(summary) for summary in summaries]
                trace["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
                trace["success"] = bool(summaries)
            except Exception as exc:
                trace["success"] = False
                trace["error"] = repr(exc)
        if not information:
            information = (
                "<information>\n"
                "未获得可靠外部网页摘要。\n"
                "</information>"
            )
        if web_information_supports_expected(information, row, expected):
            answer = normalize_training_text(gold_answer(row))
            answer_source = "gold_supported_by_web"
        else:
            information = (
                "<information>\n"
                "未检索到能支持答案的可靠信息。\n"
                "</information>"
            )
            answer = UNABLE_TO_ANSWER
            answer_source = "web_unsupported"
        trace["answer_source"] = answer_source
        trace["final_answer"] = answer
        web_trace.append(trace)
        messages.extend(
            [
                {"role": "user", "content": information},
                {"role": "assistant", "content": final_answer_output(tool=tool, answer=answer)},
            ]
        )
        warnings.extend(validate_assistant_output(messages[-1]["content"], round_name="after_web"))

    item = {
        "messages": messages,
        "record_id": row.get("record_id"),
        "images": [str(image_path)] if image_input else [],
    }
    meta = {
        "record_id": row.get("record_id"),
        "domain": row.get("domain"),
        "qa_type": qa_type(row),
        "decision_type": row.get("decision_type"),
        "search_tool": tool,
        "use_grounding": bool(row.get("use_grounding")),
        "image_path": str(image_path),
        "image_exists": image_path.exists() if image_input else True,
        "turn_count": len(messages) // 2,
        "has_information": bool(information),
        "live_web_used": live_web_used,
        "answer_source": answer_source,
        "warnings": warnings,
    }
    return item, meta


def build_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    args: argparse.Namespace,
    rag: LocalRAGSearch,
    searxng: SearXNGClient | None,
    summarizer: LlamaCppSummarizer | None,
    live_web_budget: list[int],
    web_trace: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    size = args.smoke_train_size if split == "train" else args.smoke_val_size
    selected = (
        select_smoke_rows(
            rows,
            size=size * 12,
            seed=args.seed + (0 if split == "train" else 1),
            blocklist=args.blocklist,
        )
        if args.smoke
        else [row for row in rows if str(row.get("record_id")) not in args.blocklist and has_required_label(row)]
    )
    items: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for row in selected:
        item, meta = build_item(
            row,
            repo_dir=args.repo_dir,
            rag=rag,
            searxng=searxng,
            summarizer=summarizer,
            live_web_budget=live_web_budget,
            web_top_k=args.web_top_k,
            summarize_pages=args.summarize_pages,
            web_trace=web_trace,
        )
        if args.smoke and meta["answer_source"] == "web_unsupported":
            continue
        items.append(item)
        metas.append(meta)
        if args.smoke and len(items) >= size:
            break
    return items, metas


def split_report(items: list[dict[str, Any]], metas: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [meta for meta in metas if meta["warnings"]]
    return {
        "rows": len(items),
        "unique_record_ids": len({str(item["record_id"]) for item in items}),
        "missing_images": sum(1 for meta in metas if not meta["image_exists"]),
        "invalid_format": len(invalid),
        "qa_type_counts": dict(Counter(meta["qa_type"] for meta in metas)),
        "decision_type_counts": dict(Counter(str(meta["decision_type"]) for meta in metas)),
        "search_tool_counts": dict(Counter(str(meta["search_tool"]) for meta in metas)),
        "turn_count_counts": dict(Counter(str(meta["turn_count"]) for meta in metas)),
        "information_rows": sum(1 for meta in metas if meta["has_information"]),
        "live_web_rows": sum(1 for meta in metas if meta["live_web_used"]),
        "grounding_rows": sum(1 for meta in metas if meta["use_grounding"]),
        "examples_with_warnings": invalid[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict multi-turn SFT JSONL for ms-swift.")
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("REPO_DIR", repo_root())))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("SFT_DATA_DIR", DEFAULT_DATA_DIR)))
    parser.add_argument("--train-source", type=Path, default=None)
    parser.add_argument("--val-source", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-size", type=int, default=32)
    parser.add_argument("--smoke-val-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--web-mode", choices=("offline", "searxng"), default="offline")
    parser.add_argument("--searxng-url", default="http://127.0.0.1:8080")
    parser.add_argument("--searxng-engines", default=os.environ.get("SEARXNG_ENGINES", "bing"))
    parser.add_argument("--summary-url", default="http://127.0.0.1:8088")
    parser.add_argument("--summary-model", default="qwen3.5-9b-summary")
    parser.add_argument("--web-top-k", type=int, default=5)
    parser.add_argument("--summarize-pages", type=int, default=2)
    parser.add_argument("--live-web-limit", type=int, default=0)
    parser.add_argument("--require-live-web", action="store_true")
    parser.add_argument("--blocklist-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_dir = args.repo_dir.resolve()
    args.blocklist = load_blocklist(args.blocklist_file)
    project_data_dir = resolve_data_root(args.repo_dir)
    train_source = args.train_source or project_data_dir / "tool_call" / "tool_call_records_train.jsonl"
    val_source = args.val_source or project_data_dir / "tool_call" / "tool_call_records_test.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rag = LocalRAGSearch.from_dataset(project_data_dir, top_k=5)
    searxng = SearXNGClient(args.searxng_url, engines=args.searxng_engines) if args.web_mode == "searxng" else None
    summarizer = LlamaCppSummarizer(args.summary_url, model=args.summary_model) if args.web_mode == "searxng" else None
    live_web_budget = [args.live_web_limit]
    web_trace: list[dict[str, Any]] = []
    train_rows = read_jsonl(train_source)
    val_rows = read_jsonl(val_source)

    started_at = time.perf_counter()
    train_items, train_meta = build_split(
        train_rows,
        split="train",
        args=args,
        rag=rag,
        searxng=searxng,
        summarizer=summarizer,
        live_web_budget=live_web_budget,
        web_trace=web_trace,
    )
    val_items, val_meta = build_split(
        val_rows,
        split="val",
        args=args,
        rag=rag,
        searxng=searxng,
        summarizer=summarizer,
        live_web_budget=live_web_budget,
        web_trace=web_trace,
    )

    prefix = "smoke_" if args.smoke else ""
    write_jsonl(args.output_dir / f"{prefix}train.ms_swift.jsonl", train_items)
    write_jsonl(args.output_dir / f"{prefix}val.ms_swift.jsonl", val_items)
    write_jsonl(args.output_dir / f"{prefix}review_samples.jsonl", train_items[:20] + val_items[:20])
    write_jsonl(args.output_dir / f"{prefix}metadata.jsonl", train_meta + val_meta)
    write_jsonl(args.output_dir / f"{prefix}web_summary_trace.jsonl", web_trace)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "repo_dir": str(args.repo_dir),
        "output_dir": str(args.output_dir),
        "smoke": args.smoke,
        "web_mode": args.web_mode,
        "searxng_engines": args.searxng_engines if args.web_mode == "searxng" else None,
        "live_web_limit": args.live_web_limit,
        "live_web_attempts": len([row for row in web_trace if row.get("mode") == "searxng_llama_summary"]),
        "live_web_successes": len([row for row in web_trace if row.get("success")]),
        "blocklist_size": len(args.blocklist),
        "blocked_train_rows": sum(1 for row in train_rows if str(row.get("record_id")) in args.blocklist),
        "blocked_val_rows": sum(1 for row in val_rows if str(row.get("record_id")) in args.blocklist),
        "missing_label_filtered_train_rows": sum(
            1 for row in train_rows if str(row.get("record_id")) not in args.blocklist and not has_required_label(row)
        ),
        "missing_label_filtered_val_rows": sum(
            1 for row in val_rows if str(row.get("record_id")) not in args.blocklist and not has_required_label(row)
        ),
        "splits": {
            "train": split_report(train_items, train_meta),
            "val": split_report(val_items, val_meta),
        },
    }
    write_json(args.output_dir / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = []
    for split, payload in report["splits"].items():
        if payload["missing_images"] or payload["invalid_format"]:
            failed.append(f"{split}:missing={payload['missing_images']} invalid={payload['invalid_format']}")
    if args.require_live_web and report["live_web_successes"] <= 0:
        failed.append("live_web_successes=0")
    if failed:
        raise SystemExit("; ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
