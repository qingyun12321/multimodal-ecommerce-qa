from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ecom_qa.common.model_client import ServerConfig, post_json
from ecom_qa.retrieval.local_catalog import (
    LocalRAGSearch,
    RetrievalItem,
    format_information,
)
from ecom_qa.tool_calls.generation import extract_tags
from ecom_qa.retrieval.web import (
    LlamaCppSummarizer,
    SearXNGClient,
    summarize_search_results,
)


UNABLE_TO_ANSWER = "无法回答，缺少相关信息。"
ROUND_TAG_NAMES = ("Think", "Answer", "Web_search")
ROUND_TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in ROUND_TAG_NAMES
}


@dataclass(slots=True)
class WebSearch:
    mode: str = "mock"
    top_k: int = 5
    base_url: str = "http://127.0.0.1:8888"
    engines: str = "google,duckduckgo,qwant,wikipedia"
    summary_base_url: str = "http://127.0.0.1:8088"
    summary_model: str = "qwen3.5-9b-summary"
    summarize_pages: int = 2
    allow_summary_fallback: bool = True

    def search(self, query: str, *, question: str | None = None) -> list[RetrievalItem]:
        if self.mode == "mock":
            return [
                RetrievalItem(
                    title=f"Web search placeholder for {query}",
                    snippet="离线 Web_search 占位结果；真实流程应使用 SearXNG 并由 Qwen3.5-9B 摘要网页。",
                    source="mock_web",
                    metadata={},
                )
            ]
        if self.mode != "searxng":
            raise ValueError(f"Unsupported web search mode: {self.mode}")

        client = SearXNGClient(self.base_url, engines=self.engines)
        results = client.search(query, num_results=self.top_k)
        summarizer = LlamaCppSummarizer(self.summary_base_url, model=self.summary_model)
        summaries = summarize_search_results(
            results=results,
            question=question or query,
            summarizer=summarizer,
            max_pages=self.summarize_pages,
            allow_fallback=self.allow_summary_fallback,
        )
        return [
            RetrievalItem(
                title=item.title,
                snippet=item.summary,
                source=item.url,
                metadata={
                    "snippet": item.snippet,
                    "fetched_chars": item.fetched_chars,
                    "elapsed_seconds": item.elapsed_seconds,
                    "error": item.error,
                },
            )
            for item in summaries
        ]


def build_after_rag_prompt(question: str) -> str:
    return (
        "你已经收到 RAG_search 返回的信息。你的目标是使用这些新信息回答原始问题："
        f"{question}\n\n"
        "查看 <information>...</information> 内的信息。必须先输出 <Think>...</Think>。"
        "如果已经可以回答，输出 <Answer>最终答案</Answer>。如果仍需要外部互联网信息，"
        "输出 <Web_search>精确查询词</Web_search>。不要输出 Markdown、JSON 或额外解释。"
    )


def build_after_web_prompt(question: str) -> str:
    return (
        "你已经收到 Web_search 返回的网页摘要。你的目标是回答原始问题："
        f"{question}\n\n"
        "查看 <information>...</information> 内的信息。必须先输出 <Think>...</Think>。"
        "如果可以回答，输出 <Answer>最终答案</Answer>。如果仍无法回答，"
        f"输出 <Answer>{UNABLE_TO_ANSWER}</Answer>。不要输出 Markdown、JSON 或额外解释。"
    )


def build_information_user_content(information: str) -> str:
    return information


def parse_round_output(raw_output: str, *, round_name: str) -> dict[str, Any]:
    tags = {
        tag: [match.group(1).strip() for match in pattern.finditer(raw_output)]
        for tag, pattern in ROUND_TAG_PATTERNS.items()
    }

    def latest(tag: str) -> str:
        values = tags.get(tag) or []
        return values[-1] if values else ""

    warnings: list[str] = []
    think = latest("Think")
    answer = latest("Answer")
    web_search = latest("Web_search")

    if not think:
        warnings.append("missing_think")
    if answer and web_search:
        warnings.append("answer_with_web_search")

    if answer:
        action = "answer"
    elif round_name == "after_rag" and web_search:
        action = "web_search"
    else:
        action = "invalid"
        warnings.append("missing_valid_action")

    return {
        "raw_output": raw_output.strip(),
        "think": think,
        "answer": answer,
        "web_search": web_search,
        "action": action,
        "parsed_tags": tags,
        "parse_warnings": warnings,
    }


def call_text_model(
    *,
    config: ServerConfig,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": config.alias,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.2,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    started_at = time.perf_counter()
    response = post_json(f"{config.base_url}/v1/chat/completions", payload)
    elapsed_seconds = time.perf_counter() - started_at
    raw_output = str(response["choices"][0]["message"]["content"]).strip()
    return {"raw_output": raw_output}, elapsed_seconds


def extract_first_round_prediction(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("prediction"), dict):
        return dict(row["prediction"])
    if "search_tool" in row or "direct_answer" in row:
        return {
            "raw_output": row.get("raw_output", ""),
            "think": row.get("think", ""),
            "answer": row.get("answer", ""),
            "direct_answer": row.get("direct_answer", ""),
            "decision_type": row.get("decision_type", ""),
            "search_tool": row.get("search_tool", "none"),
            "search_input": row.get("search_input", ""),
            "use_grounding": row.get("use_grounding", False),
            "grounding_input": row.get("grounding_input", ""),
            "tool_calls": row.get("tool_calls", []),
            "parse_warnings": row.get("parse_warnings", []),
        }
    raw_output = str(row.get("raw_output") or "")
    parsed = extract_tags(raw_output)
    return {
        "raw_output": raw_output,
        "think": (parsed.get("Think") or [""])[-1],
        "direct_answer": (parsed.get("Answer") or [""])[-1],
        "search_tool": "RAG_search" if parsed.get("RAG_search") else "Web_search" if parsed.get("Web_search") else "none",
        "search_input": ((parsed.get("RAG_search") or parsed.get("Web_search") or [""])[-1]),
        "use_grounding": bool(parsed.get("Grounding")),
        "grounding_input": (parsed.get("Grounding") or [""])[-1],
        "tool_calls": [],
        "parse_warnings": [],
    }


def run_multiturn_for_record(
    row: dict[str, Any],
    *,
    config: ServerConfig,
    rag: LocalRAGSearch,
    web: WebSearch,
    max_tokens: int,
) -> dict[str, Any]:
    question = str(row.get("query") or row.get("question") or "").strip()
    first_round = extract_first_round_prediction(row)
    turns: list[dict[str, Any]] = [
        {
            "turn": 1,
            "stage": "first_round",
            "action": first_round.get("decision_type") or "tool_call",
            "search_tool": first_round.get("search_tool", "none"),
            "search_input": first_round.get("search_input", ""),
            "use_grounding": bool(first_round.get("use_grounding")),
            "grounding_input": first_round.get("grounding_input", ""),
            "raw_output": first_round.get("raw_output", ""),
            "parse_warnings": first_round.get("parse_warnings", []),
        }
    ]

    if first_round.get("search_tool") in ("", "none", None):
        answer = str(first_round.get("direct_answer") or first_round.get("answer") or "").strip()
        return build_multiturn_record(row, question, turns, status="answered", answer=answer)

    search_tool = str(first_round["search_tool"])
    search_input = str(first_round.get("search_input") or question).strip()

    if search_tool == "RAG_search":
        rag_items = rag.search(search_input)
        rag_information = format_information(rag_items)
        raw_payload, elapsed = call_text_model(
            config=config,
            system_prompt=build_after_rag_prompt(question),
            user_content=build_information_user_content(rag_information),
            max_tokens=max_tokens,
        )
        parsed = parse_round_output(raw_payload["raw_output"], round_name="after_rag")
        turns.append(
            {
                "turn": 2,
                "stage": "after_rag",
                "tool": "RAG_search",
                "query": search_input,
                "information": rag_information,
                "retrieved_count": len(rag_items),
                "model_elapsed_seconds": round(elapsed, 3),
                "model_output": parsed,
            }
        )
        if parsed["action"] == "answer":
            return build_multiturn_record(row, question, turns, status="answered", answer=parsed["answer"])
        if parsed["action"] == "web_search":
            search_input = parsed["web_search"]
            search_tool = "Web_search"
        else:
            return build_multiturn_record(row, question, turns, status="failed_parse", answer="")

    if search_tool == "Web_search":
        web_items = web.search(search_input, question=question)
        web_information = format_information(web_items)
        raw_payload, elapsed = call_text_model(
            config=config,
            system_prompt=build_after_web_prompt(question),
            user_content=build_information_user_content(web_information),
            max_tokens=max_tokens,
        )
        parsed = parse_round_output(raw_payload["raw_output"], round_name="after_web")
        turns.append(
            {
                "turn": len(turns) + 1,
                "stage": "after_web",
                "tool": "Web_search",
                "query": search_input,
                "information": web_information,
                "retrieved_count": len(web_items),
                "model_elapsed_seconds": round(elapsed, 3),
                "model_output": parsed,
            }
        )
        if parsed["action"] == "answer":
            return build_multiturn_record(row, question, turns, status="answered", answer=parsed["answer"])
        return build_multiturn_record(row, question, turns, status="failed_parse", answer="")

    return build_multiturn_record(row, question, turns, status="unsupported_tool", answer="")


def build_multiturn_record(
    row: dict[str, Any],
    question: str,
    turns: list[dict[str, Any]],
    *,
    status: str,
    answer: str,
) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "source_dataset": row.get("source_dataset"),
        "source_record_key": row.get("source_record_key"),
        "domain": row.get("domain"),
        "query": question,
        "image_path": row.get("image_path"),
        "status": status,
        "answer": answer,
        "turn_count": len(turns),
        "turns": turns,
    }


def summarize_multiturn(records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status")) for record in records)
    turn_counts = Counter(str(record.get("turn_count")) for record in records)
    first_tool_counts = Counter(
        str((record.get("turns") or [{}])[0].get("search_tool") or "none")
        for record in records
    )
    return {
        "record_count": len(records),
        "failure_count": len(failures),
        "status_distribution": dict(status_counts),
        "turn_count_distribution": dict(turn_counts),
        "first_round_search_tool_distribution": dict(first_tool_counts),
    }
