from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecom_qa.data.catalog import ProductRecord, build_query_text, load_products
from ecom_qa.data.model_client import ServerConfig, post_json
from ecom_qa.data.tool_call_generation import extract_tags
from ecom_qa.retrieval.web import SerpAPIClient, SerpAPISearchConfig


UNABLE_TO_ANSWER = "Unable to answer due to lack of relevant information"
ROUND_TAG_NAMES = ("Think", "Answer", "Web_search")
ROUND_TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in ROUND_TAG_NAMES
}


@dataclass(frozen=True, slots=True)
class RetrievalItem:
    title: str
    snippet: str
    source: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class LocalRAGSearch:
    products: list[ProductRecord]
    indexed_texts: list[str]
    top_k: int = 5

    @classmethod
    def from_dataset(cls, dataset_dir: Path, *, top_k: int = 5) -> LocalRAGSearch:
        products = load_products(dataset_dir)
        indexed_texts = [build_query_text(product, mode="rich_zh") for product in products]
        return cls(products=products, indexed_texts=indexed_texts, top_k=top_k)

    def search(self, query: str) -> list[RetrievalItem]:
        query_terms = tokenize_query(query)
        scored: list[tuple[float, int]] = []
        for index, text in enumerate(self.indexed_texts):
            score = lexical_score(query, query_terms, text)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return []
        items: list[RetrievalItem] = []
        for _, index in scored[: self.top_k]:
            product = self.products[index]
            items.append(
                RetrievalItem(
                    title=product.title,
                    snippet=build_product_snippet(product),
                    source="local_catalog",
                    metadata={"product_id": product.id, "image": product.image},
                )
            )
        return items


@dataclass(slots=True)
class WebSearch:
    mode: str = "mock"
    top_k: int = 5

    def search(self, query: str) -> list[RetrievalItem]:
        if self.mode == "serpapi":
            client = SerpAPIClient(
                SerpAPISearchConfig(
                    api_key=os.environ.get("SERPAPI_API_KEY"),
                )
            )
            payload = client.text_search(query=query, num=self.top_k)
            rows = payload.get("organic_results") or []
            return [
                RetrievalItem(
                    title=str(row.get("title") or ""),
                    snippet=str(row.get("snippet") or row.get("link") or ""),
                    source=str(row.get("link") or "serpapi"),
                    metadata={"position": row.get("position")},
                )
                for row in rows[: self.top_k]
            ]
        if self.mode != "mock":
            raise ValueError(f"Unsupported web search mode: {self.mode}")
        return [
            RetrievalItem(
                title=f"Web search placeholder for {query}",
                snippet=(
                    "当前为离线 Web_search 占位结果。正式评测如需真实互联网证据，"
                    "请使用 --web-mode serpapi 并设置 SERPAPI_API_KEY。"
                ),
                source="mock_web",
                metadata={},
            )
        ]


def tokenize_query(query: str) -> set[str]:
    query = query.lower()
    terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", query))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", query):
        terms.update(chunk[index : index + 2] for index in range(max(0, len(chunk) - 1)))
    return {term for term in terms if term.strip()}


def lexical_score(query: str, query_terms: set[str], text: str) -> float:
    haystack = text.lower()
    score = 0.0
    if query.strip() and query.lower() in haystack:
        score += 10.0
    for term in query_terms:
        if term in haystack:
            score += 1.0 + min(haystack.count(term), 3) * 0.2
    return score


def build_product_snippet(product: ProductRecord) -> str:
    fields = [
        f"商品ID: {product.id}",
        f"类目: {product.category} / {product.subcategory}",
        f"品牌: {product.brand}",
        f"标题: {product.title}",
        f"价格: {product.price:.2f}",
        f"颜色: {'、'.join(product.colors)}" if product.colors else "",
        f"规格: {'、'.join(product.sizes)}" if product.sizes else "",
        f"参数: {json.dumps(product.parameters, ensure_ascii=False)}" if product.parameters else "",
        f"店铺: {product.shop_name}",
        f"评分: {product.rating}",
        f"售后: {json.dumps(product.after_sales, ensure_ascii=False)}" if product.after_sales else "",
        f"描述: {product.description}",
    ]
    return "\n".join(field for field in fields if field)


def format_information(items: list[RetrievalItem]) -> str:
    if not items:
        return "<information>\n未检索到相关信息。\n</information>"
    lines = ["<information>"]
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"[{index}] 标题: {item.title}",
                f"来源: {item.source}",
                f"内容: {item.snippet}",
            ]
        )
    lines.append("</information>")
    return "\n".join(lines)


def build_after_rag_prompt(question: str) -> str:
    return (
        "你已经收到 RAG_search 返回的信息。你的目标是使用这些新信息回答原始问题："
        f"{question}\n\n"
        "第 1 步：分析结果。\n"
        "查看 <information>...</information> 标签内提供的信息，综合你对问题中视觉元素和检索结果的理解。\n\n"
        "第 2 步：规划下一步动作。\n"
        "必须先把简洁思考过程写在 <Think>...</Think> 标签内，然后从以下动作中选择一个：\n\n"
        "Action 1：直接回答。\n"
        "如果 RAG_search 结果已经帮助你识别视觉元素，并且你能够结合检索信息和自身知识确信可以回答问题，"
        "请把最终简洁答案写在 <Answer>...</Answer> 标签内。\n"
        "输出格式：\n"
        "<Think>推理过程</Think>\n"
        "<Answer>最终答案</Answer>\n\n"
        "Action 2：使用 Web_search。\n"
        "如果 RAG_search 结果帮助你识别了视觉元素，但仍需要更具体的信息才能回答问题，请调用 Web_search。"
        "请基于 RAG_search 结果构造精确查询词，并写在 <Web_search>...</Web_search> 标签内。\n"
        "输出格式：\n"
        "<Think>推理过程</Think>\n"
        "<Web_search>精确查询词</Web_search>\n\n"
        "本轮不要输出 Markdown，不要输出 JSON，不要输出额外解释。所有标签必须完整闭合。"
    )


def build_after_web_prompt(question: str) -> str:
    return (
        "你已经收到 Web_search 返回的结果。你的目标是分析这些新信息，并决定回答原始问题的下一步："
        f"{question}\n\n"
        "第 1 步：分析结果。\n"
        "查看 <information>...</information> 标签内提供的新信息，将其与已有信息和回答问题仍需要的信息进行比较。\n\n"
        "第 2 步：规划下一步动作。\n"
        "必须先把简洁思考过程写在 <Think>...</Think> 标签内，然后从以下动作中选择一个：\n\n"
        "Action 1：直接回答。\n"
        "如果你现在已经收集到所有必要信息，请把最终简洁答案写在 <Answer>...</Answer> 标签内。\n"
        "输出格式：\n"
        "<Think>推理过程</Think>\n"
        "<Answer>最终答案</Answer>\n\n"
        "Action 2：放弃回答。\n"
        f"如果仍然无法回答，输出固定文本：{UNABLE_TO_ANSWER}\n"
        "输出格式：\n"
        "<Think>推理过程</Think>\n"
        f"{UNABLE_TO_ANSWER}\n\n"
        "本轮不要输出 Markdown，不要输出 JSON，不要输出额外解释。所有标签必须完整闭合。"
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
    unable = UNABLE_TO_ANSWER.lower() in raw_output.lower()

    if not think:
        warnings.append("missing_think")
    if answer and web_search:
        warnings.append("answer_with_web_search")

    action = "invalid"
    if answer:
        action = "answer"
    elif round_name == "after_rag" and web_search:
        action = "web_search"
    elif round_name == "after_web" and unable:
        action = "give_up"
    else:
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
        web_items = web.search(search_input)
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
        if parsed["action"] == "give_up":
            return build_multiturn_record(row, question, turns, status="give_up", answer=UNABLE_TO_ANSWER)
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
