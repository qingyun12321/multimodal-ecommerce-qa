from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecom_qa.datasets.catalog import ProductRecord, build_query_text, load_products

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenCC = None  # type: ignore[assignment]


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9.-]*\.(?:com|org|net|cn|io|gov|edu)\b\S*",
    re.IGNORECASE,
)
_OPENCC = OpenCC("t2s") if OpenCC is not None else None

__all__ = [
    "RetrievalItem",
    "LocalRAGSearch",
    "tokenize_query",
    "lexical_score",
    "build_product_snippet",
    "format_information",
]


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
        snippet = _normalize_training_text(item.snippet)
        lines.extend(
            [
                f"[{index}] 内容: {snippet}",
            ]
        )
    lines.append("</information>")
    return "\n".join(lines)


def _normalize_training_text(text: str) -> str:
    text = str(text or "").strip()
    text = URL_PATTERN.sub("", text)
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
