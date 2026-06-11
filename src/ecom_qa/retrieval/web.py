from __future__ import annotations

import json
import re
import time
import hashlib
import os
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import fcntl
import requests

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenCC = None  # type: ignore[assignment]


WEB_SUMMARY_SYSTEM_ZH = (
    "你是一名有帮助的助手。你的任务是总结给定网页的主要内容，不超过五句话。"
    "摘要应覆盖网页整体关键点，而不是只总结与用户问题相关的局部内容。"
    "只能输出简体中文或英文，禁止输出繁体中文。"
    "不要输出网址、来源链接、引用编号、Markdown 或额外解释。"
)
WEB_SUMMARY_USER_ZH = (
    "如果网页内容中的任何部分有助于回答用户的问题，必须在摘要中明确保留这些信息。"
    "不要忽略相关信息，同时也要保留网页的总体结构和主要观点。"
    "摘要必须简洁、事实准确、信息充分。"
    "如果用户问题是中文，用简体中文摘要；如果用户问题是英文，可以用英文摘要。"
    "不得输出繁体中文、网址或来源链接。\n\n"
    "网页内容（已截断到上下文可处理范围）如下：\n{webpage_content}\n\n"
    "用户问题：{question}"
)
ANSWER_SYNTHESIS_SYSTEM_ZH = (
    "你是用于构建 SFT 数据的答案合成器。"
    "只能根据给定 information 回答问题；information 不支持答案时，必须只输出：无法回答，缺少相关信息。"
    "只能输出简体中文或英文，禁止输出繁体中文。"
    "不要输出网址、来源、引用编号、标签、Markdown 或推理过程。"
)
ANSWER_SYNTHESIS_USER_ZH = (
    "请根据下面的 information 回答用户问题。"
    "如果问题是中文，用简体中文回答；如果问题是英文，可以用英文回答。"
    "如果 information 中有多个说法，应给出稳妥表述，说明不是单一确定答案。"
    "答案控制在 1-3 句话内。"
    "如果无法从 information 得到答案，只输出：无法回答，缺少相关信息。\n\n"
    "用户问题：{question}\n\n"
    "{information}"
)
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9.-]*\.(?:com|org|net|cn|io|gov|edu)\b\S*",
    re.IGNORECASE,
)
_OPENCC = OpenCC("t2s") if OpenCC is not None else None
LOW_QUALITY_SUMMARY_PATTERNS = (
    "youtube footer",
    "youtube 页脚",
    "reddit verification",
    "raw pdf",
    "jbig2",
    "pdf object",
    "anubis",
    "captcha",
    "400 bad request",
    "403 forbidden",
    "网页摘要生成失败",
    "未包含",
    "未提及",
    "未涉及",
    "并未包含",
    "并未提及",
    "并未涉及",
    "无法回答",
    "无法从",
    "不能回答",
    "不包含",
    "does not contain",
    "does not mention",
    "does not include",
    "not contain",
    "not mention",
    "not include",
    "cannot answer",
    "no information about",
    "没有关于",
    "不涉及",
    "登录页",
    "shopping platform homepage",
    "购物平台主页",
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str = ""
    score: float | None = None


@dataclass(frozen=True, slots=True)
class WebSummary:
    title: str
    url: str
    snippet: str
    summary: str
    fetched_chars: int
    elapsed_seconds: float
    error: str = ""


@dataclass(frozen=True, slots=True)
class WebDocument:
    title: str
    url: str
    snippet: str
    engine: str
    content: str
    searxng_score: float
    hybrid_score: float
    rerank_score: float
    fetched_chars: int = 0
    error: str = ""


@dataclass(frozen=True, slots=True)
class WebEvidenceResult:
    query: str
    information_items: list[str]
    documents: list[WebDocument]
    timings: dict[str, float]
    params: dict[str, Any]
    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WebRerankConfig:
    candidate_k: int = 20
    top_k: int = 5
    min_score: float = 0.3
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_weight: float = 0.6
    searxng_weight: float = 0.4
    device: str = "auto"
    batch_size: int = 8
    fetch_timeout: int = 20
    max_fetch_chars: int = 12000
    max_prompt_chars_per_item: int = 1400


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.parts.append(text)


class SearXNGClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8888",
        *,
        timeout: int = 30,
        engines: str = "google,duckduckgo,qwant,wikipedia",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.engines = engines.strip()
        self.session = requests.Session()

    def healthcheck(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/config", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def search(self, query: str, *, num_results: int = 5, language: str = "auto") -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "safesearch": 1,
            "categories": "general",
        }
        if self.engines:
            params["engines"] = self.engines
        response = self.session.get(
            f"{self.base_url}/search",
            params=params,
            headers={"User-Agent": "multimodal-ecommerce-qa/1.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results") or []
        results: list[SearchResult] = []
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not url or not urlparse(url).scheme.startswith("http"):
                continue
            results.append(
                SearchResult(
                    title=str(row.get("title") or url).strip(),
                    url=url,
                    snippet=str(row.get("content") or row.get("snippet") or "").strip(),
                    engine=str(row.get("engine") or ""),
                    score=float(row["score"]) if isinstance(row.get("score"), (int, float)) else None,
                )
            )
            if len(results) >= num_results:
                break
        return results


class SearxngWebEvidenceProvider:
    def __init__(
        self,
        client: SearXNGClient,
        *,
        cache_dir: Path | None = None,
        config: WebRerankConfig | None = None,
        use_cache: bool = True,
    ) -> None:
        self.client = client
        self.config = config or WebRerankConfig()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self._cache_lock = Lock()
        self._cache_index = load_web_cache_index(cache_dir / "records.jsonl") if cache_dir and use_cache else {}
        self._embedder: BGETextEmbedder | None = None
        self._reranker: BGEWebReranker | None = None
        self._model_lock = Lock()

    def search(self, *, query: str, question: str, max_items: int | None = None, language: str = "auto") -> WebEvidenceResult:
        max_items = max(1, int(max_items or self.config.top_k))
        cache_key = self.cache_key(query=query, question=question, max_items=max_items, language=language)
        if self.use_cache:
            with self._cache_lock:
                cached = self._cache_index.get(cache_key)
            if cached and cached.get("status") == "success":
                return web_result_from_cache(cached, cache_hit=True)

        timings: dict[str, float] = {}
        errors: list[str] = []
        started = time.perf_counter()
        search_started = time.perf_counter()
        results = self.client.search(query, num_results=self.config.candidate_k, language=language)
        timings["searxng_search"] = round(time.perf_counter() - search_started, 3)

        dedupe_started = time.perf_counter()
        results = dedupe_search_results(results)[: self.config.candidate_k]
        timings["dedupe"] = round(time.perf_counter() - dedupe_started, 3)

        hybrid_started = time.perf_counter()
        ranked = self._hybrid_rank(query=query, question=question, results=results, timings=timings, errors=errors)
        timings["hybrid_score"] = round(time.perf_counter() - hybrid_started, 3)

        rerank_started = time.perf_counter()
        reranked = self._rerank(query=query, question=question, ranked=ranked, errors=errors)
        timings["rerank"] = round(time.perf_counter() - rerank_started, 3)

        selected = [item for item in reranked if item["rerank_score"] >= self.config.min_score][:max_items]
        fetch_started = time.perf_counter()
        documents = self._fetch_documents(selected)
        timings["page_fetch"] = round(time.perf_counter() - fetch_started, 3)
        timings["wall_seconds"] = round(time.perf_counter() - started, 3)

        information_items = [doc.content for doc in documents if doc.content]
        result = WebEvidenceResult(
            query=query,
            information_items=information_items,
            documents=documents,
            timings=timings,
            params=self.params(),
            cache_hit=False,
            errors=errors,
        )
        self._record_cache(cache_key=cache_key, result=result)
        return result

    def params(self) -> dict[str, Any]:
        return {
            "searxng_url": self.client.base_url,
            "searxng_engines": self.client.engines,
            "candidate_k": self.config.candidate_k,
            "top_k": self.config.top_k,
            "min_score": self.config.min_score,
            "embedding_model": self.config.embedding_model,
            "reranker_model": self.config.reranker_model,
            "embedding_weight": self.config.embedding_weight,
            "searxng_weight": self.config.searxng_weight,
            "device": self.actual_device(),
            "batch_size": self.config.batch_size,
            "fetch_timeout": self.config.fetch_timeout,
            "max_fetch_chars": self.config.max_fetch_chars,
            "max_prompt_chars_per_item": self.config.max_prompt_chars_per_item,
        }

    def cache_params(self) -> dict[str, Any]:
        params = self.params()
        params["device"] = self.config.device
        return params

    def actual_device(self) -> str:
        if self._embedder is not None:
            return self._embedder.device
        if self._reranker is not None:
            return self._reranker.device
        return self.config.device

    def cache_key(self, *, query: str, question: str, max_items: int, language: str) -> str:
        payload = {
            "query": query,
            "question": question,
            "max_items": max_items,
            "language": language,
            "params": self.cache_params(),
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        return f"web.{digest}"

    def _hybrid_rank(
        self,
        *,
        query: str,
        question: str,
        results: list[SearchResult],
        timings: dict[str, float],
        errors: list[str],
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        texts = [search_result_text(result) for result in results]
        searx_scores = normalize_scores(
            [
                float(result.score)
                if isinstance(result.score, (int, float))
                else 1.0 / (index + 1)
                for index, result in enumerate(results)
            ]
        )
        embedding_scores = searx_scores
        embedding_started = time.perf_counter()
        try:
            embedder = self._get_embedder()
            query_embedding = embedder.encode([question or query])[0]
            document_embeddings = embedder.encode(texts)
            embedding_scores = [cosine_similarity(query_embedding, embedding) for embedding in document_embeddings]
            embedding_scores = normalize_scores(embedding_scores)
        except Exception as exc:  # noqa: BLE001 - model availability should not kill generation
            errors.append(f"embedding_error={exc!r}")
        timings["embedding"] = round(time.perf_counter() - embedding_started, 3)

        ranked: list[dict[str, Any]] = []
        for result, searx_score, embedding_score in zip(results, searx_scores, embedding_scores, strict=False):
            hybrid_score = self.config.embedding_weight * embedding_score + self.config.searxng_weight * searx_score
            ranked.append(
                {
                    "result": result,
                    "text": search_result_text(result),
                    "searxng_score": round(searx_score, 6),
                    "embedding_score": round(embedding_score, 6),
                    "hybrid_score": round(hybrid_score, 6),
                    "rerank_score": round(hybrid_score, 6),
                }
            )
        ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return ranked

    def _rerank(self, *, query: str, question: str, ranked: list[dict[str, Any]], errors: list[str]) -> list[dict[str, Any]]:
        if not ranked:
            return []
        query_text = question or query
        candidates = ranked[: self.config.candidate_k]
        try:
            reranker = self._get_reranker()
            scores = reranker.score(query_text, [str(item["text"]) for item in candidates])
            for item, score in zip(candidates, scores, strict=False):
                item["rerank_score"] = round(float(score), 6)
        except Exception as exc:  # noqa: BLE001 - keep deterministic fallback available
            errors.append(f"rerank_error={exc!r}")
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidates

    def _fetch_documents(self, ranked: list[dict[str, Any]]) -> list[WebDocument]:
        documents: list[WebDocument] = []
        for item in ranked:
            result: SearchResult = item["result"]
            fetched = ""
            error = ""
            try:
                fetched = fetch_webpage_text(
                    result.url,
                    timeout=self.config.fetch_timeout,
                    max_chars=self.config.max_fetch_chars,
                )
            except Exception as exc:  # noqa: BLE001 - use snippets when pages block fetching
                error = repr(exc)
            content = normalize_training_text(fetched or result.snippet)
            if len(content) > self.config.max_prompt_chars_per_item:
                content = content[: self.config.max_prompt_chars_per_item].rsplit(" ", 1)[0].strip() or content[
                    : self.config.max_prompt_chars_per_item
                ]
            documents.append(
                WebDocument(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    engine=result.engine,
                    content=content,
                    searxng_score=float(item["searxng_score"]),
                    hybrid_score=float(item["hybrid_score"]),
                    rerank_score=float(item["rerank_score"]),
                    fetched_chars=len(fetched),
                    error=error,
                )
            )
        return documents

    def _get_embedder(self) -> "BGETextEmbedder":
        with self._model_lock:
            if self._embedder is None:
                self._embedder = BGETextEmbedder(
                    model_name=self.config.embedding_model,
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                )
            return self._embedder

    def _get_reranker(self) -> "BGEWebReranker":
        with self._model_lock:
            if self._reranker is None:
                self._reranker = BGEWebReranker(
                    model_name=self.config.reranker_model,
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                )
            return self._reranker

    def _record_cache(self, *, cache_key: str, result: WebEvidenceResult) -> None:
        if not self.use_cache or self.cache_dir is None:
            return
        row = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "success",
            "cache_key": cache_key,
            "query": result.query,
            "information_items": result.information_items,
            "documents": [web_document_to_dict(doc) for doc in result.documents],
            "timings": result.timings,
            "params": result.params,
            "errors": result.errors,
        }
        with self._cache_lock:
            self._cache_index[cache_key] = row
            append_web_jsonl_locked(self.cache_dir / "records.jsonl", row)


class BGETextEmbedder:
    def __init__(self, *, model_name: str, device: str = "auto", batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index : index + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                output = self.model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.detach().cpu().tolist())
        return vectors


class BGEWebReranker:
    def __init__(self, *, model_name: str, device: str = "auto", batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._torch = torch

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        for index in range(0, len(documents), self.batch_size):
            batch = [[query, document] for document in documents[index : index + self.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self.model(**encoded).logits.view(-1)
                batch_scores = self._torch.sigmoid(logits).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores


def normalize_training_text(text: str) -> str:
    text = str(text or "").strip()
    text = URL_PATTERN.sub("", text)
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_low_quality_summary(text: str) -> bool:
    normalized = normalize_training_text(text).lower()
    return any(pattern in normalized for pattern in LOW_QUALITY_SUMMARY_PATTERNS)


def extract_text_from_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_webpage_text(url: str, *, timeout: int = 30, max_chars: int = 30000) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            )
        },
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or "<html" in response.text[:500].lower():
        text = extract_text_from_html(response.text)
    else:
        text = response.text
    return text[:max_chars].strip()


def dedupe_search_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        parsed = urlparse(result.url)
        key = f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}".strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def search_result_text(result: SearchResult) -> str:
    return normalize_training_text(" ".join(part for part in [result.title, result.snippet] if part))


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum <= minimum:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - optional before model load
        return "cpu"


def web_document_to_dict(document: WebDocument) -> dict[str, Any]:
    return {
        "title": document.title,
        "url": document.url,
        "snippet": document.snippet,
        "engine": document.engine,
        "content": document.content,
        "searxng_score": document.searxng_score,
        "hybrid_score": document.hybrid_score,
        "rerank_score": document.rerank_score,
        "fetched_chars": document.fetched_chars,
        "error": document.error,
    }


def web_document_from_dict(payload: dict[str, Any]) -> WebDocument:
    return WebDocument(
        title=str(payload.get("title") or ""),
        url=str(payload.get("url") or ""),
        snippet=str(payload.get("snippet") or ""),
        engine=str(payload.get("engine") or ""),
        content=str(payload.get("content") or ""),
        searxng_score=float(payload.get("searxng_score") or 0.0),
        hybrid_score=float(payload.get("hybrid_score") or 0.0),
        rerank_score=float(payload.get("rerank_score") or 0.0),
        fetched_chars=int(payload.get("fetched_chars") or 0),
        error=str(payload.get("error") or ""),
    )


def web_result_from_cache(row: dict[str, Any], *, cache_hit: bool) -> WebEvidenceResult:
    documents = [web_document_from_dict(item) for item in row.get("documents") or [] if isinstance(item, dict)]
    return WebEvidenceResult(
        query=str(row.get("query") or ""),
        information_items=[str(item) for item in row.get("information_items") or []],
        documents=documents,
        timings={key: float(value) for key, value in (row.get("timings") or {}).items()},
        params=dict(row.get("params") or {}),
        cache_hit=cache_hit,
        errors=[str(item) for item in row.get("errors") or []],
    )


def load_web_cache_index(path: Path) -> dict[str, dict[str, Any]]:
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


def append_web_jsonl_locked(path: Path, row: dict[str, Any]) -> None:
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


class LlamaCppSummarizer:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8088",
        *,
        model: str = "qwen3.5-9b-summary",
        timeout: int = 180,
        max_input_chars: int = 8000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.session = requests.Session()

    def healthcheck(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code in {200, 503}
        except requests.RequestException:
            return False

    def summarize(self, *, webpage_content: str, question: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": WEB_SUMMARY_SYSTEM_ZH},
                {
                    "role": "user",
                    "content": WEB_SUMMARY_USER_ZH.format(
                        webpage_content=webpage_content[: self.max_input_chars],
                        question=question,
                    ),
                },
            ],
        }
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return normalize_training_text(str(payload["choices"][0]["message"]["content"]))

    def synthesize_answer(self, *, information: str, question: str, unable_answer: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "top_p": 0.8,
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": ANSWER_SYNTHESIS_SYSTEM_ZH},
                {
                    "role": "user",
                    "content": ANSWER_SYNTHESIS_USER_ZH.format(
                        information=information,
                        question=question,
                    ),
                },
            ],
        }
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        answer = normalize_training_text(str(payload["choices"][0]["message"]["content"]))
        return answer or unable_answer


def summarize_search_results(
    *,
    results: list[SearchResult],
    question: str,
    summarizer: LlamaCppSummarizer | None,
    max_pages: int = 3,
    fetch_timeout: int = 30,
    allow_fallback: bool = True,
) -> list[WebSummary]:
    summaries: list[WebSummary] = []
    for result in results[:max_pages]:
        started_at = time.perf_counter()
        fetched = ""
        error = ""
        try:
            fetched = fetch_webpage_text(result.url, timeout=fetch_timeout)
            if not fetched:
                raise RuntimeError("empty webpage text")
            if summarizer is None:
                summary = normalize_training_text(result.snippet or fetched[:600])
            else:
                summary = summarizer.summarize(webpage_content=fetched, question=question)
        except Exception as exc:
            error = repr(exc)
            if not allow_fallback:
                raise
            summary = normalize_training_text(result.snippet) or "网页摘要生成失败，未获得可靠信息。"
        summaries.append(
            WebSummary(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                summary=normalize_training_text(summary),
                fetched_chars=len(fetched),
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
                error=error,
            )
        )
    return summaries


def format_web_information(summaries: list[WebSummary]) -> str:
    if not summaries:
        return "<information>\n未检索到相关网页摘要。\n</information>"
    lines = ["<information>"]
    for index, item in enumerate(summaries, start=1):
        summary = normalize_training_text(item.summary)
        if summary and not is_low_quality_summary(summary):
            lines.append(f"[{index}] 摘要: {summary}")
    if len(lines) == 1:
        lines.append("未检索到相关网页摘要。")
    lines.append("</information>")
    return "\n".join(lines)
