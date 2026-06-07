from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

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
    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, timeout: int = 30, engines: str = "bing") -> None:
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
