#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from ecom_qa.common.paths import repo_root, resolve_data_root
from ecom_qa.retrieval.local_catalog import LocalRAGSearch, format_information
from ecom_qa.training.sft_codex import (
    DIRECT_SCHEMA,
    RAG_SCHEMA,
    CodexClient,
    CodexGenerationError,
    make_cache_key,
    web_schema,
)
from ecom_qa.training.sft_prompts import (
    UNABLE_TO_ANSWER,
    build_direct_fill_prompt,
    build_rag_fill_prompt,
    build_system_prompt,
    build_web_fill_prompt,
)


TAG_NAMES = ("Think", "Answer", "RAG_search", "Web_search", "Grounding", "information")
TAG_PATTERN = re.compile(r"</?(?:Think|Answer|RAG_search|Web_search|Grounding|information)\b[^>]*>", re.I)
TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
    for tag in TAG_NAMES
}
OLD_REPO_ROOTS = (
    "/home/qingyun/projects/multimodal-ecommerce-qa",
    "/workspace/repos/multimodal-ecommerce-qa",
)
DEFAULT_OUTPUT_SUBDIR = ("training", "sft", "generated", "codex_multiturn")


class JsonGenerator(Protocol):
    def generate_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        cache_key: str,
        image_paths: list[Path],
        use_search: bool = False,
    ) -> dict[str, Any]:
        ...


def clean_text(text: Any) -> str:
    value = str(text or "").strip()
    value = TAG_PATTERN.sub("", value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


class LockedJsonlAppender:
    def __init__(self, path: Path, *, key_fn: Any) -> None:
        self.path = path
        self.key_fn = key_fn
        self.lock = Lock()
        self.keys = self._load_keys()

    def _load_keys(self) -> set[str]:
        keys: set[str] = set()
        if not self.path.exists():
            return keys
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(self.key_fn(row) or "")
                if key:
                    keys.add(key)
        return keys

    def contains(self, key: str) -> bool:
        with self.lock:
            return key in self.keys

    def append(self, row: dict[str, Any], *, key: str) -> bool:
        with self.lock:
            if key in self.keys:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    self.keys.add(key)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return True


def has_image(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return str(metadata.get("qa_type") or "multimodal").strip().lower() != "text_only"


def qa_type(row: dict[str, Any]) -> str:
    return "multimodal" if has_image(row) else "text_only"


def expected_answers(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    candidates: list[Any] = [
        metadata.get("source_primary_answer"),
        row.get("direct_answer"),
        row.get("answer") if str(row.get("decision_type")) == "direct_answer" else "",
    ]
    source_answers = metadata.get("source_answers")
    if isinstance(source_answers, list):
        candidates.extend(source_answers)
    answers: list[str] = []
    for candidate in candidates:
        text = clean_text(candidate)
        if text and not text.startswith("调用工具"):
            answers.append(text)
    return list(dict.fromkeys(answers))


def resolve_repoish_path(path: Path, *, repo_dir: Path) -> str:
    resolved = path.resolve() if path.exists() else path
    try:
        return resolved.relative_to(repo_dir).as_posix()
    except ValueError:
        return str(resolved)


def image_path_for_row(row: dict[str, Any], *, repo_dir: Path, data_dir: Path) -> Path | None:
    rel = str(row.get("image_rel_path") or "").strip().lstrip("/")
    raw = str(row.get("image_path") or "").strip()
    candidates: list[Path] = []
    if rel:
        candidates.extend([data_dir / rel, data_dir / "infoseek_sample" / rel])
    if raw:
        normalized_raw = raw
        for old_root in OLD_REPO_ROOTS:
            if raw.startswith(f"{old_root}/dataset/"):
                normalized_raw = raw.replace(f"{old_root}/dataset/", f"{repo_dir}/data/", 1)
            elif raw.startswith(f"{old_root}/"):
                normalized_raw = raw.replace(f"{old_root}/", f"{repo_dir}/", 1)
        raw_path = Path(normalized_raw)
        candidates.append(raw_path if raw_path.is_absolute() else repo_dir / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return None


def image_paths_for_row(row: dict[str, Any], *, repo_dir: Path, data_dir: Path) -> tuple[list[Path], list[str]]:
    if not has_image(row):
        return [], []
    absolute_path = image_path_for_row(row, repo_dir=repo_dir, data_dir=data_dir)
    if absolute_path is None:
        return [], [""]
    return [absolute_path], [resolve_repoish_path(absolute_path, repo_dir=repo_dir)]


def user_question_content(row: dict[str, Any]) -> str:
    question = clean_text(row.get("query"))
    return f"<image>\n{question}" if has_image(row) else question


def grounding_value(row: dict[str, Any]) -> str:
    if not has_image(row) or not bool(row.get("use_grounding")):
        return "No"
    value = clean_text(row.get("grounding_input"))
    return value or "No"


def search_input(row: dict[str, Any]) -> str:
    return clean_text(row.get("search_input") or row.get("query"))


def enhanced_rag_query(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    parts = [
        row.get("search_input"),
        row.get("query"),
        metadata.get("source_title"),
        metadata.get("product_id"),
    ]
    return " ".join(clean_text(part) for part in parts if clean_text(part)).strip()


def direct_first_output(first_think: str, final_answer: str) -> str:
    return f"<Think>{clean_text(first_think)}</Think>\n<Answer>{canonical_final_answer(final_answer)}</Answer>"


def tool_first_output(tool: str, first_think: str, query: str, grounding: str) -> str:
    return (
        f"<Think>{clean_text(first_think)}</Think>\n"
        f"<{tool}>{clean_text(query)}</{tool}>\n"
        f"<Grounding>{clean_text(grounding) or 'No'}</Grounding>"
    )


def final_answer_output(after_tool_think: str, final_answer: str) -> str:
    return (
        f"<Think>{clean_text(after_tool_think)}</Think>\n"
        f"<Answer>{canonical_final_answer(final_answer)}</Answer>"
    )


def canonical_final_answer(final_answer: Any) -> str:
    answer = clean_text(final_answer)
    normalized = re.sub(r"[\s。.!！]+", "", answer).lower()
    unable_keys = {
        re.sub(r"[\s。.!！]+", "", UNABLE_TO_ANSWER).lower(),
        "unabletoanswerduetolackofrelevantinformation",
    }
    if not answer or normalized in unable_keys:
        return UNABLE_TO_ANSWER
    return answer


def format_web_information(items: list[Any]) -> str:
    lines = ["<information>"]
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = clean_text(item.get("content") or item.get("evidence") or "")
        else:
            text = clean_text(item)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"(?im)^\s*(?:标题|来源|title|source|url)\s*[:：].*$", "", text)
        text = clean_text(text)
        if text:
            lines.append(f"[{index}] {text}")
    if len(lines) == 1:
        lines.append("[1] 未找到可支持回答的网页内容。")
    lines.append("</information>")
    return "\n".join(lines)


def extract_tags(text: str) -> dict[str, list[str]]:
    return {tag: [match.group(1).strip() for match in pattern.finditer(text)] for tag, pattern in TAG_PATTERNS.items()}


def validate_assistant_output(raw: str, *, round_name: str) -> list[str]:
    tags = extract_tags(raw)
    counts = {tag: len(tags[tag]) for tag in TAG_NAMES}
    warnings: list[str] = []
    if counts["Think"] != 1:
        warnings.append(f"{round_name}:think_count={counts['Think']}")
    if counts["information"]:
        warnings.append(f"{round_name}:assistant_contains_information")
    if round_name == "first_direct":
        if counts["Answer"] != 1:
            warnings.append(f"{round_name}:answer_count={counts['Answer']}")
        if counts["RAG_search"] or counts["Web_search"] or counts["Grounding"]:
            warnings.append(f"{round_name}:unexpected_tool_tag")
    elif round_name == "first_tool":
        search_count = counts["RAG_search"] + counts["Web_search"]
        if search_count != 1:
            warnings.append(f"{round_name}:search_count={search_count}")
        if counts["Answer"]:
            warnings.append(f"{round_name}:unexpected_answer")
        if counts["Grounding"] != 1:
            warnings.append(f"{round_name}:grounding_count={counts['Grounding']}")
    else:
        if counts["Answer"] != 1:
            warnings.append(f"{round_name}:answer_count={counts['Answer']}")
        if counts["RAG_search"] or counts["Web_search"] or counts["Grounding"]:
            warnings.append(f"{round_name}:unexpected_tool_tag")
    for tag in TAG_NAMES:
        opens = len(re.findall(rf"<{tag}>", raw, flags=re.IGNORECASE))
        closes = len(re.findall(rf"</{tag}>", raw, flags=re.IGNORECASE))
        if opens != closes:
            warnings.append(f"{round_name}:unclosed_{tag}")
    return warnings


def build_item(
    row: dict[str, Any],
    *,
    repo_dir: Path,
    data_dir: Path,
    generator: JsonGenerator,
    rag: LocalRAGSearch,
    web_max_items: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answers = expected_answers(row)
    image_absolute_paths, image_output_paths = image_paths_for_row(row, repo_dir=repo_dir, data_dir=data_dir)
    codex_calls: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": build_system_prompt(has_image_input=has_image(row), domain=str(row.get("domain") or ""))},
        {"role": "user", "content": user_question_content(row)},
    ]
    metadata: dict[str, Any] = {
        "record_id": row.get("record_id"),
        "domain": row.get("domain"),
        "qa_type": qa_type(row),
        "decision_type": row.get("decision_type"),
        "search_tool": row.get("search_tool") or "none",
        "use_grounding": bool(row.get("use_grounding")),
        "image_paths": image_output_paths,
        "image_exists": (not has_image(row)) or (bool(image_absolute_paths) and all(path.is_file() for path in image_absolute_paths)),
        "codex_failure": "",
        "warnings": [],
    }

    def timed_generate(
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        cache_kind: str,
        use_search: bool,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            payload = generator.generate_json(
                prompt=prompt,
                schema_name=schema_name,
                schema=schema,
                cache_key=make_cache_key(record_id=str(row.get("record_id")), kind=cache_kind, prompt=prompt),
                image_paths=image_absolute_paths,
                use_search=use_search,
            )
        except Exception as exc:
            codex_calls.append(
                {
                    "schema_name": schema_name,
                    "use_search": use_search,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                    "status": "failure",
                    "error": repr(exc),
                }
            )
            raise
        codex_calls.append(
            {
                "schema_name": schema_name,
                "use_search": use_search,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "status": "success",
            }
        )
        return payload

    try:
        if str(row.get("decision_type")) == "direct_answer" or str(row.get("search_tool") or "none") == "none":
            prompt = build_direct_fill_prompt(record=row, answers=answers)
            payload = timed_generate(
                prompt=prompt,
                schema_name="direct",
                schema=DIRECT_SCHEMA,
                use_search=False,
                cache_kind="direct",
            )
            first_output = direct_first_output(payload["first_think"], payload["final_answer"])
            messages.append({"role": "assistant", "content": first_output})
            metadata["warnings"].extend(validate_assistant_output(first_output, round_name="first_direct"))
            metadata["turn_count"] = 1
            metadata["has_information"] = False
            metadata["sample_kind"] = "direct"
        elif str(row.get("search_tool")) == "RAG_search":
            rag_items = rag.search(enhanced_rag_query(row))
            information = format_information(rag_items)
            prompt = build_rag_fill_prompt(record=row, information=information, answers=answers)
            payload = timed_generate(
                prompt=prompt,
                schema_name="rag",
                schema=RAG_SCHEMA,
                use_search=False,
                cache_kind="rag",
            )
            first_output = tool_first_output("RAG_search", payload["first_think"], search_input(row), grounding_value(row))
            metadata["warnings"].extend(validate_assistant_output(first_output, round_name="first_tool"))
            metadata["has_information"] = True
            metadata["rag_retrieved_count"] = len(rag_items)
            action = clean_text(payload.get("next_action")) or "Web_search"
            final_answer = canonical_final_answer(payload.get("final_answer"))
            if action == "Answer" and final_answer == UNABLE_TO_ANSWER:
                action = "Web_search"
            metadata["rag_next_action"] = action
            messages.extend(
                [
                    {"role": "assistant", "content": first_output},
                    {"role": "user", "content": information},
                ]
            )
            if action == "Answer":
                final_output = final_answer_output(payload["after_tool_think"], final_answer)
                messages.append({"role": "assistant", "content": final_output})
                metadata["warnings"].extend(validate_assistant_output(final_output, round_name="after_tool"))
                metadata["turn_count"] = 2
                metadata["sample_kind"] = "rag_answered"
                metadata["retrieved_count"] = len(rag_items)
            else:
                web_query = clean_text(payload.get("web_search_query")) or search_input(row) or enhanced_rag_query(row)
                web_prompt = build_web_fill_prompt(
                    record=row,
                    answers=[],
                    max_items=web_max_items,
                    search_query=web_query,
                    previous_information=information,
                    include_first_think=False,
                )
                web_payload = timed_generate(
                    prompt=web_prompt,
                    schema_name="web_after_rag",
                    schema=web_schema(max_items=web_max_items, include_first_think=False),
                    use_search=True,
                    cache_kind="rag_web",
                )
                web_items = list(web_payload["information_items"])[:web_max_items]
                web_information = format_web_information(web_items)
                web_call_output = tool_first_output("Web_search", payload["after_tool_think"], web_query, grounding_value(row))
                final_output = final_answer_output(web_payload["after_tool_think"], web_payload["final_answer"])
                messages.extend(
                    [
                        {"role": "assistant", "content": web_call_output},
                        {"role": "user", "content": web_information},
                        {"role": "assistant", "content": final_output},
                    ]
                )
                metadata["warnings"].extend(validate_assistant_output(web_call_output, round_name="first_tool"))
                metadata["warnings"].extend(validate_assistant_output(final_output, round_name="after_tool"))
                metadata["turn_count"] = 3
                metadata["sample_kind"] = "rag_to_web"
                metadata["web_information_count"] = len(web_items)
                metadata["retrieved_count"] = len(rag_items) + len(web_items)
        elif str(row.get("search_tool")) == "Web_search":
            prompt = build_web_fill_prompt(
                record=row,
                answers=answers,
                max_items=web_max_items,
                search_query=search_input(row),
                include_first_think=True,
            )
            payload = timed_generate(
                prompt=prompt,
                schema_name="web",
                schema=web_schema(max_items=web_max_items, include_first_think=True),
                use_search=True,
                cache_kind="web",
            )
            web_items = list(payload["information_items"])[:web_max_items]
            information = format_web_information(web_items)
            first_output = tool_first_output("Web_search", payload["first_think"], search_input(row), grounding_value(row))
            final_output = final_answer_output(payload["after_tool_think"], payload["final_answer"])
            messages.extend(
                [
                    {"role": "assistant", "content": first_output},
                    {"role": "user", "content": information},
                    {"role": "assistant", "content": final_output},
                ]
            )
            metadata["warnings"].extend(validate_assistant_output(first_output, round_name="first_tool"))
            metadata["warnings"].extend(validate_assistant_output(final_output, round_name="after_tool"))
            metadata["turn_count"] = 2
            metadata["has_information"] = True
            metadata["sample_kind"] = "web_first"
            metadata["web_information_count"] = len(web_items)
            metadata["retrieved_count"] = len(web_items)
        else:
            raise ValueError(f"Unsupported search_tool: {row.get('search_tool')!r}")
    except (CodexGenerationError, KeyError, ValueError) as exc:
        metadata["codex_failure"] = repr(exc)
        metadata["turn_count"] = 0
        metadata["has_information"] = False
        raise

    metadata["codex_calls"] = codex_calls
    metadata["codex_call_count"] = len(codex_calls)
    metadata["codex_schema_names"] = [str(call.get("schema_name")) for call in codex_calls]
    metadata["codex_generation_seconds"] = round(
        sum(float(call.get("elapsed_seconds") or 0.0) for call in codex_calls),
        3,
    )
    item = {
        "messages": messages,
        "record_id": row.get("record_id"),
        "images": image_output_paths,
    }
    return item, metadata


def smoke_select(rows: list[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    required = [
        lambda row: row.get("search_tool") == "Web_search",
        lambda row: row.get("search_tool") == "RAG_search",
        lambda row: row.get("decision_type") == "direct_answer",
        lambda row: bool(row.get("use_grounding")),
        lambda row: qa_type(row) == "text_only",
        lambda row: qa_type(row) == "multimodal",
    ]
    for predicate in required:
        if len(selected) >= size:
            break
        candidates = [row for row in rows if predicate(row) and str(row.get("record_id")) not in used]
        if not candidates:
            continue
        row = rng.choice(candidates)
        selected.append(row)
        used.add(str(row.get("record_id")))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                qa_type(row),
                str(row.get("decision_type") or ""),
                str(row.get("search_tool") or "none"),
                str(bool(row.get("use_grounding"))).lower(),
                str(row.get("domain") or ""),
            ]
        )
        buckets[key].append(row)
    for key in sorted(buckets):
        if len(selected) >= size:
            break
        candidates = list(buckets[key])
        rng.shuffle(candidates)
        for candidate in candidates:
            record_id = str(candidate.get("record_id"))
            if record_id not in used:
                selected.append(candidate)
                used.add(record_id)
                break
    remaining = [row for row in rows if str(row.get("record_id")) not in used]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, size - len(selected))])
    rng.shuffle(selected)
    return selected[:size]


def output_key(*, split: str, row: dict[str, Any]) -> str:
    kind = str(row.get("search_tool") or row.get("decision_type") or "unknown")
    return f"{split}:{row.get('record_id')}:{kind}"


def split_output_path(args: argparse.Namespace, split: str) -> Path:
    prefix = "smoke_" if args.smoke else ""
    return args.output_dir / f"{prefix}{split}.ms_swift.jsonl"


def metadata_output_path(args: argparse.Namespace) -> Path:
    prefix = "smoke_" if args.smoke else ""
    return args.output_dir / f"{prefix}metadata.jsonl"


def failures_output_path(args: argparse.Namespace) -> Path:
    prefix = "smoke_" if args.smoke else ""
    return args.output_dir / f"{prefix}codex_failures.jsonl"


def review_output_path(args: argparse.Namespace) -> Path:
    prefix = "smoke_" if args.smoke else ""
    return args.output_dir / f"{prefix}review_samples.jsonl"


def clear_generation_outputs(args: argparse.Namespace) -> None:
    prefix = "smoke_" if args.smoke else ""
    names = [
        f"{prefix}train.ms_swift.jsonl",
        f"{prefix}val.ms_swift.jsonl",
        f"{prefix}metadata.jsonl",
        f"{prefix}review_samples.jsonl",
        f"{prefix}codex_failures.jsonl",
        "validation_report.json",
    ]
    for name in names:
        path = args.output_dir / name
        for candidate in (path, path.with_suffix(path.suffix + ".lock")):
            if candidate.exists():
                candidate.unlink()


def build_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    args: argparse.Namespace,
    generator: JsonGenerator,
    rag: LocalRAGSearch,
) -> dict[str, Any]:
    size = args.smoke_train_size if split == "train" else args.smoke_val_size
    selected = smoke_select(rows, size=size, seed=args.seed + (0 if split == "train" else 1)) if args.smoke else rows
    item_appender = LockedJsonlAppender(
        split_output_path(args, split),
        key_fn=lambda row: f"{split}:{row.get('record_id')}",
    )
    metadata_appender = LockedJsonlAppender(
        metadata_output_path(args),
        key_fn=lambda row: str(row.get("output_key") or ""),
    )
    failure_appender = LockedJsonlAppender(
        failures_output_path(args),
        key_fn=lambda row: str(row.get("output_key") or ""),
    )

    def build_one(row: dict[str, Any]) -> dict[str, Any]:
        key = output_key(split=split, row=row)
        if item_appender.contains(f"{split}:{row.get('record_id')}"):
            return {"status": "skipped_existing", "output_key": key}
        started_at = time.perf_counter()
        try:
            item, meta = build_item(
                row,
                repo_dir=args.repo_dir,
                data_dir=args.data_dir,
                generator=generator,
                rag=rag,
                web_max_items=args.web_max_items,
            )
            meta["split"] = split
            meta["output_key"] = key
            meta["generation_seconds"] = round(time.perf_counter() - started_at, 3)
            item_appender.append(item, key=f"{split}:{row.get('record_id')}")
            metadata_appender.append(meta, key=key)
            return {"status": "generated", "output_key": key}
        except Exception as exc:  # noqa: BLE001 - failures are reported after all rows are attempted
            failure = {
                "record_id": row.get("record_id"),
                "split": split,
                "output_key": key,
                "codex_failure": repr(exc),
                "decision_type": row.get("decision_type"),
                "search_tool": row.get("search_tool"),
                "generation_seconds": round(time.perf_counter() - started_at, 3),
            }
            failure_appender.append(failure, key=key)
            return {"status": "failed", "output_key": key}

    max_workers = max(1, int(args.codex_workers))
    if max_workers == 1:
        statuses = Counter()
        for row in selected:
            statuses[build_one(row)["status"]] += 1
        return {"selected": len(selected), "statuses": dict(statuses)}

    statuses = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_one, row) for row in selected]
        for future in as_completed(futures):
            statuses[future.result()["status"]] += 1
    return {"selected": len(selected), "statuses": dict(statuses)}


def split_report(items: list[dict[str, Any]], metas: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [meta for meta in metas if meta.get("warnings")]
    kind_seconds: dict[str, float] = defaultdict(float)
    for meta in metas:
        kind_seconds[str(meta.get("sample_kind") or "unknown")] += float(meta.get("generation_seconds") or 0.0)
    failure_text = "\n".join(str(failure.get("codex_failure") or "") for failure in failures).lower()
    return {
        "rows": len(items),
        "unique_record_ids": len({str(item["record_id"]) for item in items}),
        "missing_images": sum(1 for meta in metas if not meta.get("image_exists", True)),
        "invalid_format": len(invalid),
        "codex_failures": len(failures),
        "timeout_failures": failure_text.count("timed out"),
        "json_parse_failures": failure_text.count("json"),
        "qa_type_counts": dict(Counter(str(meta.get("qa_type")) for meta in metas)),
        "decision_type_counts": dict(Counter(str(meta.get("decision_type")) for meta in metas)),
        "search_tool_counts": dict(Counter(str(meta.get("search_tool")) for meta in metas)),
        "sample_kind_counts": dict(Counter(str(meta.get("sample_kind")) for meta in metas)),
        "turn_count_counts": dict(Counter(str(meta.get("turn_count")) for meta in metas)),
        "rag_next_action_counts": dict(Counter(str(meta.get("rag_next_action")) for meta in metas if meta.get("rag_next_action"))),
        "information_rows": sum(1 for meta in metas if meta.get("has_information")),
        "grounding_rows": sum(1 for meta in metas if meta.get("use_grounding")),
        "empty_search_result_rows": sum(
            1 for meta in metas if meta.get("has_information") and int(meta.get("retrieved_count") or 0) == 0
        ),
        "codex_call_count": sum(int(meta.get("codex_call_count") or 0) for meta in metas),
        "codex_generation_seconds_total": round(sum(float(meta.get("codex_generation_seconds") or 0.0) for meta in metas), 3),
        "generation_seconds_by_kind": {key: round(value, 3) for key, value in sorted(kind_seconds.items())},
        "examples_with_warnings": invalid[:10],
        "failure_examples": failures[:10],
    }


def default_output_dir(repo_dir: Path) -> Path:
    env_value = os.environ.get("SFT_DATA_DIR")
    if env_value:
        path = Path(env_value)
        return path if path.is_absolute() else repo_dir / path
    return resolve_data_root(repo_dir).joinpath(*DEFAULT_OUTPUT_SUBDIR)


def parse_worker_fallbacks(value: str, *, requested: int) -> list[int]:
    workers: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        worker = max(1, int(raw))
        if worker <= requested and worker not in workers:
            workers.append(worker)
    if requested not in workers:
        workers.insert(0, max(1, requested))
    return workers


def collect_outputs(args: argparse.Namespace) -> dict[str, Any]:
    train_items = read_jsonl_if_exists(split_output_path(args, "train"))
    val_items = read_jsonl_if_exists(split_output_path(args, "val"))
    metadata_rows = read_jsonl_if_exists(metadata_output_path(args))
    failure_rows = read_jsonl_if_exists(failures_output_path(args))
    succeeded_keys = {str(row.get("output_key") or "") for row in metadata_rows if row.get("output_key")}
    active_failures = [row for row in failure_rows if str(row.get("output_key") or "") not in succeeded_keys]
    return {
        "train_items": train_items,
        "val_items": val_items,
        "train_meta": [row for row in metadata_rows if row.get("split") == "train"],
        "val_meta": [row for row in metadata_rows if row.get("split") == "val"],
        "train_failures": [row for row in active_failures if row.get("split") == "train"],
        "val_failures": [row for row in active_failures if row.get("split") == "val"],
    }


def write_review_samples(args: argparse.Namespace) -> None:
    samples = read_jsonl_if_exists(split_output_path(args, "train"))[:20] + read_jsonl_if_exists(split_output_path(args, "val"))[:20]
    write_jsonl(review_output_path(args), samples)


def estimate_seconds_for_rows(*, elapsed_seconds: float, generated_rows: int, target_rows: int) -> float:
    if generated_rows <= 0:
        return 0.0
    return elapsed_seconds / generated_rows * target_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build script-rendered multi-turn SFT JSONL for ms-swift using Codex content filling.")
    parser.add_argument("--train-source", type=Path, default=None)
    parser.add_argument("--val-source", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-size", type=int, default=32)
    parser.add_argument("--smoke-val-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default="gpt-5.3-codex-spark")
    parser.add_argument("--codex-reasoning-effort", default="low")
    parser.add_argument("--codex-workers", type=int, default=5)
    parser.add_argument("--worker-fallbacks", default="5,3,1")
    parser.add_argument("--web-max-items", type=int, default=5)
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-codex-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--keep-codex-debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_dir = repo_root()
    args.data_dir = resolve_data_root(args.repo_dir)
    args.output_dir = args.output_dir or default_output_dir(args.repo_dir)
    if not args.output_dir.is_absolute():
        args.output_dir = args.repo_dir / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_source = args.train_source or args.data_dir / "tool_call" / "tool_call_records_train.jsonl"
    val_source = args.val_source or args.data_dir / "tool_call" / "tool_call_records_test.jsonl"
    if not train_source.is_absolute():
        train_source = args.repo_dir / train_source
    if not val_source.is_absolute():
        val_source = args.repo_dir / val_source

    train_rows = read_jsonl(train_source)
    val_rows = read_jsonl(val_source)
    args.web_max_items = max(1, int(args.web_max_items))
    requested_workers = max(1, int(args.codex_workers))
    worker_fallbacks = parse_worker_fallbacks(args.worker_fallbacks, requested=requested_workers)

    if args.clear_output:
        clear_generation_outputs(args)
    if args.clear_cache:
        shutil.rmtree(args.output_dir / "codex_cache", ignore_errors=True)

    rag = LocalRAGSearch.from_dataset(args.data_dir, top_k=5)
    generator = CodexClient(
        repo_dir=args.repo_dir,
        cache_dir=args.output_dir / "codex_cache",
        codex_bin=args.codex_bin,
        model=args.codex_model,
        reasoning_effort=args.codex_reasoning_effort,
        timeout_seconds=args.codex_timeout_seconds,
        max_retries=args.max_retries,
        use_cache=not args.no_codex_cache,
        keep_debug=args.keep_codex_debug,
    )

    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    overall_started = time.perf_counter()
    worker_attempts: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for attempt_index, worker_count in enumerate(worker_fallbacks, start=1):
        args.codex_workers = worker_count
        if attempt_index > 1:
            clear_generation_outputs(args)
        attempt_started = time.perf_counter()
        train_stats = build_split(train_rows, split="train", args=args, generator=generator, rag=rag)
        val_stats = build_split(val_rows, split="val", args=args, generator=generator, rag=rag)
        write_review_samples(args)
        outputs = collect_outputs(args)
        train_report = split_report(outputs["train_items"], outputs["train_meta"], outputs["train_failures"])
        val_report = split_report(outputs["val_items"], outputs["val_meta"], outputs["val_failures"])
        elapsed_seconds = round(time.perf_counter() - overall_started, 3)
        generated_rows = train_report["rows"] + val_report["rows"]
        codex_seconds_total = round(
            train_report["codex_generation_seconds_total"] + val_report["codex_generation_seconds_total"],
            3,
        )
        codex_call_count = train_report["codex_call_count"] + val_report["codex_call_count"]
        estimated_seconds_10k = estimate_seconds_for_rows(
            elapsed_seconds=elapsed_seconds,
            generated_rows=generated_rows,
            target_rows=10000,
        )
        worker_attempt = {
            "attempt": attempt_index,
            "codex_workers": worker_count,
            "elapsed_seconds": round(time.perf_counter() - attempt_started, 3),
            "train_statuses": train_stats["statuses"],
            "val_statuses": val_stats["statuses"],
            "codex_failures": train_report["codex_failures"] + val_report["codex_failures"],
        }
        worker_attempts.append(worker_attempt)
        report = {
            "started_at": started_at_iso,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "repo_dir": str(args.repo_dir),
            "data_dir": resolve_repoish_path(args.data_dir, repo_dir=args.repo_dir),
            "output_dir": resolve_repoish_path(args.output_dir, repo_dir=args.repo_dir),
            "train_source": resolve_repoish_path(train_source, repo_dir=args.repo_dir),
            "val_source": resolve_repoish_path(val_source, repo_dir=args.repo_dir),
            "smoke": args.smoke,
            "web_max_items": args.web_max_items,
            "clear_cache": bool(args.clear_cache),
            "clear_output": bool(args.clear_output),
            "codex_cache_enabled": not bool(args.no_codex_cache),
            "keep_codex_debug": bool(args.keep_codex_debug),
            "codex_model": args.codex_model,
            "codex_reasoning_effort": args.codex_reasoning_effort,
            "codex_workers_requested": requested_workers,
            "codex_workers_effective": worker_count,
            "worker_fallbacks": worker_fallbacks,
            "worker_fallback_attempts": worker_attempts,
            "codex_cache": resolve_repoish_path(args.output_dir / "codex_cache", repo_dir=args.repo_dir),
            "codex_generation_seconds_total": codex_seconds_total,
            "codex_generation_seconds_avg": round(codex_seconds_total / codex_call_count, 3) if codex_call_count else 0.0,
            "records_per_minute": round(generated_rows / elapsed_seconds * 60, 3) if elapsed_seconds else 0.0,
            "estimated_seconds_for_10k": round(estimated_seconds_10k, 3),
            "estimated_hours_for_10k": round(estimated_seconds_10k / 3600, 3),
            "splits": {
                "train": train_report,
                "val": val_report,
            },
        }
        write_json(args.output_dir / "validation_report.json", report)
        if worker_attempt["codex_failures"] == 0:
            break
        if attempt_index < len(worker_fallbacks):
            print(
                f"worker={worker_count} produced {worker_attempt['codex_failures']} Codex failures; retrying with worker={worker_fallbacks[attempt_index]}",
                flush=True,
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = []
    for split, payload in report["splits"].items():
        if payload["missing_images"] or payload["invalid_format"] or payload["codex_failures"]:
            failed.append(
                f"{split}:missing={payload['missing_images']} invalid={payload['invalid_format']} codex={payload['codex_failures']}"
            )
    if failed:
        raise SystemExit("; ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
