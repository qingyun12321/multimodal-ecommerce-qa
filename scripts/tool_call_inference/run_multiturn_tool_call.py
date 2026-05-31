from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_qa.data.model_client import ServerConfig, run_llama_server, write_json
from ecom_qa.data.multiturn_tool_call import (
    LocalRAGSearch,
    WebSearch,
    build_after_rag_prompt,
    build_after_web_prompt,
    run_multiturn_for_record,
    summarize_multiturn,
)


DEFAULT_INPUT = (
    ROOT
    / "dataset/tool_call_inference/generated/qwen3_vl_tool_call_full/qwen3-vl-8b/predictions.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "dataset/tool_call_multiturn/generated"
DEFAULT_RUN_NAME = "qwen3_vl_8b_multiturn"
DEFAULT_CTX_SIZE = 16384
DEFAULT_PARALLEL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-turn tool-call workflow after first-round tool-call inference."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--web-top-k", type=int, default=5)
    parser.add_argument("--web-mode", choices=("mock", "serpapi"), default="mock")
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(str(row.get("record_id")))
    return completed


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


def write_prompt_reference(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_after_rag.txt").write_text(
        build_after_rag_prompt("{question}") + "\n\n用户输入：\n<information>...</information>\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_after_web.txt").write_text(
        build_after_web_prompt("{question}") + "\n\n用户输入：\n<information>...</information>\n",
        encoding="utf-8",
    )


def select_rows(rows: list[dict[str, Any]], *, offset: int, limit: int | None) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("--offset must be non-negative.")
    selected = rows[offset:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0.")
        selected = selected[:limit]
    return selected


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    records_path = output_dir / "multiturn_records.jsonl"
    failures_path = output_dir / "failed_rows.jsonl"
    summary_path = output_dir / "run_summary.json"
    log_path = output_dir / "llama_server.log"
    prompt_dir = output_dir / "prompts"

    if not args.resume:
        if records_path.exists():
            records_path.unlink()
        if failures_path.exists():
            failures_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    rows = select_rows(load_jsonl(args.input), offset=args.offset, limit=args.limit)
    completed_ids = load_completed_ids(records_path) if args.resume else set()
    rows = [row for row in rows if str(row.get("record_id")) not in completed_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_reference(prompt_dir)
    rag = LocalRAGSearch.from_dataset(args.dataset_dir, top_k=args.rag_top_k)
    web = WebSearch(mode=args.web_mode, top_k=args.web_top_k)
    config = ServerConfig(
        port=args.port,
        ctx_size=args.ctx_size,
        threads=args.threads,
        parallel=args.parallel,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
    )

    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Rows to process: {len(rows)} (resume skipped {len(completed_ids)})")
    print(f"llama-server: ctx={args.ctx_size}, parallel={args.parallel}, port={args.port}")
    print(f"RAG top_k={args.rag_top_k}; Web mode={args.web_mode}, top_k={args.web_top_k}")

    started_at = time.perf_counter()
    if not rows:
        generated_records = load_jsonl(records_path) if records_path.exists() else []
        persisted_failures = load_jsonl(failures_path) if failures_path.exists() else []
        summary = summarize_multiturn(generated_records, persisted_failures)
        summary.update(
            {
                "input": str(args.input),
                "output_dir": str(output_dir),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "ctx_size": args.ctx_size,
                "parallel": args.parallel,
                "max_tokens": args.max_tokens,
                "rag_top_k": args.rag_top_k,
                "web_mode": args.web_mode,
                "web_top_k": args.web_top_k,
            }
        )
        write_json(summary_path, summary)
        print(f"Summary: {summary_path}")
        return

    succeeded = 0
    failed = 0
    completed = 0
    failures: list[dict[str, Any]] = []

    with run_llama_server(config, log_path=log_path) as startup_seconds:
        print(f"llama-server ready in {startup_seconds:.1f}s")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            pending_rows = iter(rows)
            futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}

            def submit_next() -> None:
                try:
                    row = next(pending_rows)
                except StopIteration:
                    return
                future = executor.submit(
                    run_multiturn_for_record,
                    row,
                    config=config,
                    rag=rag,
                    web=web,
                    max_tokens=args.max_tokens,
                )
                futures[future] = row

            for _ in range(args.parallel):
                submit_next()

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row = futures.pop(future)
                    try:
                        record = future.result()
                        append_jsonl(records_path, record)
                        succeeded += 1
                    except Exception as exc:
                        failure = {
                            "record_id": row.get("record_id"),
                            "query": row.get("query") or row.get("question"),
                            "error": repr(exc),
                        }
                        append_jsonl(failures_path, failure)
                        failures.append(failure)
                        failed += 1
                    completed += 1
                    render_progress(completed, len(rows), succeeded, failed)
                    submit_next()

    if rows:
        print()

    generated_records = load_jsonl(records_path) if records_path.exists() else []
    persisted_failures = load_jsonl(failures_path) if failures_path.exists() else []
    summary = summarize_multiturn(generated_records, persisted_failures)
    summary.update(
        {
            "input": str(args.input),
            "output_dir": str(output_dir),
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "ctx_size": args.ctx_size,
            "parallel": args.parallel,
            "max_tokens": args.max_tokens,
            "rag_top_k": args.rag_top_k,
            "web_mode": args.web_mode,
            "web_top_k": args.web_top_k,
        }
    )
    write_json(summary_path, summary)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
