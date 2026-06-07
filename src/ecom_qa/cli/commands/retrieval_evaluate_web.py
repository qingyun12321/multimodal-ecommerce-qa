from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ecom_qa.common.paths import generated_reports_root
from ecom_qa.retrieval.web import (
    LlamaCppSummarizer,
    SearXNGClient,
    format_web_information,
    summarize_search_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SearXNG + Qwen3.5 webpage-summary smoke check.")
    parser.add_argument("--report-dir", type=Path, default=generated_reports_root())
    parser.add_argument("--query", default="向日葵最早被发现的地方")
    parser.add_argument("--question", default="向日葵这种花卉最早是在哪里被发现的？")
    parser.add_argument("--searxng-url", default="http://127.0.0.1:8080")
    parser.add_argument("--searxng-engines", default="bing")
    parser.add_argument("--summary-url", default="http://127.0.0.1:8088")
    parser.add_argument("--summary-model", default="qwen3.5-9b-summary")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--summarize-pages", type=int, default=2)
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = args.report_dir / f"searxng_web_summary_{timestamp}"
    client = SearXNGClient(args.searxng_url, engines=args.searxng_engines)
    summarizer = None if args.skip_summary else LlamaCppSummarizer(args.summary_url, model=args.summary_model)
    results = client.search(args.query, num_results=args.top_k)
    summaries = summarize_search_results(
        results=results,
        question=args.question,
        summarizer=summarizer,
        max_pages=args.summarize_pages,
        allow_fallback=True,
    )
    payload = {
        "query": args.query,
        "question": args.question,
        "searxng_url": args.searxng_url,
        "searxng_engines": args.searxng_engines,
        "summary_url": None if args.skip_summary else args.summary_url,
        "result_count": len(results),
        "summary_count": len(summaries),
        "results": [asdict(item) for item in results],
        "summaries": [asdict(item) for item in summaries],
        "information": format_web_information(summaries),
    }
    write_json(report_dir / "web_retrieval_results.json", payload)
    print(json.dumps({"report_dir": str(report_dir), "result_count": len(results), "summary_count": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
