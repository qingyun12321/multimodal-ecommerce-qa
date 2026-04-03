from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ecom_rag.web_retrieval import (
    JinaReaderClient,
    Qwen3_5TextGenerator,
    SerpAPIClient,
    SerpAPISearchConfig,
    ensure_workspace_env,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run out-of-domain web retrieval experiments.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--text-query", default="")
    parser.add_argument("--image-query", default="")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--question", default="这个页面的主要内容是什么？哪些信息有助于回答用户检索问题？")
    parser.add_argument("--hl", default="zh-cn")
    parser.add_argument("--gl", default="cn")
    parser.add_argument("--location", default="Beijing, China")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--ijn", type=int, default=0)
    parser.add_argument("--skip-summarization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_updates = ensure_workspace_env()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"out_of_domain_web_retrieval_{timestamp}"
    report_dir = args.report_dir / run_name
    artifact_dir = args.artifact_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    serpapi_api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")
    jina_api_key = os.environ.get("JINA_API_KEY") or os.environ.get("JINA_KEY")

    serp_client = SerpAPIClient(
        SerpAPISearchConfig(
            api_key=serpapi_api_key,
            hl=args.hl,
            gl=args.gl,
            location=args.location,
        )
    )
    reader = JinaReaderClient(api_key=jina_api_key)

    qwen = None
    if not args.skip_summarization:
        qwen = Qwen3_5TextGenerator(model_id=args.qwen_model_id, device=args.device)

    result: dict[str, object] = {
        "run_name": run_name,
        "workspace_env": env_updates,
        "serpapi_api_key_present": bool(serpapi_api_key),
        "jina_api_key_present": bool(jina_api_key),
        "text_query": args.text_query,
        "image_query": args.image_query,
        "image_path": str(args.image_path) if args.image_path else None,
        "qwen_model_id": args.qwen_model_id if qwen else None,
        "steps": [],
    }

    if args.text_query:
        try:
            text_search = serp_client.text_search(query=args.text_query, start=args.start)
            result["text_search"] = text_search
            result["steps"].append("text_search")
        except Exception as exc:
            result["text_search_error"] = str(exc)

    resolved_image_query = args.image_query
    if args.image_path and qwen is not None:
        try:
            resolved_image_query = qwen.image_to_search_query(image_path=args.image_path)
            result["generated_image_query"] = resolved_image_query
            result["steps"].append("image_to_query")
        except Exception as exc:
            result["generated_image_query_error"] = str(exc)

    if resolved_image_query:
        try:
            image_search = serp_client.image_search(query=resolved_image_query, ijn=args.ijn)
            result["image_search"] = image_search
            result["steps"].append("image_search")
        except Exception as exc:
            result["image_search_error"] = str(exc)

    organic_results = []
    if "text_search" in result:
        organic_results = result["text_search"].get("organic_results", [])[:3]
    elif args.url:
        organic_results = [{"title": url, "link": url} for url in args.url]

    reader_rows = []
    for item in organic_results:
        url = item.get("link")
        if not url:
            continue
        try:
            reader_payload = reader.read(url=url)
            row = {
                "title": item.get("title"),
                "url": url,
                "reader_content_length": reader_payload["content_length"],
                "reader_content_preview": reader_payload["content"][:500],
            }
            if qwen is not None:
                row["summary"] = qwen.summarize_webpage(
                    webpage_content=reader_payload["content"],
                    question=args.question,
                )
            reader_rows.append(row)
        except Exception as exc:
            reader_rows.append(
                {
                    "title": item.get("title"),
                    "url": url,
                    "error": str(exc),
                }
            )
    result["reader_rows"] = reader_rows

    write_json(report_dir / "web_retrieval_results.json", result)
    write_json(
        artifact_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "report_dir": str(report_dir),
            "artifact_dir": str(artifact_dir),
        },
    )

    if qwen is not None:
        qwen.cleanup()

    print(json.dumps({"run_name": run_name, "report_dir": str(report_dir), "steps": result["steps"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
