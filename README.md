# Multimodal Ecommerce QA

This repository is organized around reusable capability areas rather than assignment-number folders, so the current retrieval work and upcoming QA / agent work can keep growing in one codebase.

Current implemented workflows include:

- official-aligned multilingual text-to-image retrieval
- SigLIP2 in-domain follow-up experiments:
  - image-to-image
  - image+text-to-image
  - multi-route recall
  - BGE and Qwen3.5 reranking / filtering
- out-of-domain retrieval code paths for:
  - SerpAPI text search
  - SerpAPI image search
  - Jina Reader webpage extraction
  - Qwen3.5 webpage summarization

The repo also keeps durable Markdown notes so the long experiment chain can be resumed later without rebuilding context.

## Layout

- `src/ecom_qa/data/`: dataset schemas and catalog loading
- `src/ecom_qa/data/qa_generation.py`: local Gemma 4 QA generation helpers
- `src/ecom_qa/data/tool_call_generation.py`: local Gemma 4 tool-call annotation helpers
- `src/ecom_qa/retrieval/`: embedding models, in-domain retrieval, reranking, and web retrieval
- `src/ecom_qa/evaluation/`: ranked-list and subcategory evaluation helpers
- `dataset/catalog/`: structured product metadata such as `products.jsonl`
- `dataset/manifests/`: manifests and split metadata
- `dataset/images/`: gallery images grouped by category and subcategory
- `dataset/qa/generated/`: generated QA preview runs and timing summaries
- `scripts/run_retrieval_benchmark.py`: official-aligned multilingual text-to-image benchmark
- `scripts/run_retrieval_pipeline.py`: SigLIP2 image-to-image, image+text-to-image, multi-route recall, and reranking
- `scripts/run_reranking.py`: reusable BGE / Qwen3.5 reranking on saved candidates
- `scripts/run_web_search.py`: SerpAPI + Jina Reader + Qwen3.5 web retrieval pipeline
- `scripts/generate_qa/generate_qa_preview.py`: generate preview QA pairs with local `llama.cpp` + Gemma 4
- `scripts/generate_qa/generate_qa.py`: generate 5k QA pairs with local `llama.cpp` + Gemma 4 from the current catalog
- `scripts/generate_qa/generate_targeted_qa.py`: generate QA pairs targeted at specific expected tool-call classes
- `scripts/generate_tool_calls/`: InfoSeek sampling, unified VQA view building, tool-call annotation generation, and label cleaning
- `scripts/tool_call_inference/`: Qwen3-VL GGUF tool-call inference harness
- `scripts/render_benchmark_report.py`: renders the benchmark white-table image
- `scripts/render_pipeline_report.py`: renders the in-domain retrieval and reranking summary tables
- `reports/official_multilingual_results.md`: final benchmark summary
- `reports/model_official_practice_check.md`: official-usage cross-check for CLIP, Chinese-CLIP, SigLIP2, and GME
- `reports/full_experiment_journal.md`: durable running notes across the full assignment
- `reports/remaining_pdf_experiments_report.md`: summary of the later PDF experiments
- `artifacts/`: workspace-local caches and model weights

## Quick Start

```bash
export HF_HOME="$(pwd)/artifacts/hf_home"
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
uv sync
.venv/bin/python scripts/run_retrieval_benchmark.py --help
.venv/bin/python scripts/run_retrieval_pipeline.py --help
.venv/bin/python scripts/run_web_search.py --help
.venv/bin/python scripts/generate_qa/generate_qa_preview.py --help
.venv/bin/python scripts/generate_qa/generate_qa.py --help
.venv/bin/python scripts/generate_qa/generate_targeted_qa.py --help
.venv/bin/python scripts/generate_tool_calls/generate_tool_calls.py --help
.venv/bin/python scripts/tool_call_inference/run_qwen3_vl_tool_call_inference.py --help
```

All caches and model downloads stay inside this workspace so they persist across sessions.

## QA Preview Generation

Use the local `llama.cpp` Gemma 4 path to generate a small QA preview before scaling up:

```bash
.venv/bin/python scripts/generate_qa/generate_qa_preview.py \
  --multimodal-count 8 \
  --text-count 2
```

The script writes each run under `dataset/qa/generated/<run_name>/` with:

- `qa_pairs.jsonl`: generated QA rows
- `run_summary.json`: startup and per-sample timing summary
- `llama_server.log`: local server log for debugging

## Assignment 2 Full Generation

Use the end-to-end script when you want to generate a full 5k QA set with local Gemma 4 from the current catalog. If you want to preserve the current catalog state, create a manual backup before running the script.

```bash
.venv/bin/python scripts/generate_qa/generate_qa.py
```

Each run writes:

- `dataset/qa/generated/<run_name>/qa_pairs.jsonl`: generated QA rows
- `dataset/qa/generated/<run_name>/run_summary.json`: QA generation timing summary
- `dataset/qa/generated/<run_name>/llama_server.log`: `llama.cpp` server log

## Tool-Call Inference

Use the Qwen3-VL GGUF harness to run tool-call inference against the balanced tool-call dataset. The default run starts the 4B and 8B models one at a time through the local `llama.cpp` OpenAI-compatible server.

```bash
.venv/bin/python scripts/tool_call_inference/run_qwen3_vl_tool_call_inference.py \
  --input dataset/tool_call/tool_call_records_balanced.jsonl
```

For a short smoke run:

```bash
.venv/bin/python scripts/tool_call_inference/run_qwen3_vl_tool_call_inference.py \
  --models qwen3-vl-4b \
  --limit 20 \
  --run-name smoke_qwen3_vl_4b
```

The inference harness defaults to `--grounding-policy balanced`, which is conservative for mixed old and targeted labels. Use `--grounding-policy aggressive` when evaluating grounding-heavy targeted samples. It also runs with memory-conscious defaults for this short tool-call task: `--parallel 1`, `--ctx-size 8192`, `--max-tokens 512`, `--batch-size 1024`, and `--ubatch-size 256`.

Each run writes under `dataset/tool_call_inference/generated/<run_name>/<model_key>/`:

- `predictions.jsonl`: raw model output, parsed tool-call decision, and exact-match fields
- `run_summary.json`: per-model metrics, timing, and generation parameters
- `failed_rows.jsonl`: rows that failed during inference, only when failures occur
- `llama_server.log`: local server log for debugging
