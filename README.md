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
- `src/ecom_qa/retrieval/`: embedding models, in-domain retrieval, reranking, and web retrieval
- `src/ecom_qa/evaluation/`: ranked-list and subcategory evaluation helpers
- `dataset/catalog/`: structured product metadata such as `products.jsonl`
- `dataset/manifests/`: manifests and split metadata
- `dataset/images/`: gallery images grouped by category and subcategory
- `scripts/run_retrieval_benchmark.py`: official-aligned multilingual text-to-image benchmark
- `scripts/run_retrieval_pipeline.py`: SigLIP2 image-to-image, image+text-to-image, multi-route recall, and reranking
- `scripts/run_reranking.py`: reusable BGE / Qwen3.5 reranking on saved candidates
- `scripts/run_web_search.py`: SerpAPI + Jina Reader + Qwen3.5 web retrieval pipeline
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
```

All caches and model downloads stay inside this workspace so they persist across sessions.
