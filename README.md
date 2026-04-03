# Multimodal Ecommerce QA

This repository now contains the main assignment workflow beyond the initial `文搜图` benchmark:

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

- `src/ecom_rag/`: reusable loading, embedding, and evaluation code for the final benchmark
- `scripts/run_official_multilingual_benchmark.py`: final official-aligned 8-row multilingual benchmark
- `scripts/run_siglip2_in_domain_pipeline.py`: SigLIP2 image-to-image, image+text-to-image, multi-route recall, and BGE reranking
- `scripts/run_rerank_filters.py`: reusable BGE / Qwen3.5 reranking on saved candidates
- `scripts/run_out_of_domain_retrieval.py`: SerpAPI + Jina Reader + Qwen3.5 web retrieval pipeline
- `scripts/render_sample_style_results.py`: renders the final white-table result image
- `reports/official_multilingual_results.md`: final benchmark summary
- `reports/model_official_practice_check.md`: official-usage cross-check for CLIP, Chinese-CLIP, SigLIP2, and GME
- `reports/full_experiment_journal.md`: durable running notes across the full assignment
- `reports/remaining_pdf_experiments_report.md`: summary of the later PDF experiments
- `artifacts/`: workspace-local caches and model weights

## Quick Start

```bash
export HF_HOME=/workspace/multimodal_ecommerce_qa/artifacts/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
uv sync
.venv/bin/python scripts/run_official_multilingual_benchmark.py --help
.venv/bin/python scripts/run_siglip2_in_domain_pipeline.py --help
.venv/bin/python scripts/run_out_of_domain_retrieval.py --help
```

All caches and model downloads stay inside this workspace so they persist across sessions.
