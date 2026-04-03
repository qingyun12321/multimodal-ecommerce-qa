# Multimodal Ecommerce QA

This repository contains the first step of the in-domain RAG workflow from the assignment PDF: text-to-image retrieval (`文搜图`) using the final official-aligned 8-row benchmark with both English and Chinese inputs.

## Layout

- `src/ecom_rag/`: reusable loading, embedding, and evaluation code for the final benchmark
- `scripts/run_official_multilingual_benchmark.py`: final official-aligned 8-row multilingual benchmark
- `scripts/render_sample_style_results.py`: renders the final white-table result image
- `reports/official_multilingual_results.md`: final benchmark summary
- `reports/model_official_practice_check.md`: official-usage cross-check for CLIP, Chinese-CLIP, SigLIP2, and GME
- `artifacts/`: workspace-local caches and model weights

## Quick Start

```bash
export HF_HOME=/workspace/multimodal_ecommerce_qa/artifacts/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
uv sync
.venv/bin/python scripts/run_official_multilingual_benchmark.py --help
```

All caches and model downloads stay inside this workspace so they persist across sessions.
