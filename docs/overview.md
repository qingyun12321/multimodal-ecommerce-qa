# Project Overview

This project supports multimodal ecommerce QA workflows across six capability areas:

- data preparation for product catalogs, image galleries, QA rows, VQA views, and tool-call labels
- retrieval evaluation for text-to-image, image-to-image, image+text-to-image, multi-route recall, reranking, and web retrieval
- tool-call data generation with direct answer, local retrieval, web search, and grounding labels
- VLM tool-call inference and multi-turn information injection with `<information>` blocks
- SFT and GRPO training support for tool-call behavior
- reporting and retained experiment summaries

The repository stores generalized requirements and implementation decisions. Source specification files remain outside git.

## Command Surface

All supported workflows run through:

```bash
uv run ecom-qa <group> <command> [args]
```

Old script paths are intentionally removed so command discovery and documentation stay centralized.

## Runtime Roots

- `data/`: raw, processed, generated, and run data
- `artifacts/`: model caches, weights, manifests, and temporary runtime files
- `reports/generated/`: generated report runs
- `reports/experiments/`: curated retained summaries and compact metrics

Only curated reports and configs should be committed. Runtime data and large artifacts should stay local or on external storage.
