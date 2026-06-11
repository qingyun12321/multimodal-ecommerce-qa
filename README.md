# Multimodal Ecommerce QA

This repository contains reusable workflows for multimodal ecommerce question answering: data preparation, retrieval evaluation, tool-call data generation, VLM tool-call inference, multi-turn information use, and SFT and GRPO training support.

The repository is organized by capability rather than by one-off experiment stages. External specifications stay outside git; this checkout keeps only generalized requirements, implementation decisions, code, configs, and curated experiment summaries.

## Layout

- `src/ecom_qa/`: Python package implementation and CLI modules.
- `configs/`: checked-in runtime and training configuration.
- `ops/`: shell helpers for local services, remote training, MLflow, TensorBoard, SearXNG, and llama.cpp.
- `docs/`: project overview, workflow notes, data policy, training notes, and requirement summaries.
- `reports/`: curated human-readable reports and selected metrics.
- `data/`: local raw, processed, generated, and run data. This directory is ignored by git.
- `artifacts/`: local model caches, weights, temporary manifests, and other large runtime artifacts. This directory is ignored by git.

## Setup

```bash
uv sync
uv run ecom-qa --help
```

Model caches and large runtime outputs should stay under `artifacts/` or an external workspace volume. Data roots default to `data/` and can be overridden with:

```bash
export ECOM_QA_DATA_ROOT=/path/to/data
export ECOM_QA_ARTIFACTS_ROOT=/path/to/artifacts
```

## Main Commands

```bash
uv run ecom-qa data build-qa --help
uv run ecom-qa data build-targeted-qa --help
uv run ecom-qa data build-unified-vqa --help
uv run ecom-qa data build-tool-call-labels --help
uv run ecom-qa data clean-tool-call-labels --help
uv run ecom-qa data build-tool-call-splits --help
uv run ecom-qa retrieval evaluate-multilingual --help
uv run ecom-qa retrieval evaluate-in-domain --help
uv run ecom-qa retrieval evaluate-web --help
uv run ecom-qa retrieval evaluate-rerankers --help
uv run ecom-qa infer tool-call --help
uv run ecom-qa infer multiturn-tool-call --help
uv run ecom-qa train sft-build-data --help
uv run ecom-qa train sft-sync-mlflow --help
uv run ecom-qa report render-retrieval --help
uv run ecom-qa report render-in-domain --help
```

The CLI is the supported entrypoint. Old script paths are intentionally not retained.

## Data And Reports

`data/` is the only repository-local data root. Generated data and run outputs should go under `data/**/generated/` or `data/runs/` and remain outside version control.

SFT multi-turn data generation reads tool-call splits from `data/tool_call/` and writes ms-swift JSONL under `data/training/sft/generated/codex_multiturn/` by default. The SFT builder uses Codex CLI to fill tag-free answer fields; repository scripts own rendering of `<Think>`, `<Answer>`, `<RAG_search>`, `<Web_search>`, `<Grounding>`, and `<information>` tags. RAG turns can continue into Web_search when local retrieval is insufficient, and only the Web turn may end with the fixed unable-to-answer text. Web `<information>` rows contain content only, without title/source URL fields. SFT data generation defaults to `gpt-5.3-codex-spark` with low reasoning, 5 Codex workers, worker fallback `5,3,1`, and at most 5 web evidence items. It does not require the local llama.cpp summary server or SearXNG service.

`reports/experiments/` keeps selected, durable summaries and compact metrics needed to understand retained results. Raw generated report runs belong in `reports/generated/`, which is ignored by git.

## Docs

- `docs/overview.md`: capability map and end-to-end workflow.
- `docs/data.md`: data root, manifests, generated-data policy, and path overrides.
- `docs/retrieval.md`: retrieval and reranking workflows.
- `docs/tool-calling.md`: tool-call labels, schemas, multi-turn protocol, and metrics.
- `docs/training.md`: SFT and GRPO, MLflow, remote training, and artifact policy.
- `docs/experiments.md`: retained experiment summaries.
- `docs/requirements/`: generalized requirements derived from external specifications.
