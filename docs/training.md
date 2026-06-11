# Training

Training support is organized under:

- `configs/training/sft/`
- `configs/training/grpo/`
- `ops/training/sft/`
- `ops/training/grpo/`
- `src/ecom_qa/training/`
- `reports/sft/`
- `reports/grpo/`

Primary commands:

```bash
uv run ecom-qa train sft-build-data --help
uv run ecom-qa train sft-sync-mlflow --help
```

SFT data conversion reads the repository-local tool-call splits from `data/tool_call/` by default and writes ms-swift formatted files to `data/training/sft/generated/mimo_multiturn/`. The checked-in SFT environment keeps this directory under `REPO_DIR` while external model, checkpoint, and run roots remain absolute `/workspace/...` paths for training hosts.

The multi-turn SFT builder uses OpenRouter `xiaomi/mimo-v2.5` through the Python API to fill natural-language fields for direct answers and tool-use turns. The model returns tag-free JSON fields only; repository scripts render the training protocol tags, including `<Think>`, `<Answer>`, `<RAG_search>`, `<Web_search>`, `<Grounding>`, and `<information>`. RAG turns follow the homework prompt flow: answer from RAG when the retrieved product information is sufficient, otherwise continue into Web_search; only after Web_search may the script render the fixed unable-to-answer text. Web `<information>` contains content summaries only, without titles or source URLs.

API data generation defaults to `xiaomi/mimo-v2.5`, `reasoning.effort=none`, `--api-workers 5`, fallback workers `5,3,1`, and `--web-max-items 5`. Workers are long-lived Python thread-pool workers that process multiple records from the queue. Worker output is appended directly to JSONL files under file locks, while compact API cache entries are stored in `api_cache/records.jsonl` and compact Web retrieval cache entries are stored in `web_cache/records.jsonl`. The validation report records API token usage, Web search/rerank/fetch stage timings, wall time, throughput, and an estimated 10k-row generation time.

Web_search uses the repo-managed SearXNG settings in `configs/retrieval/searxng-sft.yml`, normally started with `ops/retrieval/run_searxng_sft.sh`. The SFT retrieval chain queries Google, DuckDuckGo, Qwant, Wikipedia, and Wikimedia-family engines, deduplicates URLs, ranks candidates with `BAAI/bge-m3` hybrid scoring, reranks with `BAAI/bge-reranker-v2-m3`, and keeps `top_k=5` results above `min_score=0.3` by default.

The ms-swift arg files use repo-relative dataset paths such as `data/training/sft/generated/mimo_multiturn/train.ms_swift.jsonl`. The smoke and main SFT runner scripts change to `REPO_DIR` before invoking `swift sft` so those paths resolve consistently. Remote training machines can still override roots with environment variables such as `REPO_DIR`, `SFT_DATA_DIR`, `SFT_CHECKPOINT_ROOT`, and `MLFLOW_TRACKING_URI`.

Large checkpoints and full model weights should not be uploaded to this repository. MLflow should store configs, metrics, compact logs, selected evaluation artifacts, and paths or hashes for large external artifacts.
