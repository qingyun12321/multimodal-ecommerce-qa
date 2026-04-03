# Full Experiment Journal

## Objective

This journal tracks the remaining work from the assignment PDF after the `文搜图` stage was completed.
The goal is to keep one durable place for:

- the original task scope from the PDF
- the official model-usage rules we align to
- parameter choices and optimizations
- run-by-run experiment notes
- known blockers and their current handling

## Assignment Scope Snapshot

The PDF asks for three large blocks:

1. In-domain RAG
   - text-to-image (`文搜图`)
   - image-to-image (`图搜图`)
   - image+text-to-image (`图文搜图`)
   - multi-route recall (`文搜图 + 图搜图`)
   - ranking / filtering on the recalled descriptions
2. Out-of-domain web retrieval
   - SerpAPI image retrieval
   - SerpAPI text retrieval
   - Jina Reader webpage extraction
   - Qwen summary generation
3. Experiment recording and evaluation
   - metric tables
   - saved artifacts
   - GPT-based relevance evaluation

## Completed So Far

### Step 1: Official-Aligned Text-to-Image Benchmark

- Dataset size: `5000` product images
- Canonical subcategories: `50`
- Models compared:
  - `openai/clip-vit-base-patch32`
  - `OFA-Sys/chinese-clip-vit-base-patch16`
  - `google/siglip2-base-patch16-224`
  - `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`
- Final benchmark shape: `8` rows
  - `4` English inputs
  - `4` Chinese inputs
- Final result summary:
  - `SigLIP2 / English`: `0.9409`
  - `GME-2B / English`: `0.9408`
  - `SigLIP2 / Chinese`: `0.8607`
  - `Chinese-CLIP / Chinese`: `0.7995`

Decision used for the remaining in-domain RAG stages:

- selected model family: `SigLIP2`
- selected checkpoint: `google/siglip2-base-patch16-224`
- reason: strongest and most stable result under the official-aligned benchmark

Reference artifacts:

- `reports/official_multilingual_results.md`
- `reports/model_official_practice_check.md`
- `reports/generated/official_multilingual_t2i_20260403T125235Z/combined_metrics.json`
- `official_multilingual_results_sample_style.jpg`

## Official Alignment Baseline For Remaining Work

### SigLIP2

Sources:

- `https://huggingface.co/docs/transformers/model_doc/siglip2`
- `https://huggingface.co/google/siglip2-base-patch16-224`

Rules carried forward:

- text prompt for label-style retrieval: `This is a photo of {label}.`
- keep official text preprocessing behavior:
  - lowercase
  - `padding="max_length"`
  - `max_length=64`
  - truncation enabled
- use L2-normalized image and text embeddings for similarity

### BGE Reranker

Sources:

- `https://github.com/FlagOpen/FlagEmbedding`
- `https://huggingface.co/BAAI/bge-reranker-v2-m3`

Planned baseline:

- model: `BAAI/bge-reranker-v2-m3`
- pair format: `[query, document]`
- tokenization:
  - `padding=True`
  - `truncation=True`
  - `max_length=512`

### Qwen3.5

Sources:

- `https://huggingface.co/Qwen/Qwen3.5-4B`
- `https://huggingface.co/Qwen/Qwen3.5-9B`

Required replacement from the user:

- replace PDF `Qwen3` usage with:
  - `Qwen/Qwen3.5-4B`
  - `Qwen/Qwen3.5-9B`

Official guidance we will align to:

- use the latest compatible `transformers`
- non-thinking filtering mode
- recommended non-thinking generation settings:
  - `temperature=0.7`
  - `top_p=0.8`
  - `top_k=20`
  - `presence_penalty=1.5`
  - `chat_template_kwargs={"enable_thinking": False}`

### SerpAPI And Jina Reader

Sources:

- `https://serpapi.com/search-api`
- `https://serpapi.com/google-images-api`
- `https://jina.ai/reader/`

Planned workflow:

- text retrieval: `SerpAPI (engine=google)`
- image retrieval: `SerpAPI (engine=google_images)`
- webpage extraction: `https://r.jina.ai/<url>`
- summarization prompt: keep the PDF prompt unchanged except replacing Qwen3 with Qwen3.5

## Environment Notes

- workspace root: `/workspace/multimodal_ecommerce_qa`
- GPU detected: `RTX 5090 32GB`
- persistent caches stay under workspace paths
- current `.gitignore` excludes:
  - dataset
  - artifacts
  - weights
  - PDF
  - sample reference image

## Known Constraints

- `SERPAPI_API_KEY` was not present in the environment during the last check.
- `JINA_API_KEY` was not present in the environment during the last check.
- the original local `transformers==4.51.3` was too old for `Qwen3.5`; the repo was upgraded to a current `transformers` mainline build.
- `Qwen3.5-9B` needed a more conservative runtime path than `4B`:
  - sequential file download
  - 4-bit quantized loading via `bitsandbytes`
  - otherwise the host killed the process during weight preparation

## Execution Plan

- [x] Finish official-aligned `文搜图`
- [x] Upgrade the environment for `Qwen3.5`
- [x] Implement `SigLIP2 图搜图`
- [x] Implement `SigLIP2 图文搜图`
- [x] Implement multi-route recall and compare with `图文搜图 Top10`
- [x] Implement `BGE reranker`
- [x] Implement `Qwen3.5-4B` filtering
- [x] Implement `Qwen3.5-9B` filtering
- [x] Run GPT-5.4 subagent relevance evaluation artifacts
- [x] Implement out-of-domain web retrieval
- [x] Write the consolidated report

## Run Log

### 2026-04-03 UTC

- Re-read the extracted PDF notes to re-establish the remaining task list.
- Confirmed the Step 1 repository had been cleaned down to the final `文搜图` code and outputs.
- Confirmed the next in-domain stages should use `SigLIP2`.
- Collected the official-alignment notes for SigLIP2, BGE, Qwen3.5, SerpAPI, and Jina Reader.
- Confirmed `transformers==4.51.3` cannot load `Qwen3.5` because the checkpoint uses `model_type=qwen3_5`.
- Started the environment upgrade and created this durable journal before implementing the remaining stages.
- Upgraded the environment to a newer `transformers` main build and added `openai` support so `Qwen3.5` checkpoints could load.
- Implemented the in-domain SigLIP2 pipeline in `src/ecom_rag/in_domain_pipeline.py` and `scripts/run_siglip2_in_domain_pipeline.py`.
- Ran the in-domain retrieval benchmark and saved the final retained run at `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z`.
- Final in-domain retrieval results:
  - image-to-image mean precision: `0.8325`
  - image+text-to-image mean precision: `0.8872`
  - multi-route `P@10`: `0.9080`
  - image+text baseline `P@10`: `0.8916`
- Implemented reranking / filtering with:
  - `BAAI/bge-reranker-v2-m3`
  - `Qwen/Qwen3.5-4B`
  - `Qwen/Qwen3.5-9B`
- The `Qwen3.5-9B` path initially failed under default loading pressure. The final successful path used:
  - sequential repo download
  - `bitsandbytes` 4-bit quantization
  - the same official non-thinking prompt / sampling settings
- Final label-grounded rerank metrics:
  - `BGE`: top5 `0.9320`, filtered `0.3160`
  - `Qwen3.5-4B`: top5 `0.9413`, filtered `0.3200`
  - `Qwen3.5-9B`: top5 `0.9080`, filtered `0.2920`
- Ran GPT-5.4 semantic evaluation through subagents in six batches and merged them into final files:
  - `gpt54_eval_bge.json`
  - `gpt54_eval_qwen3_5_4b.json`
  - `gpt54_eval_qwen3_5_9b.json`
- Final GPT-5.4 semantic rerank metrics:
  - `BGE`: top5 `0.9640`, filtered `0.1920`
  - `Qwen3.5-4B`: top5 `0.9893`, filtered `0.2320`
  - `Qwen3.5-9B`: top5 `0.9840`, filtered `0.5040`
- Implemented the out-of-domain retrieval code in:
  - `src/ecom_rag/web_retrieval.py`
  - `scripts/run_out_of_domain_retrieval.py`
- Final retained out-of-domain run is `reports/generated/out_of_domain_web_retrieval_20260403T165452Z`.
- Verified:
  - image-to-query generation works with `Qwen3.5-4B`
  - `Jina Reader + Qwen3.5-4B` webpage summarization works
  - SerpAPI code path is implemented and records the missing-key state clearly
- Cleaned superseded partial runs so the final retained result directories are:
  - `reports/generated/official_multilingual_t2i_20260403T125235Z`
  - `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z`
  - `reports/generated/out_of_domain_web_retrieval_20260403T165452Z`
- Upgraded the environment to `transformers` mainline (`5.6.0.dev0` in this run) and added `openai` so the repo can follow the current Qwen3.5-compatible stack.
- Implemented the reusable in-domain pipeline modules:
  - `src/ecom_rag/in_domain_pipeline.py`
  - `src/ecom_rag/rerankers.py`
  - `scripts/run_siglip2_in_domain_pipeline.py`
  - `scripts/run_rerank_filters.py`
- Implemented the out-of-domain retrieval modules:
  - `src/ecom_rag/web_retrieval.py`
  - `scripts/run_out_of_domain_retrieval.py`
- Ran the SigLIP2 in-domain pipeline:
  - run name: `siglip2_in_domain_pipeline_20260403T155304Z`
  - image-to-image mean precision: `0.8325`
  - image+text-to-image mean precision: `0.8872`
  - multi-route `P@10`: `0.9080`
  - image+text-to-image `P@10`: `0.8916`
- Ran reranking on aggregated multi-route candidates:
  - `BGE reranker v2 m3`: `avg_top5_relevance_ratio = 0.9320`, `avg_filtered_irrelevant_ratio = 0.3160`
  - `Qwen3.5-4B`: `avg_top5_relevance_ratio = 0.9413`, `avg_filtered_irrelevant_ratio = 0.3200`
  - `Qwen3.5-9B`: `avg_top5_relevance_ratio = 0.9080`, `avg_filtered_irrelevant_ratio = 0.2920`
- Stabilized `Qwen3.5-9B` with a more conservative path:
  - sequential repository download
  - `bitsandbytes` 4-bit quantized loading
  - final full `50`-case run completed successfully
- Generated GPT-5.4 subagent evaluation artifacts:
  - `gpt54_eval_bge.json`
  - `gpt54_eval_qwen3_5_4b.json`
  - `gpt54_eval_qwen3_5_9b.json`
  - `gpt54_eval_summary.csv`
- Kept the final out-of-domain run:
  - `out_of_domain_web_retrieval_20260403T165452Z`
  - this run demonstrates image-to-query generation, explicit SerpAPI missing-key recording, Jina Reader extraction, and Qwen3.5-4B summary generation in one place
- Cleaned redundant intermediate run directories so only the final in-domain and final out-of-domain result directories remain relevant
- Remaining external limitation:
  - live SerpAPI execution is still blocked by missing `SERPAPI_API_KEY`
