# Remaining PDF Experiments Report

## Scope

This report covers the assignment stages after the official-aligned `文搜图` benchmark:

- `图搜图`
- `图文搜图`
- `多路召回`
- `排序 / 过滤`
- `域外网络检索`

The durable running log remains in:

- `reports/full_experiment_journal.md`

## Final Result Directories

The final retained result directories are:

- `reports/generated/official_multilingual_t2i_20260403T125235Z`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z`
- `reports/generated/out_of_domain_web_retrieval_20260403T165452Z`

## Official Alignment Choices

### In-domain retrieval model

- selected model family: `SigLIP2`
- checkpoint: `google/siglip2-base-patch16-224`
- reason: strongest and most stable family in the final official-aligned `文搜图` benchmark

### SigLIP2 usage

- text prompt template: `This is a photo of {label}.`
- text preprocessing kept aligned with the official guidance:
  - lowercase
  - `padding="max_length"`
  - `max_length=64`
  - truncation
- image and text embeddings are L2-normalized before similarity

### Reranking / filtering models

- `BGE` baseline: `BAAI/bge-reranker-v2-m3`
  - pair input: `[query, document]`
  - `padding=True`
  - `truncation=True`
  - `max_length=512`
- PDF `Qwen3` replacement:
  - `Qwen/Qwen3.5-4B`
  - `Qwen/Qwen3.5-9B`
- Qwen generation settings aligned to the official non-thinking guidance:
  - `temperature=0.7`
  - `top_p=0.8`
  - `top_k=20`
  - `enable_thinking=False`

## In-domain RAG Results

Run directory:

- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z`

### 1. 图搜图

Protocol:

- `5` random reference-image repeats per canonical subcategory
- reference image excluded from the gallery
- metrics averaged over `250` total query cases

Results:

- `P@1 = 0.9200`
- `P@3 = 0.8533`
- `P@5 = 0.8592`
- `P@10 = 0.8432`
- `P@20 = 0.7994`
- `P@50 = 0.7198`
- mean precision across `1/3/5/10/20/50`: `0.8325`

### 2. 图文搜图

Implementation:

- same `5` reference images as `图搜图`
- fuse:
  - the reference-image embedding
  - the official SigLIP2 text embedding for the same subcategory prompt
- fusion rule: average then renormalize

Results:

- `P@1 = 0.9320`
- `P@3 = 0.9000`
- `P@5 = 0.8976`
- `P@10 = 0.8916`
- `P@20 = 0.8766`
- `P@50 = 0.8257`
- mean precision across `1/3/5/10/20/50`: `0.8872`

Takeaway:

- `图文搜图` is consistently stronger than pure `图搜图` at every reported `K`.

### 3. 多路召回

Implementation:

- text branch: SigLIP2 `文搜图` top `5`
- image branch: SigLIP2 `图搜图` top `5`
- merged result: `10` unique items, preserving a `5 + 5` route contribution
- comparison baseline: `图文搜图 Top10`

Results:

- multi-route `P@10 = 0.9080`
- image+text `P@10 = 0.8916`
- multi-route `Hit@10 = 1.0000`
- image+text `Hit@10 = 0.9960`

Takeaway:

- On this dataset, the explicit `文搜图 + 图搜图` merge is slightly stronger than the single fused `图文搜图` query at `Top10`.

## Ranking / Filtering Results

Candidate source:

- aggregated `10`-item multi-route candidate set per canonical subcategory
- `50` subcategory cases total

Document form:

- compact Chinese product text built from:
  - category
  - subcategory
  - title
  - brand
  - description

### 1. Label-grounded machine metrics

These metrics use the dataset's known ground-truth subcategory labels.

`top5_relevance_ratio`

- fraction of the selected top `5` documents whose ground-truth subcategory matches the target canonical subcategory

`filtered_irrelevant_ratio`

- fraction of the filtered-out documents that are actually irrelevant under the same ground-truth subcategory label

Results:

| Method | Avg Top5 Relevance | Avg Filtered Irrelevant |
| --- | ---: | ---: |
| `BGE reranker v2 m3` | `0.9320` | `0.3160` |
| `Qwen3.5-4B` | `0.9413` | `0.3200` |
| `Qwen3.5-9B` | `0.9080` | `0.2920` |

Interpretation:

- On exact-label matching, `Qwen3.5-4B` is the best top-5 selector by a small margin.
- `Qwen3.5-9B` is weaker than the other two on strict label matching, which suggests it often keeps semantically close but not exactly same-subcategory items.

### 2. GPT-5.4 semantic evaluation

The PDF asked for GPT-based evaluation for the reranking stage. In this repo, that was implemented with GPT-5.4 subagents split into six semantic-review batches and then merged into final result files.

Artifacts:

- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/gpt54_eval_bge.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/gpt54_eval_qwen3_5_4b.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/gpt54_eval_qwen3_5_9b.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/gpt54_eval_summary.csv`

`avg_top5_relevant_ratio_gpt`

- fraction of top-5 documents judged semantically relevant to the custom user query

`avg_filtered_correct_ratio_gpt`

- fraction of filtered documents judged to be correctly removed

Results:

| Method | GPT Top5 Relevant | GPT Filtered Correct |
| --- | ---: | ---: |
| `BGE reranker v2 m3` | `0.9640` | `0.1920` |
| `Qwen3.5-4B` | `0.9893` | `0.2320` |
| `Qwen3.5-9B` | `0.9840` | `0.5040` |

Interpretation:

- `Qwen3.5-4B` is the strongest top-5 semantic selector.
- `Qwen3.5-9B` is the strongest true filter by a wide margin under semantic judgment.
- The difference between the label-grounded view and the GPT semantic view is meaningful:
  - `Qwen3.5-9B` often keeps semantically relevant near-neighbor items that are not exact subcategory matches.
  - That hurts strict label metrics, but helps the semantic filtering metric.

## Out-of-domain Web Retrieval

Implemented code:

- `src/ecom_rag/web_retrieval.py`
- `scripts/run_out_of_domain_retrieval.py`

Capabilities in the repo:

- SerpAPI text search via `engine=google`
- SerpAPI image search via `engine=google_images`
- Jina Reader extraction via `https://r.jina.ai/<url>`
- Qwen3.5 webpage summarization with the PDF's summary prompt
- optional image-to-query generation via Qwen3.5 for image-driven web retrieval

### Final retained run

Artifact:

- `reports/generated/out_of_domain_web_retrieval_20260403T165452Z/web_retrieval_results.json`

What this run demonstrates:

- `Qwen3.5-4B` can convert a local product image into a concise search query
- `Jina Reader + Qwen3.5-4B` can summarize webpage content successfully
- the SerpAPI code path is implemented, but the current environment does not expose `SERPAPI_API_KEY`
- because of that missing key, live external search requests are recorded explicitly as unavailable rather than failing silently

Important limitation:

- the repo is ready for live SerpAPI runs, but this environment did not provide an API key, so the web-search portion could only be completed up to the implemented-and-recorded code-path stage

## Main Takeaways

1. `SigLIP2` remains a strong and practical choice for the in-domain multimodal retrieval stages after `文搜图`.
2. `图文搜图` beats pure `图搜图`, but explicit `文搜图 + 图搜图` multi-route recall is even better at `Top10`.
3. In reranking, `Qwen3.5-4B` is the best semantic top-5 selector.
4. In semantic filtering quality, `Qwen3.5-9B` is clearly best once the more memory-conservative 4-bit loading path is used.
5. The out-of-domain code path is complete, and the only missing live external result is the SerpAPI request itself because no API key was available.

## Output References

### In-domain

- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/in_domain_summary.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/per_case_metrics.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/aggregated_candidates.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/bge_reranker_results.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/qwen3_5_4b_filter_results.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/qwen3_5_9b_filter_results.json`
- `reports/generated/siglip2_in_domain_pipeline_20260403T155304Z/gpt54_eval_summary.csv`

### Out-of-domain

- `reports/generated/out_of_domain_web_retrieval_20260403T165452Z/web_retrieval_results.json`

## Final Status

Completed:

- official-aligned `文搜图`
- `SigLIP2 图搜图`
- `SigLIP2 图文搜图`
- `多路召回`
- `BGE reranker`
- `Qwen3.5-4B` filtering
- `Qwen3.5-9B` filtering
- GPT-5.4 semantic rerank evaluation
- out-of-domain retrieval code path
- `Jina Reader + Qwen3.5-4B` live summary demo

Environment-limited:

- live SerpAPI retrieval requests, because no `SERPAPI_API_KEY` was available in the environment
