# Retrieval Workflows

Retrieval workflows cover:

- multilingual text-to-image retrieval
- in-domain image-to-image and image+text-to-image retrieval
- multi-route recall
- reranking and filtering with cross-encoder or generative filters
- out-of-domain web retrieval with SearXNG and local summarization

Primary commands:

```bash
uv run ecom-qa retrieval evaluate-multilingual --help
uv run ecom-qa retrieval evaluate-in-domain --help
uv run ecom-qa retrieval evaluate-rerankers --help
uv run ecom-qa retrieval evaluate-web --help
```

Generated retrieval outputs should go to `reports/generated/` or `data/runs/`. Retained summaries belong under `reports/experiments/retrieval/`.
