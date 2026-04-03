# Assignment Notes

## Source

- Local file: `ecommerce_qa_assignment.pdf`
- Extracted on: 2026-04-03 UTC

## Relevant Scope For This Repository

This repository implements only the first step of the in-domain RAG workflow:

- `文搜图`
- model scope for this run: `GME 2B` only

## PDF Requirements For Step 1

The PDF asks for the following workflow in the retrieval module:

1. Build an ecommerce multimodal dataset with category and subcategory structure.
2. Run `文搜图` by using subcategory text as input and retrieving images from the gallery.
3. Compare several models in the full assignment, but for the current task we only test `GME-2B`.
4. Report TopK accuracy for `K = 1 / 3 / 5 / 10 / 20 / 50`.
5. Average the results across subcategories.
6. Record the test results.

## Local Interpretation Used Here

Because the synthetic `subcategory` field inside `dataset/products.jsonl` contains many noisy synonyms, the reproducible canonical subcategory definition comes from the image folder layout:

- `dataset/images/<category_slug>/<subcategory_slug>/<image_file>`

This gives 50 stable subcategories, which matches the PDF expectation of 5 to 10 categories with at least 5 subcategories each.

For the GME 2B benchmark in this repository:

- Retrieval query unit: canonical subcategory
- Query text mode used for the main benchmark: canonical Chinese subcategory text derived from the dominant label inside each leaf image folder
- Retrieval gallery: all 5000 product images
- Positive set for one query: every image under the matching canonical subcategory folder
- Primary metric reported as `avg_precision@K`
- Secondary diagnostics also reported: `avg_recall@K` and `avg_hit@K`

## Query Text Choice

The PDF describes Step 1 as "input subcategory text, then retrieve images". The primary benchmark in this repository therefore uses Chinese subcategory text to stay aligned with the assignment wording and the Chinese product metadata.

The image folder layout is still used as the stable source of ground-truth grouping because the raw `subcategory` field contains many noisy synonyms. For each leaf folder, the benchmark uses the dominant Chinese `subcategory` label among its member products as the canonical query text.
