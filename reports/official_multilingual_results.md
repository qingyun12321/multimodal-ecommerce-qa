# Official-Aligned Multilingual Text-to-Image Benchmark

## Scope

- Assignment target: `文搜图`
- Dataset size: `5000` images
- Canonical subcategory count: `50`
- Comparison shape: `8` rows total
- Input split: `4` English rows + `4` Chinese rows
- Execution mode: serial, one model at a time, reusing image embeddings inside each model

## Official Alignment Rules Used

- `CLIP`
  - English input: OpenAI zero-shot prompt style `a photo of a {label}`
  - Chinese input: same official prompt shell with the Chinese label substituted in
- `Chinese-CLIP`
  - English input: raw English label
  - Chinese input: raw Chinese label
  - This follows the official examples more closely than forcing an extra prompt template
- `SigLIP2`
  - English input: `This is a photo of {label}.`
  - Chinese input: same official prompt shell with the Chinese label substituted in
  - Text handling follows the official Transformers guidance: `padding=max_length`, `max_length=64`, lowercase
- `GME-2B`
  - English input: raw English label
  - Chinese input: raw Chinese label
  - Retrieval uses the official text-to-image instruction and `is_query=False` for corpus image embeddings

## Result Table

Average below is the mean of `P@1 / P@3 / P@5 / P@10 / P@20 / P@50`.

| Model | Input | P@1 | P@3 | P@5 | P@10 | P@20 | P@50 | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP | English | 0.8400 | 0.8533 | 0.8360 | 0.8040 | 0.7590 | 0.7136 | 0.8010 |
| Chinese-CLIP | English | 0.8400 | 0.8067 | 0.7680 | 0.7460 | 0.7010 | 0.6436 | 0.7509 |
| SigLIP2 | English | 0.9400 | 0.9600 | 0.9560 | 0.9460 | 0.9360 | 0.9076 | 0.9409 |
| GME-2B | English | 1.0000 | 0.9733 | 0.9520 | 0.9400 | 0.9100 | 0.8696 | 0.9408 |
| CLIP | Chinese | 0.1000 | 0.0533 | 0.0520 | 0.0540 | 0.0520 | 0.0420 | 0.0589 |
| Chinese-CLIP | Chinese | 0.8200 | 0.8267 | 0.8200 | 0.8040 | 0.7840 | 0.7424 | 0.7995 |
| SigLIP2 | Chinese | 0.9400 | 0.8733 | 0.8520 | 0.8520 | 0.8390 | 0.8080 | 0.8607 |
| GME-2B | Chinese | 0.8600 | 0.8600 | 0.8640 | 0.8180 | 0.8150 | 0.7608 | 0.8296 |

## Main Takeaways

1. On English input, `SigLIP2` and `GME-2B` are effectively tied at the top, with `SigLIP2` ahead by a very small mean margin in this run.
2. On Chinese input, `SigLIP2` is the strongest of the four tested rows under the exact official-aligned setup used here.
3. `Chinese-CLIP` benefits clearly from Chinese input and remains the most stable explicitly Chinese-oriented baseline.
4. `CLIP` collapses on the Chinese row when forced into the official English prompt shell with Chinese labels inserted. This should be read as a cross-lingual prompt mismatch, not as a universal image encoder failure.
5. `GME-2B` stays strong in both languages, but on this benchmark its Chinese-input row is weaker than its English-input row.

## Interpretation Notes

- The `CLIP` and `SigLIP2` Chinese rows use the official English prompt shell with Chinese labels substituted in, because the official references do not provide a separate Chinese prompt template.
- The `Chinese-CLIP` and `GME-2B` rows are closer to their native official usage for Chinese text.
- Because of that difference, the Chinese rows are useful and requested, but they are not all equally native to each model family.

## Output Files

- Combined JSON: `reports/generated/official_multilingual_t2i_20260403T125235Z/combined_metrics.json`
- Combined CSV: `reports/generated/official_multilingual_t2i_20260403T125235Z/combined_metrics.csv`
- Run manifest: `artifacts/runs/official_multilingual_t2i_20260403T125235Z/run_manifest.json`
- Table image: `official_multilingual_results_sample_style.jpg`
