# Official Practice Check

This note maps the PDF reference links to the official model usage patterns that matter for the `文搜图` benchmark.

## CLIP

- PDF reference: `https://github.com/openai/CLIP`
- Model used here: `openai/clip-vit-base-patch32`

Official usage pattern:

- The OpenAI CLIP zero-shot example builds text prompts like `a photo of a {class_name}`.
- It encodes image and text separately, then L2-normalizes both sides before similarity scoring.

Implication for this repo:

- Our baseline CLIP run used bare English labels, not an explicit prompt template.
- That baseline is still a valid internal comparison, but it is not the most official-prompt-aligned CLIP setup.

## Chinese-CLIP

- PDF reference: `https://github.com/OFA-Sys/Chinese-CLIP`
- Model used here: `OFA-Sys/chinese-clip-vit-base-patch16`

Official usage pattern:

- The official examples use Chinese text labels for zero-shot image classification and retrieval.
- The repo examples also normalize image and text features before similarity.
- The project demo explicitly supports custom prompt templates, so prompt engineering is a supported practice.

Implication for this repo:

- The first four-model run used English slug queries for all models, which was reproducible but not ideal for Chinese-CLIP.
- A Chinese-query rerun was added and should be considered the fairer Chinese-CLIP result.

## SigLIP2

- PDF reference: `https://modelscope.cn/models/google/siglip2-base-patch16-224`
- Model used here: `google/siglip2-base-patch16-224`

Official usage pattern:

- The official model card and Transformers docs present this checkpoint as a zero-shot image classification and image-text retrieval model.
- The recommended prompt format for label-style queries is `This is a photo of {label}.`
- The docs also note that, to reproduce pipeline behavior, the text side should use `padding="max_length"`, `max_length=64`, and lowercased text.

Implication for this repo:

- The original poor SigLIP2 result came from using bare labels instead of the official prompt style.
- After switching to the official text handling, SigLIP2 performance became strong on this dataset.
- We used the official Google checkpoint name on Hugging Face. It matches the ModelScope page name from the PDF, but this repo did not perform a byte-level equality check between the two hosting mirrors.

## GME-2B

- PDF references:
  - `https://modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct`
  - `https://modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct`
- Model used here: `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`

Official usage pattern:

- The official examples use `get_text_embeddings(..., instruction=t2i_prompt)` for text-to-image retrieval.
- They use `get_image_embeddings(..., is_query=False)` for image corpus embeddings.
- The model card also documents memory-sensitive pixel or token controls for reducing resource usage.

Implication for this repo:

- Our implementation already follows the official retrieval API pattern.
- The assignment only required testing the `2B` version for GME, and that is the only GME model evaluated here.

## Corrected Reading of Results

- The baseline four-model table should be read as an initial uniform-query benchmark, not as the final word on every model's best-practice performance.
- In the official-aligned multilingual rerun, `SigLIP2` becomes one of the strongest rows instead of one of the weakest.
- `Chinese-CLIP` should be judged with Chinese queries, not only English folder slugs.
- `CLIP` remains a strong English baseline, but its Chinese row is poor when the official English prompt shell is forced onto Chinese labels.
- `GME-2B` remains strong and stable, and still follows its official embedding API path cleanly.
