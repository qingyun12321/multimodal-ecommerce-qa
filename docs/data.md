# Data Policy

`data/` is the only repository-local data root. The default can be overridden with `ECOM_QA_DATA_ROOT` when running on another machine or mounted volume.

Recommended structure:

- `data/catalog/`: product metadata and catalog backups
- `data/images/`: ecommerce image gallery
- `data/qa/`: generated and curated QA rows
- `data/infoseek_sample/`: external sample rows and images
- `data/unified_vqa/`: merged VQA views
- `data/tool_call/`: tool-call labels, train/test splits, and summaries
- `data/tool_call_inference/`: first-round inference outputs
- `data/tool_call_multiturn/`: multi-turn runs and prompts
- `data/runs/`: ad hoc generated run outputs

`data/` is ignored by git. Commit manifests, compact summaries, and documentation only when they are needed for reproducibility without committing the full generated dataset.

## Path Conventions

Package code should read the data root through `ecom_qa.common.paths.data_root()` or a CLI option. Avoid hardcoded absolute machine paths in committed code.
