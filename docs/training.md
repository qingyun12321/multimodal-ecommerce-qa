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

SFT data conversion reads the repository-local tool-call splits from `data/tool_call/` by default and writes ms-swift formatted files to the configured training data directory. Remote training machines can override roots with environment variables such as `REPO_DIR`, `SFT_DATA_DIR`, `SFT_CHECKPOINT_ROOT`, and `MLFLOW_TRACKING_URI`.

Large checkpoints and full model weights should not be uploaded to this repository. MLflow should store configs, metrics, compact logs, selected evaluation artifacts, and paths or hashes for large external artifacts.
