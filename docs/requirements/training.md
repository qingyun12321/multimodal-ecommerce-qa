# Training Requirements

The training system should support tool-call SFT and future GRPO experiments.

Required capabilities:

- convert tool-call data into SFT-ready multi-turn rows
- validate paths, tags, answer fields, and split integrity
- run smoke training before main training
- sync metrics, configs, selected artifacts, and summaries to MLflow
- keep large checkpoints and full model weights outside git
- compare base, SFT, and SFT+GRPO behavior with shared evaluation data

Requirement summaries in this directory are generalized from external specifications and should not reference local source files.
