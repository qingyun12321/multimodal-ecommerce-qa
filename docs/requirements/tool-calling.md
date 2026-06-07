# Tool-Calling Requirements

The tool-calling system should produce and evaluate structured model behavior for ecommerce QA.

Required capabilities:

- build unified VQA rows from in-domain and out-of-domain sources
- generate first-round tool-call labels
- clean and split tool-call labels into train/test sets
- infer tool-call outputs from VLMs
- parse and score direct answer, local retrieval, web search, and grounding behavior
- run multi-turn flows where retrieved information is passed back to the model

Requirement summaries in this directory are generalized from external specifications and should not reference local source files.
