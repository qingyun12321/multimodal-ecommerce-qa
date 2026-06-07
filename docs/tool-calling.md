# Tool Calling

Tool-call workflows build and evaluate labels for four behavior families:

- direct answer
- local retrieval
- web search
- grounding

The multi-turn protocol injects retrieved or summarized evidence inside `<information>` blocks before the model continues or answers.

Primary commands:

```bash
uv run ecom-qa data build-tool-call-labels --help
uv run ecom-qa data clean-tool-call-labels --help
uv run ecom-qa data build-tool-call-splits --help
uv run ecom-qa infer tool-call --help
uv run ecom-qa infer multiturn-tool-call --help
```

Tool-call train/test data lives under `data/tool_call/`. First-round model outputs live under `data/tool_call_inference/`; multi-turn outputs live under `data/tool_call_multiturn/`.
