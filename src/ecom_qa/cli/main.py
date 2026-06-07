from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


COMMANDS: dict[tuple[str, str], str] = {
    ("data", "build-qa"): "ecom_qa.cli.commands.data_build_qa",
    ("data", "build-qa-preview"): "ecom_qa.cli.commands.data_build_qa_preview",
    ("data", "build-targeted-qa"): "ecom_qa.cli.commands.data_build_targeted_qa",
    ("data", "download-infoseek-sample"): "ecom_qa.cli.commands.data_download_infoseek_sample",
    ("data", "build-unified-vqa"): "ecom_qa.cli.commands.data_build_unified_vqa",
    ("data", "build-tool-call-labels"): "ecom_qa.cli.commands.data_build_tool_call_labels",
    ("data", "clean-tool-call-labels"): "ecom_qa.cli.commands.data_clean_tool_call_labels",
    ("data", "build-tool-call-splits"): "ecom_qa.cli.commands.data_build_tool_call_splits",
    ("retrieval", "evaluate-multilingual"): "ecom_qa.cli.commands.retrieval_evaluate_multilingual",
    ("retrieval", "evaluate-in-domain"): "ecom_qa.cli.commands.retrieval_evaluate_in_domain",
    ("retrieval", "evaluate-web"): "ecom_qa.cli.commands.retrieval_evaluate_web",
    ("retrieval", "evaluate-rerankers"): "ecom_qa.cli.commands.retrieval_evaluate_rerankers",
    ("infer", "tool-call"): "ecom_qa.cli.commands.infer_tool_call",
    ("infer", "multiturn-tool-call"): "ecom_qa.cli.commands.infer_multiturn_tool_call",
    ("train", "sft-build-data"): "ecom_qa.training.sft_build_multiturn_dataset",
    ("train", "sft-sync-mlflow"): "ecom_qa.training.sft_sync_mlflow",
    ("train", "sft-clean-mlflow"): "ecom_qa.training.mlflow_cleanup",
    ("train", "sft-estimate-smoke-time"): "ecom_qa.training.sft_estimate_smoke_time",
    ("report", "render-retrieval"): "ecom_qa.cli.commands.report_render_retrieval",
    ("report", "render-in-domain"): "ecom_qa.cli.commands.report_render_in_domain",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecom-qa",
        description="Run multimodal ecommerce QA data, retrieval, inference, training, and report workflows.",
    )
    parser.add_argument("group", nargs="?", choices=sorted({group for group, _ in COMMANDS}))
    parser.add_argument("command", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.group is None:
        parser.print_help()
        return 0
    choices = sorted(command for group, command in COMMANDS if group == ns.group)
    if ns.command in {None, "-h", "--help"}:
        print(f"usage: ecom-qa {ns.group} <command> [args]\n")
        print("commands:")
        for command in choices:
            print(f"  {command}")
        return 0
    module_name = COMMANDS.get((ns.group, ns.command))
    if module_name is None:
        parser.error(f"unknown command {ns.group} {ns.command!r}; expected one of: {', '.join(choices)}")
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [f"ecom-qa {ns.group} {ns.command}", *ns.args]
    try:
        result = module.main()
    finally:
        sys.argv = old_argv
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
