from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ecom_qa.retrieval.local_catalog import RetrievalItem
from ecom_qa.training import sft_build_multiturn_dataset as builder
from ecom_qa.training.sft_prompts import build_rag_fill_prompt


class FakeGenerator:
    def __init__(self, *, rag_action: str = "Answer") -> None:
        self.rag_action = rag_action

    def generate_json(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        cache_key: str,
        image_paths: list[Path],
        use_search: bool = False,
    ) -> dict[str, Any]:
        del prompt, schema, cache_key, image_paths
        if schema_name == "direct":
            return {
                "first_think": "图片和问题已提供足够信息，可以直接回答。",
                "final_answer": "绿色叶子。",
            }
        if schema_name == "rag":
            if self.rag_action == "Web_search":
                return {
                    "first_think": "问题先需要商品库信息。",
                    "after_tool_think": "RAG 信息没有给出完整答案，需要继续网页搜索。",
                    "next_action": "Web_search",
                    "web_search_query": "sunflower native origin",
                    "final_answer": "",
                }
            return {
                "first_think": "问题需要商品库中的价格和参数。",
                "after_tool_think": "检索信息给出了商品属性。",
                "next_action": "Answer",
                "web_search_query": "",
                "final_answer": "这款商品售价 99 元。",
            }
        if schema_name == "web":
            self.assertTrue(use_search)
            return {
                "first_think": "问题需要外部事实补充。",
                "information_items": ["Sunflowers are native to North America."],
                "after_tool_think": "网页证据说明了向日葵原产地。",
                "final_answer": "Sunflowers are native to North America.",
            }
        if schema_name == "web_after_rag":
            self.assertTrue(use_search)
            return {
                "information_items": ["Sunflowers are native to North America."],
                "after_tool_think": "网页证据补足了 RAG 中缺失的信息。",
                "final_answer": "Sunflowers are native to North America.",
            }
        raise AssertionError(f"unexpected schema_name={schema_name}")

    def assertTrue(self, value: bool) -> None:  # unittest-like helper for the fake
        if not value:
            raise AssertionError("expected true")


class FakeRAG:
    def search(self, query: str) -> list[RetrievalItem]:
        self.last_query = query
        return [
            RetrievalItem(
                title="测试商品",
                snippet="商品ID: p1\n标题: 测试商品\n价格: 99.00",
                source="local_catalog",
                metadata={"product_id": "p1"},
            )
        ]


class SftBuildMultiturnDatasetTest(unittest.TestCase):
    def test_image_path_prefers_repo_data_relative_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            data_dir = repo_dir / "data"
            image_path = data_dir / "images" / "flowers" / "item.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake")
            row = {
                "image_rel_path": "images/flowers/item.jpg",
                "metadata": {"qa_type": "multimodal"},
            }

            absolute_paths, output_paths = builder.image_paths_for_row(row, repo_dir=repo_dir, data_dir=data_dir)

            self.assertEqual(absolute_paths, [image_path])
            self.assertEqual(output_paths, ["data/images/flowers/item.jpg"])

    def test_direct_item_uses_script_rendered_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            data_dir = repo_dir / "data"
            image_path = data_dir / "images" / "flowers" / "item.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake")
            row = {
                "record_id": "tool_call:test:000001",
                "domain": "in_domain",
                "source_dataset": "ecom_qa_pairs",
                "query": "叶子是什么颜色？",
                "decision_type": "direct_answer",
                "search_tool": "none",
                "direct_answer": "绿色叶子。",
                "image_rel_path": "images/flowers/item.jpg",
                "metadata": {"qa_type": "multimodal", "source_answers": ["绿色叶子。"]},
            }

            item, meta = builder.build_item(
                row,
                repo_dir=repo_dir,
                data_dir=data_dir,
                generator=FakeGenerator(),
                rag=FakeRAG(),  # type: ignore[arg-type]
            )

            self.assertEqual(item["images"], ["data/images/flowers/item.jpg"])
            self.assertEqual(len(item["messages"]), 3)
            self.assertEqual(
                item["messages"][-1]["content"],
                "<Think>图片和问题已提供足够信息，可以直接回答。</Think>\n<Answer>绿色叶子。</Answer>",
            )
            self.assertEqual(meta["warnings"], [])

    def test_web_item_wraps_codex_evidence_as_information(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            data_dir = repo_dir / "data"
            row = {
                "record_id": "tool_call:test:000002",
                "domain": "out_of_domain",
                "source_dataset": "infoseek_sample",
                "query": "Where are sunflowers native to?",
                "decision_type": "tool_call",
                "search_tool": "Web_search",
                "search_input": "sunflower native origin",
                "use_grounding": False,
                "grounding_input": "No",
                "metadata": {"qa_type": "text_only"},
            }

            item, meta = builder.build_item(
                row,
                repo_dir=repo_dir,
                data_dir=data_dir,
                generator=FakeGenerator(),
                rag=FakeRAG(),  # type: ignore[arg-type]
            )

            self.assertEqual(item["images"], [])
            self.assertEqual(len(item["messages"]), 5)
            self.assertIn("<Web_search>sunflower native origin</Web_search>", item["messages"][2]["content"])
            self.assertIn("<Grounding>No</Grounding>", item["messages"][2]["content"])
            self.assertIn("<information>", item["messages"][3]["content"])
            self.assertIn("[1] Sunflowers are native to North America.", item["messages"][3]["content"])
            self.assertNotIn("标题:", item["messages"][3]["content"])
            self.assertNotIn("来源:", item["messages"][3]["content"])
            self.assertNotIn("https://", item["messages"][3]["content"])
            self.assertIn("<Answer>Sunflowers are native to North America.</Answer>", item["messages"][4]["content"])
            self.assertEqual(meta["warnings"], [])

    def test_rag_item_can_escalate_to_web_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            data_dir = repo_dir / "data"
            row = {
                "record_id": "tool_call:test:000005",
                "domain": "in_domain",
                "source_dataset": "ecom_qa_pairs",
                "query": "Where are sunflowers native to?",
                "decision_type": "tool_call",
                "search_tool": "RAG_search",
                "search_input": "sunflower native origin",
                "use_grounding": False,
                "grounding_input": "No",
                "metadata": {"qa_type": "text_only"},
            }

            item, meta = builder.build_item(
                row,
                repo_dir=repo_dir,
                data_dir=data_dir,
                generator=FakeGenerator(rag_action="Web_search"),
                rag=FakeRAG(),  # type: ignore[arg-type]
            )

            self.assertEqual(len(item["messages"]), 7)
            self.assertIn("<RAG_search>sunflower native origin</RAG_search>", item["messages"][2]["content"])
            self.assertIn("<Web_search>sunflower native origin</Web_search>", item["messages"][4]["content"])
            self.assertIn("[1] Sunflowers are native to North America.", item["messages"][5]["content"])
            self.assertIn("<Answer>Sunflowers are native to North America.</Answer>", item["messages"][6]["content"])
            self.assertEqual(meta["turn_count"], 3)
            self.assertEqual(meta["sample_kind"], "rag_to_web")
            self.assertEqual(meta["rag_next_action"], "Web_search")

    def test_validate_rejects_direct_answer_with_grounding(self) -> None:
        warnings = builder.validate_assistant_output(
            "<Think>x</Think>\n<Answer>y</Answer>\n<Grounding>No</Grounding>",
            round_name="first_direct",
        )

        self.assertIn("first_direct:unexpected_tool_tag", warnings)

    def test_missing_multimodal_image_is_not_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp)
            data_dir = repo_dir / "data"
            data_dir.mkdir()
            row = {
                "record_id": "tool_call:test:000003",
                "domain": "in_domain",
                "source_dataset": "ecom_qa_pairs",
                "query": "这是什么？",
                "decision_type": "direct_answer",
                "search_tool": "none",
                "direct_answer": "测试。",
                "metadata": {"qa_type": "multimodal"},
            }

            item, meta = builder.build_item(
                row,
                repo_dir=repo_dir,
                data_dir=data_dir,
                generator=FakeGenerator(),
                rag=FakeRAG(),  # type: ignore[arg-type]
            )

            self.assertEqual(item["images"], [""])
            self.assertFalse(meta["image_exists"])

    def test_tool_prompt_does_not_include_reference_answers(self) -> None:
        row = {
            "record_id": "tool_call:test:000004",
            "query": "品牌什么时候成立？",
            "metadata": {
                "qa_type": "multimodal",
                "source_primary_answer": "秘密答案",
                "source_answers": ["秘密答案"],
            },
        }

        prompt = build_rag_fill_prompt(record=row, information="<information>无</information>", answers=["秘密答案"])

        self.assertNotIn("秘密答案", prompt)
        self.assertNotIn("source_primary_answer", prompt)
        self.assertNotIn("source_answers", prompt)

    def test_unable_answer_is_canonicalized(self) -> None:
        self.assertEqual(
            builder.final_answer_output("信息不足", "无法回答，缺少相关信息"),
            "<Think>信息不足</Think>\n<Answer>无法回答，缺少相关信息。</Answer>",
        )

    def test_web_information_strips_source_like_content(self) -> None:
        rendered = builder.format_web_information(
            [
                "标题: Example page\n来源: https://example.com/item\nThe item supports Bluetooth.",
                "See https://example.com/spec for the specification.",
            ]
        )

        self.assertIn("The item supports Bluetooth.", rendered)
        self.assertNotIn("标题:", rendered)
        self.assertNotIn("来源:", rendered)
        self.assertNotIn("https://", rendered)

    def test_locked_jsonl_appender_deduplicates_concurrent_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            appender = builder.LockedJsonlAppender(path, key_fn=lambda row: str(row.get("id")))

            def append(index: int) -> None:
                item_id = str(index % 10)
                appender.append({"id": item_id, "value": index}, key=item_id)

            with ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(append, range(100)))

            rows = builder.read_jsonl(path)
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["id"] for row in rows}, {str(index) for index in range(10)})


if __name__ == "__main__":
    unittest.main()
