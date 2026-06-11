from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecom_qa.training import sft_api


class SftApiTest(unittest.TestCase):
    def test_message_content_uses_image_url_parts_for_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "item.jpg"
            image_path.write_bytes(b"fake-image")

            content = sft_api.message_content(prompt="answer this", image_paths=[image_path])

            self.assertIsInstance(content, list)
            self.assertEqual(content[0], {"type": "text", "text": "answer this"})
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_message_content_uses_plain_text_without_images(self) -> None:
        self.assertEqual(sft_api.message_content(prompt="answer this", image_paths=[]), "answer this")

    def test_response_format_prefers_json_schema(self) -> None:
        payload = sft_api.response_format(schema_name="direct", schema=sft_api.DIRECT_SCHEMA, schema_mode="json_schema")

        self.assertEqual(payload["type"], "json_schema")
        self.assertTrue(payload["json_schema"]["strict"])
        self.assertEqual(payload["json_schema"]["schema"], sft_api.DIRECT_SCHEMA)

    def test_response_format_can_fallback_to_json_object(self) -> None:
        self.assertEqual(
            sft_api.response_format(schema_name="direct", schema=sft_api.DIRECT_SCHEMA, schema_mode="json_object"),
            {"type": "json_object"},
        )

    def test_local_schema_validation_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValueError):
            sft_api.validate_payload_schema(
                {"first_think": "ok", "final_answer": "ok", "extra": "no"},
                sft_api.DIRECT_SCHEMA,
            )

    def test_local_schema_validation_accepts_web_schema(self) -> None:
        sft_api.validate_payload_schema(
            {
                "information_items": ["content only"],
                "after_tool_think": "evidence supports answer",
                "final_answer": "answer",
            },
            sft_api.web_schema(max_items=5, include_first_think=False),
        )


if __name__ == "__main__":
    unittest.main()
