from __future__ import annotations

import json
from typing import Any


UNABLE_TO_ANSWER = "无法回答，缺少相关信息。"


def build_system_prompt(*, has_image_input: bool, domain: str) -> str:
    image_note = (
        "当前输入包含图片。"
        if has_image_input
        else "当前输入没有图片，因此不能使用图像线索；如果调用搜索工具，必须输出 <Grounding>No</Grounding>。"
    )
    domain_note = (
        "当前样本属于电商域内 VQA：如果问题需要商品页或本地商品库中的价格、规格、参数、售后、店铺、评分、材质、成分等信息，应优先使用 RAG_search。"
        if domain == "in_domain"
        else "当前样本属于域外开源 VQA，不属于本地电商商品库：不要使用 RAG_search；如果图片和常识不足以直接回答，应使用 Web_search。"
    )
    return (
        "你是一名专业的视觉助手。你的任务是基于给定图片回答用户问题，并在需要时规划工具调用。\n"
        f"{image_note}\n"
        f"{domain_note}\n\n"
        "第一轮必须先在 <Think>...</Think> 中写出简洁判断，然后选择一个动作：\n"
        "Action 1：如果视觉信息、可读文字、常识或自身知识已经足够回答，输出 <Answer>最终答案</Answer>。\n"
        "Action 2：如果需要电商商品库、商品页或本地商品属性信息，输出 <RAG_search>检索词</RAG_search>。\n"
        "Action 3：如果需要互联网外部信息，输出 <Web_search>检索词</Web_search>。\n"
        "选择 RAG_search 或 Web_search 时必须同时输出 <Grounding>目标</Grounding>；不需要裁剪时输出 <Grounding>No</Grounding>。\n"
        "工具返回会在后续轮次以 <information>...</information> 放入用户消息。"
        "后续轮次如果能够回答，必须输出 <Answer>最终答案</Answer>；仍缺少信息时输出固定答案："
        f"<Answer>{UNABLE_TO_ANSWER}</Answer>。\n"
        "不要输出 Markdown、JSON 或额外解释，所有标签必须完整闭合。"
    )


def build_direct_fill_prompt(*, record: dict[str, Any], answers: list[str]) -> str:
    return _base_fill_prompt(record=record, answers=answers) + (
        "\n任务：生成直接回答样本所需的无标签字段。\n"
        "输出 JSON 字段：first_think, final_answer。\n"
        "first_think 要简洁说明为什么可以直接回答。final_answer 是最终短答案。"
    )


def build_rag_fill_prompt(*, record: dict[str, Any], information: str, answers: list[str]) -> str:
    del answers
    return _base_fill_prompt(record=record, answers=[]) + (
        "\n任务：生成 RAG_search 多轮样本所需的无标签字段。\n"
        "你会收到脚本已构造好的 RAG information。"
        "如果 RAG information 支持回答，next_action 输出 Answer，并填写 final_answer。"
        "如果 RAG information 不足以回答，next_action 必须输出 Web_search，并填写 web_search_query 继续联网搜索；此时 final_answer 留空。"
        "不要在 RAG 后直接输出无法回答。\n"
        "输出 JSON 字段：first_think, after_tool_think, next_action, web_search_query, final_answer。\n"
        "first_think 要说明为什么需要 RAG_search。after_tool_think 要说明 RAG 信息是否足够，以及下一步动作。"
        "不要在任何字段里提及 reference answers、gold answer、标注答案或数据集答案。\n\n"
        f"RAG information:\n{information}"
    )


def build_web_fill_prompt(
    *,
    record: dict[str, Any],
    answers: list[str],
    max_items: int,
    search_query: str | None = None,
    previous_information: str | None = None,
    include_first_think: bool = True,
) -> str:
    del answers
    fields = (
        "first_think, information_items, after_tool_think, final_answer"
        if include_first_think
        else "information_items, after_tool_think, final_answer"
    )
    first_think_instruction = (
        "first_think 要说明为什么需要 Web_search；"
        if include_first_think
        else "本次 Web_search 发生在 RAG_search 之后，不需要输出 first_think；"
    )
    query_note = f"\nWeb search query:\n{search_query}" if search_query else ""
    previous_note = f"\nPrevious RAG information:\n{previous_information}" if previous_information else ""
    return _base_fill_prompt(record=record, answers=[]) + (
        "\n任务：生成 Web_search 多轮样本所需的无标签字段。\n"
        "请使用 Codex 的联网搜索能力查找能支持回答的信息。"
        f"最多使用 {max_items} 个网页/搜索结果，优先选择直接支持答案的页面，避免展开过多网页以节省 token。"
        f"输出 JSON 字段：{fields}。\n"
        "information_items 是检索证据内容列表，每个 item 只能是一段事实摘要字符串。"
        "不要在 information_items 中写标题、来源、URL、Markdown、引用编号或 XML/HTML 标签。"
        "每条事实摘要写 1-3 句，必须是后续回答可见的信息。"
        f"{first_think_instruction}after_tool_think 要说明如何根据 information_items 得到答案。"
        "final_answer 是最终短答案；如果 information_items 不支持回答，即使 reference answers 中有答案，也必须使用固定无法回答句。"
        "不要在任何字段里提及 reference answers、gold answer、标注答案或数据集答案。"
        f"{query_note}{previous_note}"
    )


def _base_fill_prompt(*, record: dict[str, Any], answers: list[str]) -> str:
    compact_record = {
        "record_id": record.get("record_id"),
        "domain": record.get("domain"),
        "source_dataset": record.get("source_dataset"),
        "query": record.get("query"),
        "decision_type": record.get("decision_type"),
        "search_tool": record.get("search_tool"),
        "search_input": record.get("search_input"),
        "use_grounding": record.get("use_grounding"),
        "grounding_input": record.get("grounding_input"),
        "metadata": _compact_metadata(record.get("metadata") or {}),
    }
    return (
        "你在为多模态电商问答 SFT 数据填充自然语言内容。"
        "脚本会统一生成所有 <Think>/<Answer>/<RAG_search>/<Web_search>/<Grounding>/<information> 标签，"
        "因此你的 JSON 字段值中严禁包含任何尖括号标签、Markdown 代码块或额外解释。\n"
        "答案语言应跟随用户问题；中文问题用简体中文，英文问题用英文。\n"
        "Reference answers 只能作为质量对齐目标，不能在输出中被提及，也不能替代当前对话可见的信息证据。\n"
        "不要提及 prompt、system、instruction、约束、脚本、任务要求、reference answers、gold answer、标注答案或数据集答案。\n"
        f"信息不足时 final_answer 必须精确输出：{UNABLE_TO_ANSWER}\n\n"
        "Record:\n"
        f"{json.dumps(compact_record, ensure_ascii=False, indent=2)}\n\n"
        "Reference answers:\n"
        f"{json.dumps(answers, ensure_ascii=False, indent=2)}"
    )


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "qa_type",
        "source_title",
        "product_id",
        "entity_text",
        "source_task_type",
        "data_split",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}
