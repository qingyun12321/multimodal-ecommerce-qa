# 电商问答-高阶-实验作业1 Prompt 摘录

来源：`/workspace/docs/电商问答-高阶-实验作业1.pdf`

## 1. 网络检索-信息总结模型 Prompt

### English Original

```text
System Message: You are a helpful assistant. Your task is to summarize the main content of the given web page in no more than five sentences. Your summary should cover the overall key points of the page, not just parts related to the user’s question.

Prompt: If any part of the content is helpful for answering the user’s question, be sure to include it clearly in the summary. Do not ignore relevant information, but also make sure the general structure and main ideas of the page are preserved. Your summary should be concise, factual, and informative.

Webpage Content (first 30000 characters) is: {webpage_content}

Question: {question}
```

### 中文版

```text
系统消息：你是一个有帮助的助手。你的任务是在不超过五句话内总结给定网页的主要内容。你的总结应覆盖该页面的整体关键点，而不仅仅是与用户问题相关的部分。

提示词：如果内容中的任何部分有助于回答用户的问题，请务必在总结中清楚包含它。不要忽略相关信息，同时也要确保保留页面的一般结构和主要观点。你的总结应简洁、真实且信息充分。

网页内容（前 30000 个字符）为：{webpage_content}

问题：{question}
```
