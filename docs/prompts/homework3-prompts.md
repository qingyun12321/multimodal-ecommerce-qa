# 电商问答-高阶-实验作业3 Prompt 摘录

来源：`/workspace/docs/电商问答-高阶-实验作业3.pdf`

说明：本 PDF 第 1 页有一段嵌入截图中的 `system_prompt` 示例，普通文本抽取不会包含。该段已根据截图人工转录。

## 1. GRPO 数据格式 System Prompt 示例（截图）

### English Original

```text
# 设置固定数据匹配格式

reasoning_start = "<start_working_out>"  # Acts as <think>
reasoning_end   = "<end_working_out>"    # Acts as </think>

solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"


system_prompt = \
f"""You are given a problem.

Think about the problem and provide your working out.

Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


"""
You are given a problem.

Think about the problem and provide your working out.

Place it between <start_working_out> and <end_working_out>.

Then, provide your solution between <SOLUTION></SOLUTION>
"""
```

### 中文版

```text
# 设置固定数据匹配格式

reasoning_start = "<start_working_out>"  # 作用相当于 <think>
reasoning_end   = "<end_working_out>"    # 作用相当于 </think>

solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"


system_prompt = \
f"""给你一个问题。

思考该问题并给出你的推理过程。

将推理过程放在 {reasoning_start} 和 {reasoning_end} 之间。
然后，将你的解答放在 {solution_start}{solution_end} 之间。"""


"""
给你一个问题。

思考该问题并给出你的推理过程。

将推理过程放在 <start_working_out> 和 <end_working_out> 之间。

然后，将你的解答放在 <SOLUTION></SOLUTION> 之间。
"""
```

## 2. 工具调用模型输出格式模板

### 中文原文

```text
模型输出内容：
<Think>思考过程</Think>
<RAG_search> 电商搜索输入 </RAG_search>
<Web_search> 网络搜索输入 </Web_search>
<Grounding> 裁剪目标名字 </Grounding>
<Answer>最终答案</Answer>
```

### English Version (translated from Chinese source)

```text
Model output content:
<Think>Reasoning process</Think>
<RAG_search> E-commerce search input </RAG_search>
<Web_search> Web search input </Web_search>
<Grounding> Name of the cropping target </Grounding>
<Answer>Final answer</Answer>
```

## 3. 多轮调用数据输出格式模板

### 中文原文

```text
<Think>思考过程</Think>
<RAG_search> 电商搜索输入 </RAG_search>
<Web_search> 网络搜索输入 </Web_search>
<Grounding> 裁剪目标名字 </Grounding>
<information> 检索返回信息 </information>
<Answer>最终答案</Answer>
```

### English Version (translated from Chinese source)

```text
<Think>Reasoning process</Think>
<RAG_search> E-commerce search input </RAG_search>
<Web_search> Web search input </Web_search>
<Grounding> Name of the cropping target </Grounding>
<information> Retrieved information </information>
<Answer>Final answer</Answer>
```

## 4. 奖励模型参考 Prompt（GPT-4o as reward model prompt）

### English Original

```text
You are a strict evaluation judge for short-answer matching. Given a model’s final answer and a list of gold answers, decide if the model’s answer matches ANY gold answer.

Rules:

Semantic Equivalence: Consider synonyms, paraphrases, and common aliases as valid matches. Example: "NYC" ≈ "New York City".

Ignore Trivial Differences: Do not penalize differences in articles, punctuation, word order, or casing. Example: "The Pacific Ocean" ≈ "pacific ocean".

At Least One Match: If the model’s answer aligns with ANY gold answer based on the rules, set match=true. Otherwise, match=false.

Numerical Flexibility: For answers involving numbers, an answer is a MATCH if it meets any of these criteria: (a) Range Inclusion: The model provides a range that contains the gold answer. Example: Model: "20 to 24", Gold: ["21"]. (b) Reasonable Rounding: The model’s answer is a reasonably rounded version of the gold answer. Example: Model: "176", Gold: ["176.124"]. (c) Unit Conversion: The model’s answer is equivalent but in a different unit. Example: Model: "3 km", Gold: ["3000 m"].

Substantive Difference: If the meaning, entity, or value differs in a way not covered by the rules above, it is NOT a match. Example: "Jupiter" ≠ "Mars". Example: "5.2" ≠ "52". Example: Model: "10-15", Gold: ["16"] → NO MATCH.

Output Format:
MATCH: true/false
REASON: A concise explanation focusing only on why the answer matches or does not match.
```

### 中文版

```text
你是一名严格的短答案匹配评估裁判。给定模型的最终答案和一组标准答案，判断模型答案是否匹配任意一个标准答案。

规则：

语义等价：将同义词、改写表达和常见别名视为有效匹配。示例："NYC" ≈ "New York City"。

忽略细微差异：不要因为冠词、标点、词序或大小写差异而扣分。示例："The Pacific Ocean" ≈ "pacific ocean"。

至少匹配一个：如果模型答案根据上述规则与任意一个标准答案一致，则设置 match=true。否则设置 match=false。

数字灵活性：对于涉及数字的答案，如果满足以下任一条件，则视为 MATCH：(a) 范围包含：模型给出的范围包含标准答案。示例：Model: "20 to 24", Gold: ["21"]。(b) 合理四舍五入：模型答案是标准答案的合理取整版本。示例：Model: "176", Gold: ["176.124"]。(c) 单位换算：模型答案与标准答案等价但使用不同单位。示例：Model: "3 km", Gold: ["3000 m"]。

实质性差异：如果含义、实体或数值存在上述规则未覆盖的差异，则不匹配。示例："Jupiter" ≠ "Mars"。示例："5.2" ≠ "52"。示例：Model: "10-15", Gold: ["16"] → NO MATCH。

输出格式：
MATCH: true/false
REASON: 简洁解释，仅说明答案为什么匹配或不匹配。
```

## 5. 最终问答大模型 Prompt

### English Version (corrected from PDF source)

```text
You have now received the results from your external search. Your goal is to analyze the search results to provide a final concise answer to the original question based on the image provided.

Original Question: {question}

Search Results: {information}

Follow the following process:

Briefly explain your reasoning process by analyzing the facts from the search results that are relevant to the question. Enclose this reasoning inside the <reason>...</reason> tags.

Provide the final, direct answer to the question between the <Answer> and </Answer> tags. If the information is insufficient, respond ONLY with: Unable to answer due to lack of relevant information.
```

### 中文版

```text
你现在已经收到了外部搜索的结果。你的目标是分析搜索结果，并基于所提供的图像，对原始问题给出最终的简洁回答。

原始问题：{question}

搜索结果：{information}

遵循以下流程：

通过分析搜索结果中与问题相关的事实，简要说明你的推理过程。将该推理过程包裹在 <reason>...</reason> 标签中。

在 <Answer> 和 </Answer> 标签之间给出对问题的最终直接答案。如果信息不足，仅回复：Unable to answer due to lack of relevant information.
```

## 6. 大模型评测参考 Prompt（LLM-as-judge）

### English Original

```text
You are an impartial judge evaluating a model’s answer for a visual question answering task. Your task is to determine if the Predicted Answer is correct by comparing it against the Ground-Truth Answer(s).

IMPORTANT INSTRUCTION: The Ground-Truth Answer(s) field may contain alternate correct answers. The predicted answer should be considered CORRECT if it is semantically equivalent to at least ONE of the provided ground-truth answers. Please respond with only [CORRECT] if the prediction is correct, and [INCORRECT] otherwise.

— Evaluation Details —

Question: {question}

Ground-Truth Answer(s): {references_for_prompt}

Predicted Answer: {candidate}
```

### 中文版

```text
你是一名公正的裁判，正在评估视觉问答任务中某个模型的答案。你的任务是通过将 Predicted Answer 与 Ground-Truth Answer(s) 进行比较，判断预测答案是否正确。

重要说明：Ground-Truth Answer(s) 字段可能包含多个可接受的正确答案。只要预测答案与所提供的任意一个标准答案在语义上等价，就应被视为 CORRECT。如果预测正确，请仅回复 [CORRECT]；否则回复 [INCORRECT]。

— 评估详情 —

问题：{question}

标准答案：{references_for_prompt}

预测答案：{candidate}
```
