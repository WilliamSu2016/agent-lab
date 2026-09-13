现在把src/prompt_chaining.py的 Prompt Chaining Pattern 使用 LangGraph 重写。

任务：

Research Agent

流程固定为：

START
→ planner
→ researcher
→ analyst
→ writer
→ END

要求：

1. 使用 StateGraph。
2. 每一步都是一个 Node。
3. Node 之间使用 Edge 连接。
4. 使用 State 传递中间结果。
5. 不使用 OpenAI Agents SDK。
6. 不使用 LangChain Agent abstraction。
7. 不使用 Multi-Agent。
8. 保留 search_web Tool。
9. 暂时不要 Conditional Edge。
10. 暂时不要 Persistence。
11. 暂时不要 Human-in-the-loop。

创建：

src/03_prompt_chaining_graph.py

tests/test_prompt_chaining_graph.py

docs/03-PROMPT-CHAINING-GRAPH.md

重点解释：

src/prompt_chaining.py的 Prompt Chaining：

Planner
→ Researcher
→ Analyst
→ Writer

如何映射成：

Node
→ Node
→ Node
→ Node

以及：

为什么 LangGraph 比普通函数调用更适合表达复杂 Workflow？
