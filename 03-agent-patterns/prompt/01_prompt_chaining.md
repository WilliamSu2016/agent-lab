现在实现第一个 Agent Pattern：

Prompt Chaining。

使用当前 OpenAI Agents SDK。

目标：

用户输入：

“比较 Python、TypeScript、Go 哪个更适合开发 AI Agent，并给出推荐。”

把任务拆成固定的 4 个 Agent Steps：

Step 1 — Planner
输出研究维度。

Step 2 — Researcher
根据研究维度使用 search_web Tool 获取资料。

Step 3 — Analyst
分析和比较研究结果。

Step 4 — Writer
生成最终答案。

要求：

1. 使用 OpenAI Agents SDK。
2. 不使用 Multi-Agent Handoff。
3. 不使用 Orchestrator。
4. 不使用动态任务分解。
5. 四个步骤必须由程序按照固定顺序执行。
6. 每一步的输入必须来自前一步的输出。
7. 给每一步定义清晰的 instructions。
8. 每一步尽量只完成一个职责。
9. 如果某一步失败，程序应该能够明确报告失败位置。
10. 保留 search_web Tool。
11. 不增加 MCP。
12. 不增加 Memory。

请输出：

src/prompt_chaining.py

以及：

tests/test_prompt_chaining.py

运行一次完整任务。

最后生成：

docs/01-PROMPT-CHAINING.md

解释：

* 为什么这是 Prompt Chaining？
* 哪部分是 deterministic？
* 哪部分由 LLM 完成？
* 为什么不能把它叫做 Autonomous Agent？
* 它相比单 Agent 有什么优势和缺点？
