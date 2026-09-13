现在实现第三个 Agent Pattern：

Parallelization。

使用 OpenAI Agents SDK。

目标：

比较：

Python
TypeScript
Go

不要让一个 Agent 顺序研究三个语言。

而是同时运行：

Python Researcher
TypeScript Researcher
Go Researcher

然后：

三个结果
→ Synthesizer
→ Final Answer

要求：

1. 三个 Researcher 相互独立。
2. 三个 Researcher 可以并行执行。
3. 每个 Researcher 只负责一个语言。
4. 使用 asyncio 实现并行。
5. 使用 OpenAI Agents SDK。
6. 最后使用一个 Synthesizer Agent 汇总结果。
7. 不使用 Orchestrator。
8. 不使用 Routing。
9. 不使用 Evaluator-Optimizer。
10. 不增加 MCP。
11. 不增加 Memory。

记录：

* Sequential latency
* Parallel latency

如果无法准确测量 latency，至少记录每个 Agent 的开始和结束时间。

创建：

src/parallelization.py

tests/test_parallelization.py

docs/03-PARALLELIZATION.md

重点回答：

1. 为什么三个任务可以并行？
2. 如果三个任务存在依赖关系，还能并行吗？
3. Parallelization 和 Multi-Agent 是什么关系？
4. Parallelization 和 Orchestrator-Workers 有什么区别？
