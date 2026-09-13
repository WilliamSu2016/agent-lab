现在实现第四个 Agent Pattern：

Orchestrator-Workers。

目标：

用户可以提出任意 AI Agent Research 问题。

例如：

“全面分析 2026 年 AI Agent 开发技术栈。”

要求：

创建：

Orchestrator Agent

它负责：

1. 理解用户问题。
2. 自动拆解 Research Tasks。
3. 为每个 Task 创建清晰的 Worker Task。
4. 决定需要多少 Worker。
5. 收集 Worker Results。
6. 最后进行综合。

Worker Agent：

负责执行一个具体 Research Task。

重要要求：

不要提前写死：

Python
TypeScript
Go

这些任务。

必须让 Orchestrator 根据用户问题动态产生 Tasks。

使用 OpenAI Agents SDK。

可以使用 Agent-as-a-tool 或其他 SDK 推荐方式，但不要使用 Handoff 来掩盖 Orchestrator 逻辑。

要求：

* Worker Task 必须结构化。
* Worker Result 必须结构化。
* 设置最大 Worker 数量，例如 5。
* 防止无限生成 Worker。
* 最终必须由 Orchestrator 或 Synthesizer 汇总。

创建：

src/orchestrator_workers.py

tests/test_orchestrator_workers.py

docs/04-ORCHESTRATOR-WORKERS.md

重点解释：

1. 为什么这个模式不是简单的 Parallelization？
2. 谁决定 Worker 数量？
3. 谁决定 Worker 任务？
4. 哪些控制流是 LLM 决定的？
5. 最大 Worker 数量为什么必须由程序限制？
6. 什么情况下这个模式比单 Agent 更好？
