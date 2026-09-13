现在实现 Multi-Agent 实验 4：

Parallel Multi-Agent Research。

使用 LangGraph。

任务：

“全面研究 AI Agent Framework。”

Supervisor/Planner 首先决定需要研究哪些方面。

例如可能产生：

OpenAI Agents SDK
LangGraph
Anthropic Agent Patterns
MCP

但是不要把这些任务写死。

要求：

1. Planner 动态产生 Research Tasks。
2. 每个 Task 创建一个 Research Agent。
3. Research Agents 并行运行。
4. 每个 Research Agent 只负责自己的 Task。
5. 每个 Agent 返回结构化 ResearchResult。
6. 最后由 Synthesizer 汇总。
7. 使用 LangGraph StateGraph。
8. 使用 parallel branches / fan-out / fan-in。
9. 设置最大 Worker 数量。
10. 设置最大执行时间。
11. 单个 Worker 失败不能导致整个任务无限等待。
12. 不使用 MCP。
13. 不使用 Memory。

Graph：

START
↓
Planner
↓
Dynamic Research Tasks
↓
Parallel Workers
↓
Synthesizer
↓
END

创建：

src/04_parallel_multi_agent.py

tests/test_parallel_multi_agent.py

docs/04-PARALLEL-MULTI-AGENT.md

重点回答：

1. 这和普通Parallelization pattern有什么不同？
2. 为什么现在叫 Multi-Agent？
3. Worker 是否共享同一个 State？
4. Worker 之间应该互相通信吗？
5. 为什么 Fan-in 是必要的？
6. 如何防止 Worker 数量失控？
