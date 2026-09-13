现在完成第⑧阶段 LangGraph 学习。

请不要修改代码。

创建：

docs/LANGGRAPH-ARCHITECTURE-REVIEW.md

总结：

### Part 1 — 核心概念

解释：

State
Node
Edge
Conditional Edge
START
END
Graph
Compile
Invoke
Stream
Checkpoint
Interrupt

### Part 2 — 与第⑤步比较

Minimal Agent
VS
LangGraph

解释：

手写 while loop
如何映射成 Graph。

### Part 3 — 与第⑥步比较

OpenAI Agents SDK
VS
LangGraph

分别比较：

Agent Runtime
Control Flow
State
Tool Calling
Persistence
Human-in-the-loop
Tracing
Long-running tasks

### Part 4 — 与第⑦步比较

Anthropic Patterns
VS
LangGraph

建立：

Pattern → Graph Structure

### Part 5 — 最重要的问题

回答：

1. LangGraph 是 Agent Framework 还是 Workflow Framework？
2. 为什么 LangGraph 可以不使用 LangChain？
3. LangGraph 的核心价值是不是“让 LLM 更聪明”？
4. 为什么 State 是 LangGraph 的核心？
5. Conditional Edge 为什么对 Agent 很重要？
6. Persistence 为什么对 Production Agent 很重要？
7. 什么时候不应该使用 LangGraph？

最后用一句话总结：

“LangGraph 真正解决的问题是什么？”
