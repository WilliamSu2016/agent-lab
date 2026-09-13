现在实现 Multi-Agent 实验 5：

Shared State。

使用 LangGraph。

创建：

ResearchAgent
AnalysisAgent
ReviewAgent

定义：

ResearchState

至少包含：

question
research_results
analysis
review
final_answer

流程：

ResearchAgent
↓
AnalysisAgent
↓
ReviewAgent
↓
Finalizer

要求：

1. Agent 之间通过 State 传递结果。
2. 不使用全局变量。
3. 每个 Agent 只修改自己负责的字段。
4. ReviewAgent 可以发现 AnalysisAgent 的问题。
5. 如果 Review 不通过，返回 AnalysisAgent。
6. 最大循环次数为 3。
7. 所有状态变化都可追踪。
8. 使用 LangGraph persistence。
9. 不使用 MCP。

创建：

src/05_shared_state.py

tests/test_shared_state.py

docs/05-SHARED-STATE.md

重点回答：

1. Shared State 和 Agent Memory 有什么区别？
2. Agent 之间是否应该共享完整 State？
3. 为什么应该限制 Agent 能修改的 State？
4. State Schema 为什么重要？
5. 如何避免 Agent A 修改 Agent B 的数据？
