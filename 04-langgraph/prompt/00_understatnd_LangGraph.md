现在进入 AI Agent 学习路线LangGraph阶段：

这一阶段的目标不是马上开发一个复杂 Agent，而是理解：

“LangGraph 到底解决了什么问题？”

请先不要修改代码。

请阅读当前版本 LangGraph 官方文档和官方 GitHub，并重点研究：

1. StateGraph
2. State
3. Node
4. Edge
5. Conditional Edge
6. START
7. END
8. compile()
9. invoke()
10. stream()
11. Checkpoint / Persistence
12. Interrupt
13. Human-in-the-loop

然后回答：

1. LangGraph 与 OpenAI Agents SDK 的核心区别是什么？
2. LangGraph 为什么使用 Graph？
3. StateGraph 中 State 的作用是什么？
4. Node 是什么？
5. Edge 是什么？
6. Conditional Edge 是什么？
7. START / END 是什么？
8. compile() 做什么？
9. LangGraph 如何表达 Agent Loop？
10. LangGraph 如何表达 Routing？
11. LangGraph 如何表达 Parallelization？
12. LangGraph 如何保存 Agent State？
13. LangGraph 如何支持 Human-in-the-loop？

特别注意：

不要根据旧版本 LangGraph API 猜测。

以当前官方文档和 GitHub 为准。

创建：

docs/00-LANGGRAPH-CONCEPTS.md

不要修改源代码。
