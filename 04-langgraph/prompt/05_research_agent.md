现在实现一个完整的 LangGraph Research Agent。

目标：

用户输入研究问题，例如：

“比较 Python、TypeScript 和 Go 哪个更适合开发 AI Agent。”

Graph 必须能够：

1. 分析研究问题。
2. 制定研究计划。
3. 调用 search_web。
4. 根据搜索结果判断是否需要继续研究。
5. 汇总研究结果。
6. 生成最终答案。

使用：

StateGraph
State
Nodes
Edges
Conditional Edges

Graph 至少包含：

planner
agent
tools
finalize

要求：

1. Agent Loop 必须由 Graph 表达。
2. State 必须显式定义。
3. Tool Result 必须写入 State。
4. 设置最大循环次数。
5. 使用 LLM
6. 不使用 OpenAI Agents SDK。
7. 不使用 CrewAI。
8. 不使用 LangChain Agent abstraction。
9. 不使用 MCP。
10. 不使用 Memory。
11. 不使用 Multi-Agent。

创建：

src/05_research_agent.py

tests/test_research_agent.py

docs/05-RESEARCH-AGENT.md

额外输出 Graph Mermaid Diagram。
