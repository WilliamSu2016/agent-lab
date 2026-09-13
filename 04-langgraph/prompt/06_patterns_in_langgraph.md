现在把五种 Anthropic Agent Patterns 全部使用 LangGraph 表达。

创建：

src/patterns/

├── prompt_chaining.py
├── routing.py
├── parallelization.py
├── orchestrator_workers.py
└── evaluator_optimizer.py

要求：

每个 Pattern 都必须使用 StateGraph。

实现：

1. Prompt Chaining
2. Routing
3. Parallelization
4. Orchestrator-Workers
5. Evaluator-Optimizer

每个实现必须：

* 有明确 State
* 有明确 Nodes
* 有明确 Edges
* 如果需要，使用 Conditional Edges
* 可以生成 Mermaid graph
* 有独立测试

不要增加新的 Agent Pattern。

不要引入 Multi-Agent abstraction。

不要引入 MCP。

不要引入 Memory。

创建：

docs/06-PATTERNS-IN-LANGGRAPH.md

建立对照：

Anthropic Pattern
→ LangGraph Node
→ LangGraph Edge
→ LangGraph State
→ LangGraph Conditional Edge

重点回答：

为什么 LangGraph 非常适合表达 Anthropic Agent Patterns？
