现在实现第二个 Agent Pattern：

Routing。

继续使用 OpenAI Agents SDK。

创建一个 Research Router。

输入问题可能属于：

1. Technical
2. Business
3. General

创建：

Router Agent

然后根据 Router 的结果进入不同的 Specialist Agent：

Technical Researcher
Business Researcher
General Researcher

要求：

1. Router 负责分类。
2. 每个 Specialist 使用不同 instructions。
3. Technical Researcher 可以关注技术文档、GitHub、架构。
4. Business Researcher 可以关注市场、成本、竞争等。
5. General Researcher 处理普通知识问题。
6. Router 决定走哪条路径。
7. 不要让所有 Agent 都处理同一个问题。
8. 不使用 Orchestrator-Workers。
9. 不使用 Parallelization。
10. 不使用 Evaluator-Optimizer。
11. 保留 search_web Tool。
12. 不增加 MCP。
13. 不增加 Memory。

测试至少包括：

* “LangGraph 和 OpenAI Agents SDK 有什么区别？”
* “AI Agent SaaS 市场有哪些机会？”
* “什么是 RAG？”

创建：

src/routing.py

tests/test_routing.py

docs/02-ROUTING.md

重点解释：

为什么 Routing 比单个万能 Agent 更好？

什么时候 Routing 反而会增加不必要的复杂度？
