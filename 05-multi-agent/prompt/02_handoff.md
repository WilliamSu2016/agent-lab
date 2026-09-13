现在实现 Multi-Agent 实验 2：

Handoff。

使用 OpenAI Agents SDK。

创建：

TriageAgent

ResearchAgent

CodingAgent

TriageAgent 负责判断用户需求属于：

research
或
coding

然后：

research → ResearchAgent

coding → CodingAgent

要求：

1. 使用 OpenAI Agents SDK。
2. 使用 handoffs。
3. ResearchAgent 接管 research 请求。
4. CodingAgent 接管 coding 请求。
5. TriageAgent 不生成最终答案。
6. Specialist Agent 接管后负责最终回答。
7. 不使用 Supervisor。
8. 不使用 agents-as-tools。
9. 添加 handoff_description。
10. 保留完整 trace。

测试：

“比较 LangGraph 和 OpenAI Agents SDK。”

“帮我写一个 LangGraph Agent。”

创建：

src/02_handoff.py

tests/test_handoff.py

docs/02-HANDOFF.md

重点观察 Trace：

Triage
→
transfer_to_research
→
ResearchAgent

以及：

Triage
→
transfer_to_coding
→
CodingAgent

重点回答：

为什么 Handoff 是“控制权转移”？

为什么这和普通 Tool Calling 不一样？
