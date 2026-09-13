现在实现 Multi-Agent 实验 3：

Supervisor / Manager。

使用 OpenAI Agents SDK 的 Agents-as-Tools 模式。

创建：

SupervisorAgent

以及：

ResearchAgent
CodingAgent
ReviewAgent

Supervisor 可以调用：

research_expert
coding_expert
review_expert

要求：

1. Supervisor 永远保持最终控制权。
2. Specialist 不直接回复用户。
3. Specialist 的输出返回给 Supervisor。
4. Supervisor 可以调用一个或多个 Specialist。
5. Supervisor 最终负责生成答案。
6. 使用 Agent.as_tool()。
7. 不使用 Handoff。
8. 不使用 LangGraph。
9. 不使用 MCP。
10. 保留 trace。

测试：

“比较 LangGraph 和 OpenAI Agents SDK，并给出推荐。”

要求 Supervisor 至少调用：

ResearchAgent
ReviewAgent

创建：

src/03_supervisor.py

tests/test_supervisor.py

docs/03-SUPERVISOR.md

重点比较：

Handoff
VS
Agents-as-Tools

回答：

1. 谁控制最终答案？
2. Specialist 是否直接面对用户？
3. 谁决定下一步？
4. 谁拥有 conversation control？
5. 哪种模式更适合 Customer Support？
6. 哪种模式更适合 Research？
