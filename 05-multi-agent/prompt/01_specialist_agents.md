现在实现 Multi-Agent 实验 1：

Specialist Agents。

使用 LangGraph。

创建三个独立 Agent：

1. ResearchAgent
2. CodingAgent
3. ReviewAgent

要求：

ResearchAgent：
负责研究技术资料。

CodingAgent：
负责根据需求设计代码方案。

ReviewAgent：
负责检查研究结果和代码方案。

重要：

1. 三个 Agent 暂时不要互相调用。
2. 不使用 Handoff。
3. 不使用 Supervisor。
4. 不使用 Shared State。
5. 不使用 MCP。
6. 每个 Agent 使用独立 instructions。
7. 每个 Agent 明确输入和输出。
8. 为每个 Agent 创建独立测试。

创建：

src/specialists/

research_agent.py
coding_agent.py
review_agent.py

tests/

创建：

docs/01-SPECIALIST-AGENTS.md

重点回答：

1. 为什么 Specialist Agent 可能比一个万能 Agent 更好？
2. Specialist Agent 的 instructions 应该如何设计？
3. Specialist Agent 是否应该知道其他 Agent 的存在？
4. 一个 Agent 应该承担多少职责？
