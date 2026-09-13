现在开始把当前项目 Minimal Agent 重构为 OpenAI Agents SDK 版本。

目标：

保持目前的用户体验和任务完全不变。

不要增加任何新功能。

第一阶段只迁移 Agent 定义。

要求：

1. 安装当前稳定版 openai-agents。
2. 使用：
   from agents import Agent, Runner
3. 创建一个 Agent。
4. 将当前项目中的 system instructions / agent instructions 迁移到 Agent.instructions。
5. 使用 Runner.run() 或 Runner.run_sync() 执行 Agent。
6. 暂时不要迁移 search_web Tool。
7. 暂时不要使用 Multi-Agent。
8. 不要使用 Handoff。
9. 不要使用 Sessions。
10. 不要使用 Guardrails。
11. 不要使用 MCP。
12. 不要使用 RAG。

先实现：

User
→ Agent
→ Runner
→ Final Answer

运行测试确认基础 Agent 工作。

同时记录：

“当前Minimal Agent项目哪些代码已经可以删除？”

不要修改 evaluation cases。
