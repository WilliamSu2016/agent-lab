现在根据 docs/AGENT-DESIGN.md 实现最小 Agent Loop。

要求：

1. 不使用任何 Agent Framework。
2. 不实现 Web UI。
3. 不实现数据库。
4. 不实现 Memory。
5. 不实现 RAG。
6. 不实现 Multi-Agent。
7. 不实现 MCP。
8. 不实现 Streaming。

只实现核心循环：

LLM
→ 判断是否需要 Tool
→ Tool Call
→ 执行 Tool
→ Tool Result
→ 返回 LLM
→ 判断是否继续
→ Final Answer

请把 Agent Loop 独立实现，例如：

src/agent.py

要求：

* Agent loop 必须清晰可读
* 不要过度抽象
* 不要创建复杂 class hierarchy
* 每一步都打印日志
* 设置 max_turns，例如 5
* Tool 执行失败不能导致整个程序直接崩溃
* LLM API 错误需要有基本错误处理

完成后：

1. 运行一个最简单的测试
2. 确认 Agent 可以正常结束
3. 不要增加任何额外功能
