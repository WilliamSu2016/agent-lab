现在给现有 Agent 增加唯一一个 Tool：

search_web(query: str)

要求：

1. Tool 必须是普通 Python function。
2. 不使用 MCP。
3. 不使用任何 Agent Framework。
4. Tool 必须有明确的 name、description 和参数 schema。
5. LLM 必须能够决定是否调用这个 Tool。
6. Tool 返回结果后必须重新进入 Agent Loop。
7. 不允许程序代码自己决定“什么时候搜索”。
8. 是否调用 search_web 必须由 LLM 根据用户任务和 Tool description 决定。

请增加：

src/tools/search.py

并修改 Agent Loop。

同时增加测试：

tests/test_agent.py

至少测试：

1. 不需要搜索的问题 → Agent 不调用 Tool
2. 需要搜索的问题 → Agent 调用 Tool
3. Tool 返回结果后 → Agent 继续推理
4. 达到 max_turns → Agent 安全退出
5. Tool 报错 → Agent 能够处理错误
