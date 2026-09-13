现在把search_web Tool 迁移到 OpenAI Agents SDK。

要求：

1. 保留原来的 search_web() Python function。
2. 使用 OpenAI Agents SDK 推荐的 function tool 机制。
3. 不使用 MCP。
4. 不使用 Hosted Web Search Tool。
5. 不修改 search_web 的业务逻辑。
6. Tool 的输入参数保持不变。
7. Tool 的返回值保持兼容。
8. Agent 是否调用 Tool 仍然由 LLM 决定。

完成后验证：

Case 1：
用户问一个不需要搜索的问题。

预期：
Agent 不调用 search_web。

Case 2：
用户要求研究一个需要外部信息的问题。

预期：
Agent 可以调用 search_web。

Case 3：
Tool 返回结果。

预期：
Tool result 自动回到 Agent，然后 Agent 继续执行。

不要自己实现 Tool dispatch。

不要自己实现 Agent loop。
