我们现在进入学习OpenAI Agents SDK阶段。

项目当前是完成的“最小 AI Agent”，使用 Python 手写实现，没有使用任何 Agent Framework。

现在不要修改代码。

请阅读 OpenAI Agents SDK 最新官方文档，并重点理解：

1. Agent
2. Runner
3. Function tools
4. Agent loop
5. RunResult
6. Instructions
7. Tracing

重点回答：

1. OpenAI Agents SDK 中 Agent 是什么？
2. Runner 是什么？
3. Agent Loop 是由谁负责执行的？
4. Tool Call 是由谁负责 dispatch 的？
5. Tool Result 如何重新进入 Agent Loop？
6. max_turns 如何控制？
7. State 在 SDK 中如何处理？
8. Agent.instructions 对应当前项目中的什么？
9. @function_tool 对应当前项目中的什么？
10. Runner.run() 对应当前项目中的什么？
11. 哪些代码可以被 SDK 删除？
12. 哪些代码仍然需要我们自己编写？

特别注意：

不要根据旧版本 API 猜测。

以最新 OpenAI Agents SDK 官方文档和官方 GitHub repository 为准。

请把分析写入：

docs/OPENAI-AGENTS-SDK-MAPPING.md

不要修改任何源代码。
