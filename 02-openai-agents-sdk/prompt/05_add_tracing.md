现在为这个OpenAI Agents SDK 版本的项目增加最基本的 Tracing。

要求：

1. 使用 SDK 内置 tracing。
2. 不修改 Agent 的业务逻辑。
3. 不增加第三方 observability framework。
4. 不增加 LangSmith。
5. 不增加 OpenTelemetry。
6. 不增加数据库。

运行一个需要 Tool Calling 的任务。

然后分析 Trace 中能够看到哪些信息，例如：

* Agent run
* LLM generation
* Tool call
* Tool result
* Agent continuation
* Final output

请生成：

docs/TRACING.md

解释：

1. 为什么没有框架的minimal agent很难观察 Agent Loop？
2. 本项目 SDK 如何帮助观察 Agent Loop？
3. Tracing 对 Debugging 有什么价值？
4. 如果 Agent 错误调用 Tool，我们如何定位问题？
