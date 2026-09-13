现在进入 AI Agent 学习Production Agent阶段.

现在要把现有 Multi-Agent Research Agent 演进成 Production Agent。

请先不要修改任何代码。

先检查当前项目：

1. Agent architecture
2. Graph architecture
3. State management
4. Tool layer
5. MCP integration
6. Memory
7. RAG
8. Multi-Agent orchestration
9. Evaluation
10. Error handling
11. Logging
12. Configuration
13. Security
14. Deployment

然后创建：

docs/00-PRODUCTION-ARCHITECTURE.md

回答：

### Reliability

1. 哪些地方可能失败？
2. 哪些失败可以 retry？
3. 哪些失败不能 retry？
4. 哪些操作必须 idempotent？
5. 哪些任务需要 checkpoint？
6. 哪些任务需要 timeout？

### Security

7. Prompt Injection 风险在哪里？
8. Tool Abuse 风险在哪里？
9. Agent 是否可能执行越权操作？
10. Secret 应该如何管理？
11. 用户之间的 State 是否可能泄漏？

### Observability

12. 如何知道 Agent 为什么失败？
13. 如何知道哪个 Tool 最慢？
14. 如何知道哪个 Agent 消耗 token 最多？
15. 如何知道某个版本上线后质量下降？

### Evaluation

16. 如何持续验证 Agent quality？
17. 如何进行 regression test？
18. 如何评估 tool selection？
19. 如何评估 multi-agent routing？

### Operations

20. 如何部署？
21. 如何扩容？
22. 如何处理长时间运行任务？
23. 如何恢复 crashed run？

最后输出：

Production Readiness Gap List

按照：

P0
P1
P2
P3

分类。

不要修改源代码。
