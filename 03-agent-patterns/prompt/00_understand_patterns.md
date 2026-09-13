当前项目使用 OpenAI Agents SDK，但本阶段的学习目标不是学习 SDK API，而是学习 Agent Architecture Patterns。

请阅读并理解 Anthropic 官方文章：

“Building effective agents”
https://www.anthropic.com/engineering/building-effective-agents

重点研究以下五种模式：

1. Prompt Chaining
2. Routing
3. Parallelization
4. Orchestrator-Workers
5. Evaluator-Optimizer

请回答：

1. 每种 pattern 解决什么问题？
2. 每种 pattern 的控制流是什么？
3. 哪些步骤是 deterministic 的？
4. 哪些步骤由 LLM 决定？
5. 哪些 pattern 属于 Workflow？
6. 哪些地方开始接近 Agent？
7. 每种 pattern 的优势是什么？
8. 每种 pattern 的主要缺点是什么？
9. 什么情况下不应该使用它？
10. 哪种 pattern 最适合 Research Agent？

不要修改任何代码。

创建：

docs/PATTERN-TAXONOMY.md
