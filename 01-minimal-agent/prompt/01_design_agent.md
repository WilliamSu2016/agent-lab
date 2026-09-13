我想从 0 到 1 学习 AI Agent 的本质。

现在不要写代码，也不要使用任何 Agent Framework。

我要构建一个最小的 Research Agent，需求如下：

用户输入一个研究问题，例如：

“Python 和 TypeScript 哪个更适合开发 AI Agent？”

Agent 能够：

1. 理解用户问题
2. 判断是否需要调用工具
3. 调用一个 web search 工具
4. 读取工具返回结果
5. 根据结果决定是否需要继续调用工具
6. 最终生成答案

约束：

* Python
* 不使用 LangGraph
* 不使用 OpenAI Agents SDK
* 不使用 CrewAI
* 不使用 AutoGen
* 不使用任何 Agent Framework
* 只使用 LLM API + Python functions
* Agent loop 必须由我们自己实现

请先不要写代码。

请分析并输出：

1. 什么是这个 Agent 的最小组成部分
2. Agent Loop 如何工作
3. LLM 和 Tool 如何交互
4. State 需要保存什么
5. Tool Call 的数据结构
6. 什么时候 Agent 应该继续循环
7. 什么时候 Agent 应该结束
8. 最大循环次数如何控制
9. 失败时如何处理
10. 最小项目目录结构

把设计写入：

docs/AGENT-DESIGN.md

完成后停止，不要实现。
