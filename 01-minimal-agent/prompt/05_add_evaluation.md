现在为 Agent 增加最小 Evaluation Harness。

不要增加任何 Agent Framework。

创建：

tests/evaluations/cases.json

至少设计 10 个测试案例，包括：

1. 简单问题，不需要 Tool
2. 需要一次搜索
3. 需要多次搜索
4. Tool 返回无结果
5. Tool 返回错误
6. 用户问题含糊
7. 用户要求多个信息
8. Agent 达到 max_turns
9. 搜索结果互相矛盾
10. 正常完成复杂问题

每个测试案例定义：

* input
* expected_behavior
* max_turns
* expected_tool_usage

然后创建 Evaluation Runner。

要求：

Evaluation 不仅检查程序是否运行成功，还检查：

* 是否正确使用 Tool
* 是否能够停止
* 是否产生最终答案
* 是否超过 max_turns
* 是否正确处理 Tool Error

不要追求复杂的 LLM-as-a-Judge。

先使用 deterministic assertions。
