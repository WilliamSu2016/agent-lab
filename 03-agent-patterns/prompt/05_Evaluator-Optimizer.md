现在实现第五个 Agent Pattern：

Evaluator-Optimizer。

目标：

让 Research Agent 生成一份 AI Agent Framework Comparison Report。

Generator Agent：

负责生成初稿。

Evaluator Agent：

负责评价初稿。

评价标准：

1. 是否回答用户问题
2. 是否覆盖主要比较维度
3. 是否存在明显事实错误
4. 是否有足够证据
5. 是否结构清晰
6. 是否遗漏重要内容

Evaluator 必须返回结构化结果：

{
"pass": true/false,
"score": 0-10,
"feedback": [...],
"missing_points": [...]
}

如果：

score >= 8

则结束。

否则：

Evaluator feedback
→ Generator
→ 新版本
→ Evaluator

最多循环 3 次。

要求：

1. 使用 OpenAI Agents SDK。
2. Generator 和 Evaluator 使用独立 Agent。
3. Evaluator 不直接修改答案。
4. Generator 根据 feedback 重新生成。
5. 必须设置最大 iteration。
6. 保存每一轮结果。
7. 最终输出最佳版本。
8. 不使用 Multi-Agent Handoff。
9. 不使用 Orchestrator-Workers。
10. 不使用 MCP。

创建：

src/evaluator_optimizer.py

tests/test_evaluator_optimizer.py

docs/05-EVALUATOR-OPTIMIZER.md

额外记录：

iteration 1 score
iteration 2 score
iteration 3 score

重点分析：

1. Evaluator 是否真的改善了结果？
2. 如果 score 没有提高怎么办？
3. Evaluator 自己判断错误怎么办？
4. 为什么必须设置最大 iteration？
5. 这个 Pattern 与 Agent Loop 有什么关系？
6. 它与普通 Evaluation 有什么区别？
