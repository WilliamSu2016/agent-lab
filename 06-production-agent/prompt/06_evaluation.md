现在实现 Production Agent 实验 ：

Evaluation。

建立 Production Evaluation Pipeline。

Evaluation Dataset 至少包含：

50 个真实或模拟任务。

每个任务定义：

input
expected_behavior
success_criteria
risk_level

评估：

1. Final answer quality
2. Tool selection
3. Tool arguments
4. Routing
5. Agent termination
6. Retry behavior
7. Safety
8. Groundedness
9. Citation / evidence quality
10. Cost
11. Latency

建立：

Offline Evaluation
+
Regression Evaluation

要求：

每次代码或 Prompt 修改：

run eval
→ compare baseline
→ detect regression

禁止只比较最终字符串。

创建：

evals/

dataset/
runner.py
metrics.py
baseline.py
regression.py

docs/06-EVALUATION.md

最终输出：

Evaluation Report

包含：

pass rate
failure rate
tool accuracy
routing accuracy
safety failure rate
average latency
average cost
