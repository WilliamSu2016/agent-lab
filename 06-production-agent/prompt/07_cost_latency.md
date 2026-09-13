现在实现 Production Agent 实验 ：

Cost & Latency Optimization。

对现有 Multi-Agent Research Agent 建立成本模型。

统计：

LLM calls
input tokens
output tokens
tool calls
subagent calls
retry calls
parallel workers

计算：

cost per request

latency per request

要求：

建立三个策略：

1. Fast
2. Balanced
3. Quality

Fast：

减少 Agent calls
减少 context
低成本 model

Balanced：

默认模式

Quality：

允许更多 research / review iterations

建立：

MAX_AGENT_ITERATIONS
MAX_WORKERS
MAX_TOKENS
MAX_COST
TIMEOUT

如果超过：

进入 controlled degradation。

例如：

Quality mode
→ timeout

自动降级：

Quality
→ Balanced
→ Fast
→ graceful failure

创建：

src/cost/

budget.py
policy.py
estimator.py

docs/07-COST-LATENCY.md
