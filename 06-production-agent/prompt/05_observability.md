现在实现 Production Agent 实验 ：

Observability。

建立完整 Agent Observability。

必须能够回答：

1. 一个 Request 经过了哪些 Agent？
2. 调用了哪些 Tools？
3. 每个 Tool 花了多久？
4. 每次 LLM call 消耗多少 tokens？
5. 哪个 Agent 最昂贵？
6. 哪个 Tool 最容易失败？
7. Agent loop 执行了多少次？
8. 为什么最终失败？
9. 用户是谁？
10. 使用了哪个 Agent version？

每个 execution 必须拥有：

request_id
trace_id
user_id
session_id
agent_version
environment

建立：

Metrics：

latency
token_usage
tool_failure_rate
agent_failure_rate
retry_count
cost
success_rate

Logs：

structured JSON logs

Traces：

完整 Agent execution tree。

要求：

敏感数据不得默认进入日志。

创建：

src/observability/

logging.py
metrics.py
tracing.py

docs/05-OBSERVABILITY.md

最后生成一个：

Production Agent Observability Dashboard Specification。
