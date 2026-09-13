"""Agent Observability: logs, metrics, and traces.

Answers, for any given request, all ten questions the requirement lists
(see ``docs/05-OBSERVABILITY.md`` for the full write-up):

1. 一个 Request 经过了哪些 Agent?          -> tracing.agents_visited
2. 调用了哪些 Tools?                        -> tracing.tools_called
3. 每个 Tool 花了多久?                      -> tracing.tool_durations_ms / metrics.latency_stats
4. 每次 LLM call 消耗多少 tokens?           -> tracing.llm_calls / metrics.token_usage_total
5. 哪个 Agent 最昂贵?                       -> metrics.MetricsRegistry.most_expensive_agent
6. 哪个 Tool 最容易失败?                    -> metrics.MetricsRegistry.least_reliable_tool
7. Agent loop 执行了多少次?                 -> tracing.count_loop_iterations
8. 为什么最终失败?                          -> tracing.find_root_cause_failure
9. 用户是谁?                                -> ExecutionContext.user_id
10. 使用了哪个 Agent version?               -> ExecutionContext.agent_version

This package is intentionally independent of any one LLM/agent framework
(unlike the pre-existing root ``src/tracing.py``, which is specific to the
OpenAI Agents SDK experiment) so the same tracer/metrics/logging can wrap
any of this repo's other experiments (``multi_agent_research``, ``durable``,
``security``).
"""

from src.observability.logging import (
    JsonFormatter,
    SENSITIVE_FIELD_NAMES,
    configure_json_logging,
    log_event,
)
from src.observability.metrics import (
    AgentRunRecord,
    DEFAULT_PRICE_PER_1K_TOKENS_USD,
    LatencyStats,
    LLMCallRecord,
    MetricsRegistry,
    ToolCallRecord,
    estimate_cost_usd,
)
from src.observability.tracing import (
    ExecutionContext,
    Span,
    SpanKind,
    Tracer,
    agents_visited,
    bind_execution_context,
    count_loop_iterations,
    current_execution_context,
    current_span,
    find_root_cause_failure,
    jsonl_file_sink,
    llm_calls,
    new_execution_context,
    tool_durations_ms,
    tools_called,
    walk,
)

__all__ = [
    # tracing
    "ExecutionContext",
    "new_execution_context",
    "bind_execution_context",
    "current_execution_context",
    "current_span",
    "SpanKind",
    "Span",
    "Tracer",
    "jsonl_file_sink",
    "walk",
    "agents_visited",
    "tools_called",
    "tool_durations_ms",
    "llm_calls",
    "count_loop_iterations",
    "find_root_cause_failure",
    # metrics
    "MetricsRegistry",
    "ToolCallRecord",
    "LLMCallRecord",
    "AgentRunRecord",
    "LatencyStats",
    "estimate_cost_usd",
    "DEFAULT_PRICE_PER_1K_TOKENS_USD",
    # logging
    "JsonFormatter",
    "configure_json_logging",
    "log_event",
    "SENSITIVE_FIELD_NAMES",
]
