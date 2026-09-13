"""Tests for src/observability -- tracing, metrics, and JSON logging.

Covers:
* The 10 required questions the requirement lists (agents visited, tools
  called, tool durations, LLM token usage, most expensive agent, least
  reliable tool, loop iteration count, root-cause-of-failure, user
  identity, agent version).
* Mandatory identifier propagation (request_id/trace_id/user_id/
  session_id/agent_version/environment) on every span and log line.
* Sensitive-data redaction by default in logs (value-pattern based and
  field-name based).
* Each of the 7 named metrics: latency, token_usage, tool_failure_rate,
  agent_failure_rate, retry_count, cost, success_rate.
* Parallel fan-out via explicit ``parent=`` (contextvars limitation).
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from src.observability.logging import JsonFormatter, configure_json_logging, log_event
from src.observability.metrics import MetricsRegistry, estimate_cost_usd
from src.observability.tracing import (
    ExecutionContext,
    SpanKind,
    Tracer,
    agents_visited,
    bind_execution_context,
    count_loop_iterations,
    current_execution_context,
    find_root_cause_failure,
    llm_calls,
    new_execution_context,
    tool_durations_ms,
    tools_called,
)


def make_context(**overrides) -> ExecutionContext:
    defaults = dict(
        user_id="user-42",
        session_id="session-1",
        agent_version="v1.2.3",
        environment="test",
    )
    defaults.update(overrides)
    return new_execution_context(**defaults)


def build_sample_trace(tracer: Tracer, context: ExecutionContext):
    """Builds: workflow -> supervisor -> {tool search_web x2, llm call} ->
    planner (looped twice) -> worker_c (tool that raises)."""
    with bind_execution_context(context):
        with tracer.span(SpanKind.WORKFLOW, "research_request") as root:
            with tracer.span(SpanKind.AGENT, "supervisor"):
                with tracer.span(SpanKind.TOOL, "search_web", retry_count=0):
                    pass
                with tracer.span(
                    SpanKind.LLM,
                    "llm_call",
                    agent_name="supervisor",
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                ):
                    pass
            for _ in range(2):
                with tracer.span(SpanKind.AGENT, "planner"):
                    with tracer.span(SpanKind.TOOL, "search_web", retry_count=0):
                        pass
            try:
                with tracer.span(SpanKind.AGENT, "worker_c"):
                    with tracer.span(SpanKind.TOOL, "send_email", retry_count=1):
                        raise RuntimeError("smtp timeout")
            except RuntimeError:
                pass
        return root


# ---------------------------------------------------------------------------
# 1. Which agents did a request pass through?
# ---------------------------------------------------------------------------


def test_agents_visited_lists_every_distinct_agent_in_order():
    tracer = Tracer()
    ctx = make_context()
    root = build_sample_trace(tracer, ctx)

    assert agents_visited(root) == ["supervisor", "planner", "worker_c"]


# ---------------------------------------------------------------------------
# 2. Which tools were called?
# ---------------------------------------------------------------------------


def test_tools_called_lists_every_distinct_tool():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())

    assert tools_called(root) == ["search_web", "send_email"]


# ---------------------------------------------------------------------------
# 3. How long did each tool call take?
# ---------------------------------------------------------------------------


def test_tool_durations_records_every_individual_call():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())

    durations = tool_durations_ms(root)
    # search_web was called 3 times total (once under supervisor, twice under planner)
    assert len(durations["search_web"]) == 3
    assert len(durations["send_email"]) == 1
    assert all(d >= 0 for d in durations["search_web"])


# ---------------------------------------------------------------------------
# 4. How many tokens did each LLM call consume?
# ---------------------------------------------------------------------------


def test_llm_calls_expose_token_counts():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())

    calls = llm_calls(root)
    assert len(calls) == 1
    assert calls[0].attributes["prompt_tokens"] == 100
    assert calls[0].attributes["completion_tokens"] == 50


def test_metrics_token_usage_total_aggregates_across_calls():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    usage = registry.token_usage_total()
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


# ---------------------------------------------------------------------------
# 5. Which agent is most expensive?
# ---------------------------------------------------------------------------


def test_most_expensive_agent_ranks_by_attributed_llm_cost():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    name, cost = registry.most_expensive_agent()
    assert name == "supervisor"  # only agent with any attributed LLM cost
    assert cost == pytest.approx(estimate_cost_usd(100, 50))


# ---------------------------------------------------------------------------
# 6. Which tool is least reliable?
# ---------------------------------------------------------------------------


def test_least_reliable_tool_ranks_by_failure_rate():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    name, failure_rate = registry.least_reliable_tool()
    assert name == "send_email"  # its only call failed -> 100% failure rate
    assert failure_rate == 1.0


# ---------------------------------------------------------------------------
# 7. How many times did the agent loop iterate?
# ---------------------------------------------------------------------------


def test_count_loop_iterations_counts_repeated_agent_spans():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())

    assert count_loop_iterations(root, "planner") == 2
    assert count_loop_iterations(root, "supervisor") == 1
    assert count_loop_iterations(root, "nonexistent_agent") == 0


# ---------------------------------------------------------------------------
# 8. Why did it ultimately fail?
# ---------------------------------------------------------------------------


def test_find_root_cause_failure_returns_deepest_failed_span():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())

    failure = find_root_cause_failure(root)
    assert failure is not None
    assert failure.name == "send_email"
    assert failure.kind is SpanKind.TOOL
    assert "smtp timeout" in failure.error


def test_find_root_cause_failure_returns_none_when_nothing_failed():
    tracer = Tracer()
    ctx = make_context()
    with bind_execution_context(ctx):
        with tracer.span(SpanKind.WORKFLOW, "clean_request") as root:
            with tracer.span(SpanKind.TOOL, "search_web"):
                pass

    assert find_root_cause_failure(root) is None


# ---------------------------------------------------------------------------
# 9 & 10. Who is the user, and which agent version ran?
# ---------------------------------------------------------------------------


def test_execution_context_carries_user_and_agent_version():
    ctx = make_context(user_id="alice", agent_version="v9.9.9")
    assert ctx.user_id == "alice"
    assert ctx.agent_version == "v9.9.9"


def test_every_span_is_reachable_from_its_trace_execution_context():
    tracer = Tracer()
    ctx = make_context(user_id="bob", agent_version="v2.0.0")
    root = build_sample_trace(tracer, ctx)

    assert root.request_id == ctx.request_id
    assert root.trace_id == ctx.trace_id
    # every span in the tree shares the same trace/request id
    from src.observability.tracing import walk

    for span in walk(root):
        assert span.trace_id == ctx.trace_id
        assert span.request_id == ctx.request_id


# ---------------------------------------------------------------------------
# Mandatory identifier propagation
# ---------------------------------------------------------------------------


def test_bind_execution_context_is_scoped_and_resets():
    assert current_execution_context() is None
    ctx = make_context()
    with bind_execution_context(ctx):
        assert current_execution_context() is ctx
    assert current_execution_context() is None


def test_span_without_bound_context_raises():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.span(SpanKind.AGENT, "orphan"):
            pass


# ---------------------------------------------------------------------------
# Parallel fan-out: explicit parent= override (contextvars limitation)
# ---------------------------------------------------------------------------


def test_explicit_parent_override_for_fan_out():
    tracer = Tracer()
    ctx = make_context()
    with bind_execution_context(ctx):
        with tracer.span(SpanKind.WORKFLOW, "fanout_request") as root:
            with tracer.span(SpanKind.AGENT, "supervisor") as sup:
                pass
            # simulate a worker running on a separate thread that did NOT
            # inherit the contextvar current-span -- pass parent explicitly.
            with tracer.span(SpanKind.AGENT, "worker_a", context=ctx, parent=sup):
                pass

    assert len(root.children) == 1
    assert len(root.children[0].children) == 1
    assert root.children[0].children[0].name == "worker_a"


# ---------------------------------------------------------------------------
# The 7 named metrics
# ---------------------------------------------------------------------------


def test_metric_latency_stats_computed_from_durations():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    stats = registry.latency_stats("search_web", kind=SpanKind.TOOL)
    assert stats.count == 3
    assert stats.avg_ms >= 0
    assert stats.p99_ms >= stats.p50_ms


def test_metric_tool_failure_rate():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    assert registry.tool_failure_rate("search_web") == 0.0
    assert registry.tool_failure_rate("send_email") == 1.0
    assert registry.tool_failure_rate("never_called") == 0.0


def test_metric_agent_failure_rate():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    assert registry.agent_failure_rate("worker_c") == 1.0
    assert registry.agent_failure_rate("supervisor") == 0.0


def test_metric_retry_count():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    assert registry.retry_count("send_email") == 1
    assert registry.retry_count("search_web") == 0


def test_metric_cost_tracks_llm_call_cost():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    assert registry.total_cost_usd() == pytest.approx(estimate_cost_usd(100, 50))
    assert registry.total_cost_usd(agent_name="planner") == 0.0


def test_metric_success_rate_overall_and_scoped():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    # 1 failed span (send_email tool) out of: 4 tool calls + 1 llm call + 3 agent runs = 8 events, 1 failure counted at both the tool AND agent level
    overall = registry.success_rate()
    assert 0.0 < overall < 1.0
    assert registry.success_rate(kind=SpanKind.TOOL, name="search_web") == 1.0
    assert registry.success_rate(kind=SpanKind.TOOL, name="send_email") == 0.0


def test_metric_success_rate_with_no_data_is_vacuously_one():
    registry = MetricsRegistry()
    assert registry.success_rate() == 1.0


def test_summary_includes_all_named_metrics():
    tracer = Tracer()
    root = build_sample_trace(tracer, make_context())
    registry = MetricsRegistry()
    registry.record_from_trace(root)

    summary = registry.summary()
    assert "success_rate_overall" in summary
    assert "total_cost_usd" in summary
    assert "token_usage_total" in summary
    assert "most_expensive_agent" in summary
    assert "least_reliable_tool" in summary
    assert "search_web" in summary["tools"]
    assert "supervisor" in summary["agents"]


# ---------------------------------------------------------------------------
# Structured JSON logging + sensitive-data redaction by default
# ---------------------------------------------------------------------------


def make_logger(name: str):
    stream = io.StringIO()
    logger = configure_json_logging(logger_name=name, stream=stream)
    return logger, stream


def test_json_log_line_is_valid_json_with_mandatory_identifiers():
    logger, stream = make_logger("test.mandatory_ids")
    ctx = make_context(user_id="carol", agent_version="v3.3.3")

    with bind_execution_context(ctx):
        logger.info("agent started")

    line = json.loads(stream.getvalue().strip())
    assert line["message"] == "agent started"
    assert line["request_id"] == ctx.request_id
    assert line["trace_id"] == ctx.trace_id
    assert line["user_id"] == "carol"
    assert line["session_id"] == ctx.session_id
    assert line["agent_version"] == "v3.3.3"
    assert line["environment"] == ctx.environment


def test_json_log_line_has_null_identifiers_when_no_context_bound():
    logger, stream = make_logger("test.no_context")
    logger.info("no context bound here")

    line = json.loads(stream.getvalue().strip())
    assert line["request_id"] is None
    assert line["user_id"] is None


def test_log_redacts_sensitive_values_in_message_by_default():
    logger, stream = make_logger("test.redact_message")
    logger.info("user key is sk-abcdefghijklmnopqrstuvwxyz123456")

    line = json.loads(stream.getvalue().strip())
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in line["message"]
    assert "REDACTED" in line["message"]


def test_log_redacts_sensitive_field_names_in_extra():
    logger, stream = make_logger("test.redact_fields")
    log_event(logger, logging.INFO, "login attempt", password="hunter2", username="carol")

    line = json.loads(stream.getvalue().strip())
    assert line["password"] == "[REDACTED:field-name]"
    assert line["username"] == "carol"


def test_log_redacts_sensitive_values_in_extra_fields_recursively():
    logger, stream = make_logger("test.redact_nested")
    log_event(
        logger,
        logging.INFO,
        "tool response",
        tool_result={"contact": "reach me at bob@example.com"},
    )

    line = json.loads(stream.getvalue().strip())
    assert "bob@example.com" not in json.dumps(line["tool_result"])
    assert "REDACTED" in json.dumps(line["tool_result"])


def test_configure_json_logging_is_idempotent_no_duplicate_handlers():
    configure_json_logging(logger_name="test.idempotent")
    logger = configure_json_logging(logger_name="test.idempotent")
    assert len(logger.handlers) == 1


def test_json_formatter_can_be_used_directly_without_configure_helper():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="direct", level=logging.WARNING, pathname=__file__, lineno=1, msg="raw usage", args=(), exc_info=None
    )
    rendered = json.loads(formatter.format(record))
    assert rendered["level"] == "WARNING"
    assert rendered["message"] == "raw usage"
