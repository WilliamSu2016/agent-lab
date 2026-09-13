"""Tests for src/cost — budget caps, the Fast/Balanced/Quality policies,
controlled degradation, and the retrospective/prospective cost model."""

from __future__ import annotations

import pytest

from src.cost.budget import HARD_CEILING, Budget, BudgetExceededError, BudgetTracker, BudgetUsage
from src.cost.estimator import CostPrediction, RequestCostReport, estimate_request_cost, predict_cost
from src.cost.policy import (
    BALANCED_POLICY,
    DEFAULT_MODE,
    DEFAULT_POLICIES,
    DEGRADATION_ORDER,
    FAST_POLICY,
    QUALITY_POLICY,
    ExecutionMode,
    GracefulFailure,
    next_degraded_mode,
    run_with_degradation,
)
from src.observability.tracing import Span, SpanKind, Tracer, bind_execution_context, new_execution_context


# ---------------------------------------------------------------------------
# Budget / HardCeiling.
# ---------------------------------------------------------------------------


def test_budget_within_hard_ceiling_constructs_fine():
    budget = Budget(max_agent_iterations=2, max_workers=3, max_tokens=1000, max_cost_usd=0.1, timeout_seconds=10.0)
    assert budget.max_workers == 3


def test_budget_exceeding_hard_ceiling_raises():
    with pytest.raises(ValueError, match="HARD_CEILING"):
        Budget(
            max_agent_iterations=HARD_CEILING.max_agent_iterations + 1,
            max_workers=1,
            max_tokens=1,
            max_cost_usd=0.01,
            timeout_seconds=1.0,
        )


def test_budget_rejects_non_positive_values():
    with pytest.raises(ValueError):
        Budget(max_agent_iterations=0, max_workers=1, max_tokens=1, max_cost_usd=0.01, timeout_seconds=1.0)


def test_all_default_policies_fit_under_hard_ceiling():
    for policy in DEFAULT_POLICIES.values():
        assert policy.budget.max_agent_iterations <= HARD_CEILING.max_agent_iterations
        assert policy.budget.max_workers <= HARD_CEILING.max_workers
        assert policy.budget.max_tokens <= HARD_CEILING.max_tokens
        assert policy.budget.max_cost_usd <= HARD_CEILING.max_cost_usd
        assert policy.budget.timeout_seconds <= HARD_CEILING.timeout_seconds


# ---------------------------------------------------------------------------
# BudgetTracker.
# ---------------------------------------------------------------------------


def test_tracker_raises_on_exceeding_max_agent_iterations():
    tracker = BudgetTracker(Budget(max_agent_iterations=2, max_workers=5, max_tokens=1000, max_cost_usd=1.0, timeout_seconds=10.0))
    tracker.record_iteration()
    tracker.record_iteration()
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_iteration()
    assert exc_info.value.dimension == "max_agent_iterations"


def test_tracker_raises_on_exceeding_max_workers():
    tracker = BudgetTracker(Budget(max_agent_iterations=2, max_workers=3, max_tokens=1000, max_cost_usd=1.0, timeout_seconds=10.0))
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_workers(4)
    assert exc_info.value.dimension == "max_workers"


def test_tracker_raises_on_exceeding_max_tokens():
    tracker = BudgetTracker(Budget(max_agent_iterations=2, max_workers=3, max_tokens=100, max_cost_usd=1.0, timeout_seconds=10.0))
    tracker.record_tokens(60)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_tokens(60)
    assert exc_info.value.dimension == "max_tokens"


def test_tracker_raises_on_exceeding_max_cost():
    tracker = BudgetTracker(Budget(max_agent_iterations=2, max_workers=3, max_tokens=1000, max_cost_usd=0.05, timeout_seconds=10.0))
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record_cost(0.10)
    assert exc_info.value.dimension == "max_cost_usd"


def test_tracker_raises_on_timeout():
    usage = BudgetUsage(started_at=0.0)  # started far in the past -> immediately timed out
    tracker = BudgetTracker(Budget(max_agent_iterations=2, max_workers=3, max_tokens=1000, max_cost_usd=1.0, timeout_seconds=1.0), usage=usage)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_timeout()
    assert exc_info.value.dimension == "timeout_seconds"


def test_tracker_check_all_revalidates_after_swapping_to_smaller_budget():
    tracker = BudgetTracker(Budget(max_agent_iterations=5, max_workers=10, max_tokens=10_000, max_cost_usd=1.0, timeout_seconds=100.0))
    tracker.record_iteration()
    tracker.record_iteration()
    tracker.record_iteration()  # 3 iterations used, well within the current budget

    smaller_budget = Budget(max_agent_iterations=2, max_workers=10, max_tokens=10_000, max_cost_usd=1.0, timeout_seconds=100.0)
    tracker.budget = smaller_budget
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_all()
    assert exc_info.value.dimension == "max_agent_iterations"


# ---------------------------------------------------------------------------
# The three policies.
# ---------------------------------------------------------------------------


def test_fast_policy_reduces_calls_context_and_uses_cheap_model():
    assert FAST_POLICY.max_agent_iterations < BALANCED_POLICY.max_agent_iterations
    assert FAST_POLICY.max_workers < BALANCED_POLICY.max_workers
    assert FAST_POLICY.max_context_chars < BALANCED_POLICY.max_context_chars
    assert FAST_POLICY.model_name_override is not None  # low-cost model override


def test_quality_policy_allows_more_iterations_and_workers_than_balanced():
    assert QUALITY_POLICY.max_agent_iterations > BALANCED_POLICY.max_agent_iterations
    assert QUALITY_POLICY.max_workers >= BALANCED_POLICY.max_workers
    assert QUALITY_POLICY.max_context_chars > BALANCED_POLICY.max_context_chars


def test_balanced_is_the_default_mode():
    assert DEFAULT_MODE is ExecutionMode.BALANCED


def test_degradation_ladder_order():
    assert DEGRADATION_ORDER == (ExecutionMode.QUALITY, ExecutionMode.BALANCED, ExecutionMode.FAST)
    assert next_degraded_mode(ExecutionMode.QUALITY) is ExecutionMode.BALANCED
    assert next_degraded_mode(ExecutionMode.BALANCED) is ExecutionMode.FAST
    assert next_degraded_mode(ExecutionMode.FAST) is None


# ---------------------------------------------------------------------------
# Controlled degradation runner.
# ---------------------------------------------------------------------------


def test_run_with_degradation_succeeds_immediately_when_start_mode_works():
    def execute(policy):
        return f"ok:{policy.mode.value}"

    outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY)

    assert outcome.succeeded
    assert outcome.final_mode is ExecutionMode.QUALITY
    assert outcome.result == "ok:quality"
    assert [a.mode for a in outcome.attempts] == [ExecutionMode.QUALITY]


def test_run_with_degradation_falls_back_quality_to_balanced():
    def execute(policy):
        if policy.mode is ExecutionMode.QUALITY:
            raise TimeoutError("quality mode timed out")
        return f"ok:{policy.mode.value}"

    outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY)

    assert outcome.succeeded
    assert outcome.final_mode is ExecutionMode.BALANCED
    assert [a.mode for a in outcome.attempts] == [ExecutionMode.QUALITY, ExecutionMode.BALANCED]
    assert outcome.attempts[0].succeeded is False
    assert outcome.attempts[1].succeeded is True


def test_run_with_degradation_falls_all_the_way_to_fast():
    def execute(policy):
        if policy.mode in (ExecutionMode.QUALITY, ExecutionMode.BALANCED):
            raise TimeoutError(f"{policy.mode.value} timed out")
        return "ok:fast"

    outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY)

    assert outcome.succeeded
    assert outcome.final_mode is ExecutionMode.FAST
    assert outcome.result == "ok:fast"


def test_run_with_degradation_returns_graceful_failure_when_fast_also_fails():
    def execute(policy):
        raise BudgetExceededError("max_cost_usd", 5.0, policy.budget.max_cost_usd)

    outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY)

    assert not outcome.succeeded
    assert outcome.final_mode is None
    assert isinstance(outcome.graceful_failure, GracefulFailure)
    assert [a.mode for a in outcome.attempts] == [ExecutionMode.QUALITY, ExecutionMode.BALANCED, ExecutionMode.FAST]
    assert all(not a.succeeded for a in outcome.attempts)
    assert "GracefulFailure" in outcome.graceful_failure.to_trace_line()


def test_run_with_degradation_calls_on_degrade_hook_with_reason():
    events = []

    def execute(policy):
        if policy.mode is ExecutionMode.QUALITY:
            raise TimeoutError("boom")
        return "ok"

    def on_degrade(from_mode, to_mode, reason):
        events.append((from_mode, to_mode, reason))

    outcome = run_with_degradation(execute, start_mode=ExecutionMode.QUALITY, on_degrade=on_degrade)

    assert outcome.succeeded
    assert events == [(ExecutionMode.QUALITY, ExecutionMode.BALANCED, "boom")]


# ---------------------------------------------------------------------------
# Cost model: retrospective (from a trace).
# ---------------------------------------------------------------------------


def _build_sample_trace() -> Span:
    tracer = Tracer()
    ctx = new_execution_context(user_id="u1", session_id="s1", agent_version="v1", environment="test")
    with bind_execution_context(ctx):
        with tracer.span(SpanKind.WORKFLOW, "research_request") as root:
            with tracer.span(SpanKind.AGENT, "planner"):
                with tracer.span(SpanKind.LLM, "llm_call", prompt_tokens=500, completion_tokens=200):
                    pass
            with tracer.span(SpanKind.AGENT, "planner"):
                for i in range(3):
                    with tracer.span(SpanKind.AGENT, "research_worker", retry_count=1 if i == 0 else 0):
                        with tracer.span(SpanKind.TOOL, "search_web"):
                            pass
                        with tracer.span(SpanKind.LLM, "llm_call", prompt_tokens=100, completion_tokens=50):
                            pass
            with tracer.span(SpanKind.AGENT, "synthesizer"):
                with tracer.span(SpanKind.LLM, "llm_call", prompt_tokens=300, completion_tokens=150):
                    pass
        return root


def test_estimate_request_cost_counts_every_required_dimension():
    root = _build_sample_trace()
    report = estimate_request_cost(root)

    assert isinstance(report, RequestCostReport)
    # 1 planner LLM call + 3 worker LLM calls + 1 synthesizer LLM call = 5
    assert report.llm_calls == 5
    assert report.input_tokens == 500 + 3 * 100 + 300
    assert report.output_tokens == 200 + 3 * 50 + 150
    assert report.tool_calls == 3
    # AGENT spans: planner x2, research_worker x3, synthesizer x1 = 6
    assert report.subagent_calls == 6
    assert report.retry_calls == 1
    assert report.parallel_workers == 3  # widest fan-out under one parent
    assert report.cost_usd > 0
    assert report.latency_ms >= 0


def test_estimate_request_cost_total_tokens_property():
    root = _build_sample_trace()
    report = estimate_request_cost(root)
    assert report.total_tokens == report.input_tokens + report.output_tokens


def test_estimate_request_cost_handles_empty_trace():
    tracer = Tracer()
    ctx = new_execution_context(user_id="u", session_id="s", agent_version="v", environment="test")
    with bind_execution_context(ctx):
        with tracer.span(SpanKind.WORKFLOW, "empty_request") as root:
            pass

    report = estimate_request_cost(root)
    assert report.llm_calls == 0
    assert report.tool_calls == 0
    assert report.parallel_workers == 0
    assert report.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Cost model: prospective (from a policy, before running).
# ---------------------------------------------------------------------------


def test_predict_cost_quality_predicts_more_than_fast():
    fast_prediction = predict_cost(FAST_POLICY)
    quality_prediction = predict_cost(QUALITY_POLICY)

    assert isinstance(fast_prediction, CostPrediction)
    assert quality_prediction.predicted_llm_calls > fast_prediction.predicted_llm_calls
    assert quality_prediction.predicted_cost_usd >= fast_prediction.predicted_cost_usd
    assert quality_prediction.predicted_latency_ms >= fast_prediction.predicted_latency_ms


def test_predict_cost_never_exceeds_the_policys_own_budget():
    for policy in DEFAULT_POLICIES.values():
        prediction = predict_cost(policy)
        assert prediction.predicted_cost_usd <= policy.budget.max_cost_usd
        assert prediction.predicted_tokens <= policy.budget.max_tokens
        assert prediction.predicted_latency_ms <= policy.budget.timeout_seconds * 1000.0


def test_predict_cost_parallel_workers_do_not_multiply_latency():
    # Predicted latency must reflect running workers in *parallel*, i.e. it
    # should not scale linearly with max_workers the way predicted cost does.
    prediction = predict_cost(BALANCED_POLICY)
    cost_per_iteration = prediction.predicted_cost_usd / BALANCED_POLICY.max_agent_iterations
    latency_per_iteration = prediction.predicted_latency_ms / BALANCED_POLICY.max_agent_iterations
    # cost scales with (workers + 3) calls per iteration; latency only with 4 "steps".
    assert cost_per_iteration > 0
    assert latency_per_iteration > 0
    assert BALANCED_POLICY.max_workers > 4  # sanity: fan-out is wider than the 4 sequential steps
