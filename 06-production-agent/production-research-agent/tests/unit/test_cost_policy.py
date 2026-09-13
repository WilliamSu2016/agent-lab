"""Unit tests: ``src.cost.policy`` -- Fast/Balanced/Quality policies and
the controlled-degradation ladder."""

from __future__ import annotations

import pytest

from src.cost.budget import BudgetExceededError
from src.cost.policy import (
    BALANCED_POLICY,
    DEFAULT_MODE,
    FAST_POLICY,
    QUALITY_POLICY,
    ExecutionMode,
    get_policy,
    next_degraded_mode,
    run_with_degradation,
)


def test_balanced_is_the_documented_default_mode():
    assert DEFAULT_MODE is ExecutionMode.BALANCED
    assert get_policy(ExecutionMode.BALANCED) is BALANCED_POLICY


def test_fast_uses_fewer_workers_less_context_and_a_cheaper_model_than_quality():
    assert FAST_POLICY.max_workers < QUALITY_POLICY.max_workers
    assert FAST_POLICY.max_context_chars < QUALITY_POLICY.max_context_chars
    assert FAST_POLICY.model_name_override is not None
    assert QUALITY_POLICY.model_name_override is None


def test_quality_allows_more_iterations_and_a_bigger_budget_than_balanced():
    assert QUALITY_POLICY.max_agent_iterations > BALANCED_POLICY.max_agent_iterations
    assert QUALITY_POLICY.budget.max_tokens > BALANCED_POLICY.budget.max_tokens
    assert QUALITY_POLICY.budget.max_cost_usd > BALANCED_POLICY.budget.max_cost_usd
    assert QUALITY_POLICY.budget.timeout_seconds > BALANCED_POLICY.budget.timeout_seconds


def test_degradation_ladder_order():
    assert next_degraded_mode(ExecutionMode.QUALITY) is ExecutionMode.BALANCED
    assert next_degraded_mode(ExecutionMode.BALANCED) is ExecutionMode.FAST
    assert next_degraded_mode(ExecutionMode.FAST) is None


def test_run_with_degradation_returns_the_first_successful_mode_untouched():
    outcome = run_with_degradation(lambda policy: f"ok:{policy.mode.value}")
    assert outcome.succeeded
    assert outcome.final_mode is ExecutionMode.QUALITY
    assert outcome.result == "ok:quality"
    assert len(outcome.attempts) == 1


def test_run_with_degradation_falls_back_quality_to_balanced_to_fast():
    calls: list[ExecutionMode] = []

    def execute(policy):
        calls.append(policy.mode)
        if policy.mode is not ExecutionMode.FAST:
            raise BudgetExceededError("tokens", 999_999, 40_000)
        return "fast result"

    outcome = run_with_degradation(execute)
    assert outcome.succeeded
    assert outcome.final_mode is ExecutionMode.FAST
    assert outcome.result == "fast result"
    assert calls == [ExecutionMode.QUALITY, ExecutionMode.BALANCED, ExecutionMode.FAST]
    assert [a.mode for a in outcome.attempts] == calls


def test_run_with_degradation_returns_a_graceful_failure_when_even_fast_fails():
    def always_fails(policy):
        raise TimeoutError("simulated timeout")

    outcome = run_with_degradation(always_fails)
    assert not outcome.succeeded
    assert outcome.result is None
    assert outcome.final_mode is None
    assert outcome.graceful_failure is not None
    assert "Quality -> Balanced -> Fast" in outcome.graceful_failure.reason
    assert len(outcome.graceful_failure.attempts) == 3


def test_on_degrade_hook_is_invoked_for_every_forced_downgrade():
    downgrades: list[tuple[ExecutionMode, ExecutionMode]] = []

    def execute(policy):
        if policy.mode is ExecutionMode.FAST:
            return "ok"
        raise BudgetExceededError("tokens", 999_999, 40_000)

    run_with_degradation(execute, on_degrade=lambda frm, to, reason: downgrades.append((frm, to)))
    assert downgrades == [
        (ExecutionMode.QUALITY, ExecutionMode.BALANCED),
        (ExecutionMode.BALANCED, ExecutionMode.FAST),
    ]
