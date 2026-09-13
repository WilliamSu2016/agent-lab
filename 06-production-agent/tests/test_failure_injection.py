"""Failure-injection tests: end-to-end scenarios exercising retry + timeout
+ error classification + idempotency working together as one pipeline, not
just each ``src/reliability`` module in isolation.
"""

from __future__ import annotations

import time

import pytest

from src.reliability.errors import ErrorCategory, classify_generic_exception
from src.reliability.idempotency import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
    InMemoryIdempotencyStore,
    make_create_order_tool,
    make_publish_report_tool,
    make_send_email_tool,
)
from src.reliability.retry import RetryPolicy, ToolInvocationError, retry_call
from src.reliability.timeout import run_with_timeout


class _InjectedFailureTool:
    """A tool double whose external call intermittently times out / fails
    on the first N invocations, then succeeds -- simulating a flaky
    external dependency without any real network/sleep."""

    def __init__(self, failures_before_success: int, *, slow_seconds: float = 0.0):
        self._failures_remaining = failures_before_success
        self._slow_seconds = slow_seconds
        self.call_count = 0

    def __call__(self) -> str:
        self.call_count += 1
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            if self._slow_seconds:
                time.sleep(self._slow_seconds)
            raise TimeoutError("simulated transient timeout")
        return "external-call-succeeded"


class TestRetryPlusTimeoutPipeline:
    """Combines reliability.timeout (mandatory per-call deadline) with
    reliability.retry (backoff across attempts) exactly as
    src/specialists/llm.py wires them for the real LLM call."""

    def test_transient_timeouts_recover_within_budget(self):
        tool = _InjectedFailureTool(failures_before_success=2)

        def bounded_call() -> str:
            return run_with_timeout(tool, 1.0, tool_name="flaky_external_tool")

        result = retry_call(
            bounded_call,
            policy=RetryPolicy(max_retries=3, backoff_base_seconds=0.01, jitter_ratio=0.0),
            tool_name="flaky_external_tool",
            classify=classify_generic_exception,
            sleep=lambda _seconds: None,
        )
        assert result == "external-call-succeeded"
        assert tool.call_count == 3

    def test_persistent_timeout_exhausts_retries_and_raises_structured_error(self):
        """A dependency that always exceeds its per-call timeout must
        eventually fail with a structured ToolInvocationError -- never hang
        forever, never crash with a raw ToolTimeoutError."""
        tool = _InjectedFailureTool(failures_before_success=999, slow_seconds=0.2)

        def bounded_call() -> str:
            return run_with_timeout(tool, 0.02, tool_name="always_slow_tool")

        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(
                bounded_call,
                policy=RetryPolicy(max_retries=2, backoff_base_seconds=0.01, jitter_ratio=0.0),
                tool_name="always_slow_tool",
                classify=classify_generic_exception,
                sleep=lambda _seconds: None,
            )
        assert exc_info.value.tool_error.category == ErrorCategory.TIMEOUT
        assert exc_info.value.tool_error.retryable is True
        assert exc_info.value.tool_error.attempt == 3  # 1 initial + 2 retries
        assert tool.call_count == 3

    def test_non_retryable_failure_terminates_without_ever_retrying(self):
        """A business validation failure (e.g. invalid tool arguments) must
        terminate on the very first attempt regardless of how generous the
        retry budget is -- retrying it can never help."""
        from src.reliability.errors import BusinessValidationError

        attempts = []

        def always_invalid() -> str:
            attempts.append(1)
            raise BusinessValidationError("requested quantity exceeds remaining stock")

        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(
                always_invalid,
                policy=RetryPolicy(max_retries=10),
                tool_name="order_tool",
                sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep/retry")),
            )
        assert len(attempts) == 1
        assert exc_info.value.tool_error.category == ErrorCategory.BUSINESS_VALIDATION


class TestIdempotencyUnderRetryAndDuplication:
    """The scenario idempotency exists for: a retry (or a duplicate agent
    call, or a crash-recovery replay) of a side-effecting tool must never
    re-execute the real side effect."""

    def test_send_email_retry_after_lost_response_does_not_resend(self):
        store = InMemoryIdempotencyStore()
        send_email = make_send_email_tool(store)

        first = send_email(to="user@example.com", subject="Welcome", body="Hi!", idempotency_key="welcome-order-123")
        # Simulate a retry after the success response was lost in transit --
        # same key, same content.
        second = send_email(to="user@example.com", subject="Welcome", body="Hi!", idempotency_key="welcome-order-123")
        assert first == second
        assert first.message_id == second.message_id  # exactly one email was ever sent

    def test_create_order_retry_returns_same_order_never_double_charges(self):
        store = InMemoryIdempotencyStore()
        create_order = make_create_order_tool(store)

        first = create_order(customer_id="cust-1", amount_cents=5000, idempotency_key="checkout-session-abc")
        second = create_order(customer_id="cust-1", amount_cents=5000, idempotency_key="checkout-session-abc")
        assert first.order_id == second.order_id

    def test_create_order_conflicting_reuse_of_key_is_rejected(self):
        store = InMemoryIdempotencyStore()
        create_order = make_create_order_tool(store)

        create_order(customer_id="cust-1", amount_cents=5000, idempotency_key="checkout-session-abc")
        with pytest.raises(IdempotencyConflictError):
            # Same key, different amount -- must not silently return the
            # first (unrelated) order, and must not create a second order.
            create_order(customer_id="cust-1", amount_cents=9999, idempotency_key="checkout-session-abc")

    def test_publish_report_same_version_is_a_no_op_duplicate(self):
        store = InMemoryIdempotencyStore()
        publish_report = make_publish_report_tool(store)

        first = publish_report(report_id="report-42", version=1, content="final content")
        second = publish_report(report_id="report-42", version=1, content="final content")
        assert first == second  # no second publish/notification event

    def test_publish_report_new_version_is_a_new_side_effect(self):
        store = InMemoryIdempotencyStore()
        publish_report = make_publish_report_tool(store)

        v1 = publish_report(report_id="report-42", version=1, content="draft")
        v2 = publish_report(report_id="report-42", version=2, content="final")
        assert v1.version == 1
        assert v2.version == 2

    def test_concurrent_duplicate_call_while_in_progress_is_rejected(self):
        """A second caller must never re-run the side effect while the
        first is still executing -- it must be told to back off instead."""
        store = InMemoryIdempotencyStore()
        store.put_in_progress("in-flight-key", fingerprint="user@example.com|Hi|Body")
        send_email = make_send_email_tool(store)

        with pytest.raises(IdempotencyInProgressError):
            send_email(to="user@example.com", subject="Hi", body="Body", idempotency_key="in-flight-key")

    def test_failed_side_effect_releases_the_key_so_a_later_retry_can_proceed(self):
        store = InMemoryIdempotencyStore()
        call_count = {"n": 0}

        def maybe_fail(*, idempotency_key: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("simulated network failure mid-side-effect")
            return "succeeded-on-retry"

        from src.reliability.idempotency import idempotent

        wrapped = idempotent(maybe_fail, store=store, key_fn=lambda **kw: kw["idempotency_key"])

        with pytest.raises(ConnectionError):
            wrapped(idempotency_key="retryable-key")
        # The failed attempt must not permanently "poison" the key --
        # a subsequent legitimate retry is allowed to actually run.
        result = wrapped(idempotency_key="retryable-key")
        assert result == "succeeded-on-retry"
        assert call_count["n"] == 2


class TestAgentLoopControlledFailureUnderRepeatedRejection:
    """Simulates a Planner<->Reviewer-style loop where the Reviewer never
    approves -- the loop must stop at max_iterations with a structured
    ControlledFailure, never spin forever and never crash uncontrolled."""

    def test_reviewer_never_approves_within_budget_raises_controlled_failure(self):
        from src.reliability.errors import AgentLoopGuard, ControlledFailureError

        guard = AgentLoopGuard(max_iterations=3)
        reviewer_feedback = ""
        with pytest.raises(ControlledFailureError) as exc_info:
            while True:
                guard.enter_iteration()
                reviewer_feedback = "still missing citations"
                # Reviewer never approves in this simulation.

        failure = exc_info.value.controlled_failure
        assert failure.iterations_used == 4  # 3 succeed, the 4th trips the guard
        assert failure.max_iterations == 3
        # The guard itself does not carry feedback automatically in the
        # raised path (that is the caller's job via as_controlled_failure);
        # this asserts the loop body did run every allowed iteration.
        assert reviewer_feedback == "still missing citations"

    def test_reviewer_approves_before_budget_exhausted_completes_normally(self):
        from src.reliability.errors import AgentLoopGuard

        guard = AgentLoopGuard(max_iterations=5)
        approved = False
        for _ in range(5):
            guard.enter_iteration()
            approved = True
            break
        assert approved is True
        assert guard.iterations_used == 1
