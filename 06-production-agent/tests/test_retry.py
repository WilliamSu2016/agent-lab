"""Tests for src/reliability/retry.py: exponential backoff calculation,
maximum retry count enforcement, and retryable/non-retryable classification
driving the retry decision -- with an injected fake ``sleep`` so no test
waits through a real backoff delay.
"""

from __future__ import annotations

import pytest

from src.reliability.errors import BusinessValidationError, ErrorCategory, ToolError, classify_generic_exception
from src.reliability.retry import RetryPolicy, ToolInvocationError, compute_backoff_seconds, retry_call


class TestRetryPolicyValidation:
    def test_default_policy_is_valid(self):
        RetryPolicy()

    def test_negative_max_retries_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_retries=-1)

    def test_zero_max_retries_is_allowed(self):
        RetryPolicy(max_retries=0)

    def test_non_positive_backoff_base_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(backoff_base_seconds=0)

    def test_backoff_max_below_base_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(backoff_base_seconds=5.0, backoff_max_seconds=1.0)

    def test_negative_jitter_ratio_rejected(self):
        with pytest.raises(ValueError):
            RetryPolicy(jitter_ratio=-0.1)


class TestComputeBackoffSeconds:
    def test_exponential_growth_without_jitter(self):
        policy = RetryPolicy(backoff_base_seconds=1.0, backoff_max_seconds=100.0, jitter_ratio=0.0)
        assert compute_backoff_seconds(policy, attempt=1, rand=lambda: 0.0) == 1.0
        assert compute_backoff_seconds(policy, attempt=2, rand=lambda: 0.0) == 2.0
        assert compute_backoff_seconds(policy, attempt=3, rand=lambda: 0.0) == 4.0
        assert compute_backoff_seconds(policy, attempt=4, rand=lambda: 0.0) == 8.0

    def test_capped_at_backoff_max_seconds(self):
        policy = RetryPolicy(backoff_base_seconds=1.0, backoff_max_seconds=3.0, jitter_ratio=0.0)
        assert compute_backoff_seconds(policy, attempt=10, rand=lambda: 0.0) == 3.0

    def test_jitter_adds_up_to_jitter_ratio_of_capped_delay(self):
        policy = RetryPolicy(backoff_base_seconds=2.0, backoff_max_seconds=100.0, jitter_ratio=0.5)
        # attempt=1 -> capped_delay=2.0; jitter in [0, 1.0]
        assert compute_backoff_seconds(policy, attempt=1, rand=lambda: 0.0) == 2.0
        assert compute_backoff_seconds(policy, attempt=1, rand=lambda: 1.0) == 3.0

    def test_zero_jitter_ratio_is_deterministic_regardless_of_rand(self):
        policy = RetryPolicy(backoff_base_seconds=1.0, backoff_max_seconds=100.0, jitter_ratio=0.0)
        assert compute_backoff_seconds(policy, attempt=2, rand=lambda: 0.999) == 2.0

    def test_attempt_below_one_rejected(self):
        with pytest.raises(ValueError):
            compute_backoff_seconds(RetryPolicy(), attempt=0)


class _FlakySequence:
    """Calls a function that raises the next queued exception each call,
    then finally succeeds, or raises forever if the queue is exhausted."""

    def __init__(self, exceptions, success_value="ok"):
        self._exceptions = list(exceptions)
        self._success_value = success_value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return self._success_value


class TestRetryCall:
    def test_succeeds_immediately_without_retry(self):
        sleeps = []
        result = retry_call(
            lambda: "done",
            policy=RetryPolicy(max_retries=3),
            tool_name="t",
            sleep=sleeps.append,
        )
        assert result == "done"
        assert sleeps == []

    def test_retries_transient_failure_then_succeeds(self):
        flaky = _FlakySequence([TimeoutError("slow"), ConnectionError("reset")], success_value="recovered")
        sleeps = []
        result = retry_call(
            flaky,
            policy=RetryPolicy(max_retries=5, backoff_base_seconds=1.0, jitter_ratio=0.0),
            tool_name="flaky_tool",
            classify=classify_generic_exception,
            sleep=sleeps.append,
        )
        assert result == "recovered"
        assert flaky.calls == 3
        assert sleeps == [1.0, 2.0]

    def test_non_retryable_failure_raises_immediately_without_sleeping(self):
        sleeps = []
        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(
                lambda: (_ for _ in ()).throw(BusinessValidationError("nope")),
                policy=RetryPolicy(max_retries=5),
                tool_name="biz_tool",
                sleep=sleeps.append,
            )
        assert sleeps == []
        assert exc_info.value.tool_error.category == ErrorCategory.BUSINESS_VALIDATION
        assert exc_info.value.tool_error.retryable is False
        assert exc_info.value.__cause__ is not None

    def test_retries_exhausted_raises_tool_invocation_error(self):
        always_fails = _FlakySequence([TimeoutError("1"), TimeoutError("2"), TimeoutError("3")])
        sleeps = []
        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(
                always_fails,
                policy=RetryPolicy(max_retries=2, jitter_ratio=0.0),
                tool_name="t",
                classify=classify_generic_exception,
                sleep=sleeps.append,
            )
        # attempt 1 fails (retry), attempt 2 fails (retry), attempt 3 fails
        # (max_retries=2 exhausted) -> raise. 3 calls total, 2 sleeps.
        assert always_fails.calls == 3
        assert len(sleeps) == 2
        assert exc_info.value.tool_error.attempt == 3
        assert exc_info.value.tool_error.category == ErrorCategory.TIMEOUT

    def test_max_retries_zero_fails_on_first_attempt(self):
        flaky = _FlakySequence([TimeoutError("slow")], success_value="unused")
        with pytest.raises(ToolInvocationError):
            retry_call(
                flaky,
                policy=RetryPolicy(max_retries=0),
                tool_name="t",
                sleep=lambda _seconds: None,
            )
        assert flaky.calls == 1

    def test_on_error_hook_called_for_every_attempt(self):
        recorded: list[ToolError] = []
        flaky = _FlakySequence([TimeoutError("1"), TimeoutError("2")], success_value="done")
        result = retry_call(
            flaky,
            policy=RetryPolicy(max_retries=5, jitter_ratio=0.0),
            tool_name="t",
            classify=classify_generic_exception,
            sleep=lambda _seconds: None,
            on_error=recorded.append,
        )
        assert result == "done"
        assert len(recorded) == 2
        assert [e.attempt for e in recorded] == [1, 2]

    def test_never_lets_raw_exception_propagate(self):
        """Error handling requirement: tool failure must never surface as a
        raw, unclassified exception -- always ToolInvocationError wrapping a
        structured ToolError."""
        with pytest.raises(ToolInvocationError):
            retry_call(
                lambda: (_ for _ in ()).throw(ValueError("bad shape")),
                policy=RetryPolicy(max_retries=0),
                tool_name="t",
                sleep=lambda _seconds: None,
            )
