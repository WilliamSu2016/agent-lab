"""Failure #3 Tool timeout, #4 Tool 5xx, #5 Tool invalid response.

This project's only *real* external tool call is the OpenAI-compatible LLM
endpoint (covered end-to-end in ``test_llm_failure.py`` via the real
``build_openai_text_llm_call`` seam). There is no second, independent HTTP
tool in this repository to inject failures into without inventing new
business functionality -- which this experiment explicitly forbids.

Instead, these tests exercise the exact same reusable pipeline
(``src.reliability.retry.retry_call`` + ``src.reliability.timeout.run_with_timeout``)
against a *generic* tool call built the same way any future external
tool/HTTP/business call in this codebase is expected to be wired (see both
modules' docstrings: "usable by *any* Agent Tool call"). This demonstrates
the pipeline's documented extensibility with a tool-shaped call rather than
re-testing ``tests/test_retry.py``'s/``tests/test_timeout.py``'s unit-level
guarantees a second time.

Expected behaviour (per the experiment's own example):

    Tool timeout / Tool 5xx
        retry -> backoff -> retry limit -> structured failure
        (``ToolInvocationError``), never a bare hang/crash.

    Tool invalid response
        caller-side validation classifies a syntactically-successful but
        semantically-unusable response as non-retryable
        (``BusinessValidationError``) -> immediate structured failure, no
        wasted retries against a call that will never self-correct.
"""

from __future__ import annotations

import time

import pytest

from src.reliability.errors import BusinessValidationError, classify_exception
from src.reliability.retry import RetryPolicy, ToolInvocationError, retry_call
from src.reliability.timeout import ToolTimeoutError, run_with_timeout


def _fast_policy(max_retries: int = 2) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries, backoff_base_seconds=0.001, backoff_max_seconds=0.005)


class TestToolTimeout:
    """Failure #3: a hypothetical external tool (HTTP/MCP/business call)
    never responds within its budget."""

    def test_persistent_hang_raises_tool_timeout_error(self):
        def hangs_forever() -> str:
            time.sleep(5.0)
            return "too late"

        with pytest.raises(ToolTimeoutError):
            run_with_timeout(hangs_forever, 0.05, tool_name="fake_external_tool")

    def test_timeout_composed_with_retry_exhausts_and_raises_structured_failure(self):
        call_count = {"n": 0}

        def always_hangs() -> str:
            call_count["n"] += 1
            time.sleep(5.0)
            return "unreachable"

        def bounded_call() -> str:
            return run_with_timeout(always_hangs, 0.02, tool_name="fake_external_tool")

        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(bounded_call, policy=_fast_policy(max_retries=2), tool_name="fake_external_tool")

        assert exc_info.value.tool_error.category.value == "timeout"
        assert call_count["n"] == 3  # 1 initial attempt + 2 retries

    def test_recovers_once_a_later_attempt_answers_in_time(self):
        call_count = {"n": 0}

        def flaky() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                time.sleep(5.0)
            return "ok"

        def bounded_call() -> str:
            return run_with_timeout(flaky, 0.02, tool_name="fake_external_tool")

        result = retry_call(bounded_call, policy=_fast_policy(max_retries=2), tool_name="fake_external_tool")
        assert result == "ok"
        assert call_count["n"] == 2


class TestTool5xx:
    """Failure #4: the external tool responds with a transient server
    error (modeled generically -- classification does not require an
    ``openai`` exception type; any exception is classified via
    ``classify_exception``, which falls back to
    ``classify_generic_exception`` for plain Python exceptions)."""

    class _ServiceUnavailable(ConnectionError):
        """Stands in for a transient 5xx-style failure from a generic HTTP
        tool client; ``ConnectionError`` classifies as ``NETWORK``
        (retryable) via ``classify_generic_exception``."""

    def test_persistent_failure_exhausts_retries(self):
        call_count = {"n": 0}

        def always_fails() -> str:
            call_count["n"] += 1
            raise self._ServiceUnavailable("503 service unavailable")

        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(always_fails, policy=_fast_policy(max_retries=2), tool_name="fake_external_tool")

        assert exc_info.value.tool_error.category.value == "network"
        assert call_count["n"] == 3

    def test_transient_failure_recovers_within_budget(self):
        call_count = {"n": 0}

        def flaky() -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise self._ServiceUnavailable("503 service unavailable")
            return "recovered"

        result = retry_call(flaky, policy=_fast_policy(max_retries=2), tool_name="fake_external_tool")
        assert result == "recovered"
        assert call_count["n"] == 2


class TestToolInvalidResponse:
    """Failure #5: the tool call *succeeds* transport-wise but returns a
    response the caller cannot use (malformed/incomplete payload) --
    distinct from a transport-level timeout/5xx. The caller must validate
    the response and classify a bad shape as non-retryable so it fails
    immediately instead of retrying a call that will never produce a
    different (valid) result."""

    def test_malformed_response_is_non_retryable_and_fails_immediately(self):
        call_count = {"n": 0}

        def returns_malformed_payload() -> dict:
            call_count["n"] += 1
            return {"unexpected": "shape"}  # missing the "result" key callers require

        def validated_call() -> dict:
            payload = returns_malformed_payload()
            if "result" not in payload:
                raise BusinessValidationError(f"Tool response missing required 'result' key: {payload!r}")
            return payload

        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(validated_call, policy=_fast_policy(max_retries=3), tool_name="fake_external_tool")

        assert exc_info.value.tool_error.category.value == "business_validation"
        assert exc_info.value.tool_error.retryable is False
        assert call_count["n"] == 1  # no retries wasted on an unrecoverable shape

    def test_business_validation_error_classifies_correctly_standalone(self):
        assert classify_exception(BusinessValidationError("bad shape")).value == "business_validation"

    def test_well_formed_response_passes_through_unmodified(self):
        def returns_valid_payload() -> dict:
            return {"result": "ok"}

        def validated_call() -> dict:
            payload = returns_valid_payload()
            if "result" not in payload:
                raise BusinessValidationError("missing result")
            return payload

        result = retry_call(validated_call, policy=_fast_policy(), tool_name="fake_external_tool")
        assert result == {"result": "ok"}
