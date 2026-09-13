"""Tests for src/reliability/errors.py: classification, ToolError, and the
AgentLoopGuard / ControlledFailure "max_iterations -> controlled failure"
primitives.
"""

from __future__ import annotations

import httpx2
import openai
import pytest

from src.reliability.errors import (
    NON_RETRYABLE_CATEGORIES,
    RETRYABLE_CATEGORIES,
    AgentLoopGuard,
    BusinessValidationError,
    ControlledFailure,
    ControlledFailureError,
    ErrorCategory,
    ToolError,
    classify_exception,
    classify_generic_exception,
    classify_openai_exception,
    is_retryable_category,
    make_tool_error,
)


def _status_error(status: int, message: str = "boom") -> openai.APIStatusError:
    req = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    resp = httpx2.Response(status, request=req, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=resp, body=None)


def _req() -> httpx2.Request:
    return httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


class TestCategoryPartition:
    def test_retryable_and_non_retryable_are_disjoint_and_exhaustive(self):
        assert RETRYABLE_CATEGORIES.isdisjoint(NON_RETRYABLE_CATEGORIES)
        assert RETRYABLE_CATEGORIES | NON_RETRYABLE_CATEGORIES == set(ErrorCategory)

    @pytest.mark.parametrize("category", list(RETRYABLE_CATEGORIES))
    def test_retryable_categories_report_true(self, category):
        assert is_retryable_category(category) is True

    @pytest.mark.parametrize("category", list(NON_RETRYABLE_CATEGORIES))
    def test_non_retryable_categories_report_false(self, category):
        assert is_retryable_category(category) is False


class TestClassifyGenericException:
    def test_timeout_error_is_timeout(self):
        assert classify_generic_exception(TimeoutError("slow")) == ErrorCategory.TIMEOUT

    def test_permission_error_is_permission_denied(self):
        assert classify_generic_exception(PermissionError("no")) == ErrorCategory.PERMISSION_DENIED

    def test_connection_error_is_network(self):
        assert classify_generic_exception(ConnectionError("reset")) == ErrorCategory.NETWORK

    def test_os_error_is_network(self):
        assert classify_generic_exception(OSError("dns failure")) == ErrorCategory.NETWORK

    @pytest.mark.parametrize("exc", [ValueError("bad"), TypeError("bad"), KeyError("bad"), LookupError("bad")])
    def test_value_type_key_lookup_are_invalid_parameters(self, exc):
        assert classify_generic_exception(exc) == ErrorCategory.INVALID_PARAMETERS

    def test_unrecognized_exception_is_unknown(self):
        assert classify_generic_exception(RuntimeError("mystery")) == ErrorCategory.UNKNOWN


class TestClassifyOpenAIException:
    def test_timeout_error(self):
        assert classify_openai_exception(openai.APITimeoutError(request=_req())) == ErrorCategory.TIMEOUT

    def test_connection_error_is_network(self):
        assert classify_openai_exception(openai.APIConnectionError(request=_req())) == ErrorCategory.NETWORK

    def test_rate_limit_error(self):
        exc = openai.RateLimitError("too many requests", response=_status_error(429).response, body=None)
        assert classify_openai_exception(exc) == ErrorCategory.RATE_LIMIT

    def test_authentication_error(self):
        exc = openai.AuthenticationError("bad key", response=_status_error(401).response, body=None)
        assert classify_openai_exception(exc) == ErrorCategory.AUTHENTICATION

    def test_permission_denied_error(self):
        exc = openai.PermissionDeniedError("forbidden", response=_status_error(403).response, body=None)
        assert classify_openai_exception(exc) == ErrorCategory.PERMISSION_DENIED

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: openai.BadRequestError("bad", response=_status_error(400).response, body=None),
            lambda: openai.NotFoundError("missing", response=_status_error(404).response, body=None),
            lambda: openai.UnprocessableEntityError("invalid", response=_status_error(422).response, body=None),
        ],
    )
    def test_invalid_parameter_style_errors(self, exc_factory):
        assert classify_openai_exception(exc_factory()) == ErrorCategory.INVALID_PARAMETERS

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_status_errors_are_server_error(self, status):
        assert classify_openai_exception(_status_error(status)) == ErrorCategory.SERVER_ERROR

    def test_429_status_error_is_rate_limit(self):
        assert classify_openai_exception(_status_error(429)) == ErrorCategory.RATE_LIMIT

    def test_409_status_error_is_invalid_parameters(self):
        assert classify_openai_exception(_status_error(409)) == ErrorCategory.INVALID_PARAMETERS

    def test_unmapped_status_is_unknown(self):
        assert classify_openai_exception(_status_error(418)) == ErrorCategory.UNKNOWN

    def test_non_openai_exception_falls_back_to_generic(self):
        assert classify_openai_exception(TimeoutError("slow")) == ErrorCategory.TIMEOUT


class TestClassifyException:
    def test_business_validation_error_takes_priority(self):
        assert classify_exception(BusinessValidationError("stock exceeded")) == ErrorCategory.BUSINESS_VALIDATION

    def test_openai_exception_classified_before_generic_fallback(self):
        assert classify_exception(openai.APITimeoutError(request=_req())) == ErrorCategory.TIMEOUT

    def test_plain_python_exception_uses_generic_classifier(self):
        assert classify_exception(ValueError("bad shape")) == ErrorCategory.INVALID_PARAMETERS


class TestMakeToolError:
    def test_retryable_flag_matches_category(self):
        tool_error = make_tool_error(TimeoutError("slow"), tool_name="my_tool", attempt=1)
        assert tool_error.category == ErrorCategory.TIMEOUT
        assert tool_error.retryable is True
        assert tool_error.tool_name == "my_tool"
        assert tool_error.attempt == 1
        assert tool_error.exception_type == "TimeoutError"

    def test_non_retryable_flag_for_business_validation(self):
        tool_error = make_tool_error(BusinessValidationError("bad"), tool_name="t", attempt=2)
        assert tool_error.retryable is False
        assert tool_error.category == ErrorCategory.BUSINESS_VALIDATION

    def test_message_falls_back_to_exception_class_name_when_empty(self):
        class _Empty(Exception):
            def __str__(self):
                return ""

        tool_error = make_tool_error(_Empty(), tool_name="t", attempt=1)
        assert tool_error.message == "_Empty"

    def test_to_dict_and_to_trace_line(self):
        tool_error = make_tool_error(TimeoutError("slow"), tool_name="t", attempt=1)
        as_dict = tool_error.to_dict()
        assert as_dict["category"] == "timeout"
        assert as_dict["retryable"] is True
        assert "original_exception" not in as_dict
        trace = tool_error.to_trace_line()
        assert "t" in trace and "timeout" in trace

    def test_original_exception_excluded_from_equality(self):
        a = make_tool_error(TimeoutError("one"), tool_name="t", attempt=1)
        b = ToolError(
            category=a.category,
            retryable=a.retryable,
            message=a.message,
            tool_name=a.tool_name,
            exception_type=a.exception_type,
            attempt=a.attempt,
            occurred_at=a.occurred_at,
            original_exception=RuntimeError("different"),
        )
        assert a == b


class TestAgentLoopGuard:
    def test_rejects_non_positive_max_iterations(self):
        with pytest.raises(ValueError):
            AgentLoopGuard(max_iterations=0)

    def test_allows_iterations_within_budget(self):
        guard = AgentLoopGuard(max_iterations=3)
        assert guard.enter_iteration() == 1
        assert guard.enter_iteration() == 2
        assert guard.enter_iteration() == 3

    def test_raises_controlled_failure_error_past_budget(self):
        guard = AgentLoopGuard(max_iterations=2)
        guard.enter_iteration()
        guard.enter_iteration()
        with pytest.raises(ControlledFailureError) as exc_info:
            guard.enter_iteration()
        failure = exc_info.value.controlled_failure
        assert isinstance(failure, ControlledFailure)
        assert failure.max_iterations == 2
        assert failure.iterations_used == 3
        assert "exceeded max_iterations" in failure.reason

    def test_budget_exhausted_reports_correctly(self):
        guard = AgentLoopGuard(max_iterations=1)
        assert guard.budget_exhausted() is False
        guard.enter_iteration()
        assert guard.budget_exhausted() is True

    def test_as_controlled_failure_carries_last_feedback(self):
        guard = AgentLoopGuard(max_iterations=1)
        guard.enter_iteration()
        failure = guard.as_controlled_failure(reason="not approved", last_feedback="missing citations")
        assert failure.last_feedback == "missing citations"
        assert failure.failure_id  # non-empty uuid hex
        assert "not approved" in failure.to_trace_line()
        as_dict = failure.to_dict()
        assert as_dict["reason"] == "not approved"
        assert as_dict["last_feedback"] == "missing citations"
