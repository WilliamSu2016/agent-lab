"""Failure #1 LLM timeout, #2 LLM 5xx, #7 Rate limit, #12 Invalid structured
output.

Every LLM-failure test here monkeypatches the *real* seam
(``openai.resources.chat.completions.completions.Completions.create``) so
the actual production call path -- ``src.agents.llm.build_openai_text_llm_call``
-- is exercised end-to-end, not a reimplementation of ``retry_call``/
``run_with_timeout`` in isolation (those have their own unit tests).

Expected behaviour:

    LLM timeout / LLM 5xx / Rate limit
        classify -> retry -> backoff -> retry limit -> structured failure
        (``ToolInvocationError``), never a bare/raw exception, never a
        silent hang.

    Invalid structured output (Planner)
        tolerate minor formatting deviations (markdown fences, numbered
        lists) -> otherwise a documented, currently-uncaught ``ValueError``
        that crashes the run (a known limitation -- see
        ``docs/09-FAILURE-MATRIX.md``).

    Researcher node
        never raises regardless of what its LLM call does -- every
        failure becomes a structured ``WorkerResult`` (status
        "failed"/"timeout"/"blocked"), so one researcher's total failure
        never crashes the rest of the fan-out.
"""

from __future__ import annotations

import time

import openai
import pytest
from openai.resources.chat.completions.completions import Completions

from src.agents.llm import build_openai_text_llm_call
from src.agents.researcher import make_researcher_node
from src.agents.supervisor import _parse_aspects, make_planner_node
from src.reliability.retry import ToolInvocationError
from src.security.authorization import ApprovalStore
from src.security.guardrails import AgentWorkflowGuardrail, ToolGuardrail
from src.security.tool_policy import build_default_registry
from src.observability.tracing import ExecutionContext, SpanKind, Tracer, bind_execution_context
from tests.failure_injection.conftest import openai_status_error


def _fake_message(content: str):
    class _Msg:
        pass

    msg = _Msg()
    msg.content = content
    return msg


def _fake_choice(content: str):
    class _Choice:
        pass

    choice = _Choice()
    choice.message = _fake_message(content)
    return choice


def _fake_response(content: str):
    class _Response:
        pass

    resp = _Response()
    resp.choices = [_fake_choice(content)]
    return resp


class TestLLMTimeout:
    """Failure #1: the model hangs past its configured timeout."""

    def test_persistent_hang_exhausts_retries_and_raises_structured_failure(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=1, worker_timeout_seconds=0.05)
        call_count = {"n": 0}

        def hanging_create(*args, **kwargs):
            call_count["n"] += 1
            time.sleep(1.0)  # always exceeds the 0.05s timeout budget
            return _fake_response("too late")

        monkeypatch.setattr(Completions, "create", hanging_create)
        text_llm_call = build_openai_text_llm_call(settings)

        with pytest.raises(ToolInvocationError) as exc_info:
            text_llm_call("system", "user")

        assert exc_info.value.tool_error.category.value == "timeout"
        assert call_count["n"] == 2  # initial attempt + 1 retry

    def test_recovers_if_a_later_attempt_answers_within_the_timeout(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2, worker_timeout_seconds=0.05)
        call_count = {"n": 0}

        def flaky_create(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                time.sleep(1.0)
            return _fake_response("recovered")

        monkeypatch.setattr(Completions, "create", flaky_create)
        text_llm_call = build_openai_text_llm_call(settings)

        result = text_llm_call("system", "user")
        assert result == "recovered"
        assert call_count["n"] == 2


class TestLLM5xx:
    """Failure #2: the provider returns a transient server error."""

    def test_persistent_500_exhausts_retries_and_raises_structured_failure(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2)
        call_count = {"n": 0}

        def failing_create(*args, **kwargs):
            call_count["n"] += 1
            raise openai_status_error(500, "internal server error")

        monkeypatch.setattr(Completions, "create", failing_create)
        text_llm_call = build_openai_text_llm_call(settings)

        with pytest.raises(ToolInvocationError) as exc_info:
            text_llm_call("system", "user")

        assert exc_info.value.tool_error.category.value == "server_error"
        assert exc_info.value.tool_error.retryable is True
        assert call_count["n"] == 3  # 1 initial + 2 retries

    def test_recovers_after_one_transient_500(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2)
        call_count = {"n": 0}

        def flaky_create(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise openai_status_error(503, "temporarily unavailable")
            return _fake_response("ok after retry")

        monkeypatch.setattr(Completions, "create", flaky_create)
        text_llm_call = build_openai_text_llm_call(settings)

        assert text_llm_call("system", "user") == "ok after retry"
        assert call_count["n"] == 2

    def test_non_retryable_4xx_fails_on_first_attempt_no_retry(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=3)
        call_count = {"n": 0}

        def bad_request(*args, **kwargs):
            call_count["n"] += 1
            raise openai_status_error(400, "invalid request")

        monkeypatch.setattr(Completions, "create", bad_request)
        text_llm_call = build_openai_text_llm_call(settings)

        with pytest.raises(ToolInvocationError) as exc_info:
            text_llm_call("system", "user")

        assert exc_info.value.tool_error.category.value == "invalid_parameters"
        assert exc_info.value.tool_error.retryable is False
        assert call_count["n"] == 1  # no retry attempted


class TestRateLimit:
    """Failure #7: provider throttles with HTTP 429 / ``RateLimitError``."""

    def test_429_is_retried_then_recovers(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2)
        call_count = {"n": 0}

        def throttled_create(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                exc = openai_status_error(429, "rate limited")
                raise openai.RateLimitError("rate limited", response=exc.response, body=None)
            return _fake_response("finally allowed")

        monkeypatch.setattr(Completions, "create", throttled_create)
        text_llm_call = build_openai_text_llm_call(settings)

        assert text_llm_call("system", "user") == "finally allowed"
        assert call_count["n"] == 3

    def test_429_exhausting_retries_raises_structured_failure(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=1)

        def always_throttled(*args, **kwargs):
            exc = openai_status_error(429, "rate limited")
            raise openai.RateLimitError("rate limited", response=exc.response, body=None)

        monkeypatch.setattr(Completions, "create", always_throttled)
        text_llm_call = build_openai_text_llm_call(settings)

        with pytest.raises(ToolInvocationError) as exc_info:
            text_llm_call("system", "user")
        assert exc_info.value.tool_error.category.value == "rate_limit"


@pytest.fixture
def researcher_env():
    registry = build_default_registry()
    workflow_guardrail = AgentWorkflowGuardrail(registry, ApprovalStore())
    tool_guardrail = ToolGuardrail(registry)
    tracer = Tracer()
    context = ExecutionContext(
        request_id="req-1", trace_id="trace-1", user_id="u1", session_id="s1", agent_version="test", environment="test"
    )
    return registry, workflow_guardrail, tool_guardrail, tracer, context


class TestLLMFailureFeedsResearcherStructuredFallback:
    """Ties an exhausted/hanging LLM failure through to
    ``src.agents.researcher.make_researcher_node``'s outer fallback -- the
    "fallback -> structured failure" step that lets the rest of the
    Production Research graph continue despite one researcher's total
    failure."""

    def test_researcher_reports_failed_status_instead_of_raising(self, monkeypatch, make_settings, researcher_env):
        registry, workflow_guardrail, tool_guardrail, tracer, context = researcher_env
        settings = make_settings(max_retries=1, worker_timeout_seconds=1.0)

        def always_500(*args, **kwargs):
            raise openai_status_error(500, "down")

        monkeypatch.setattr(Completions, "create", always_500)
        text_llm_call = build_openai_text_llm_call(settings)
        node = make_researcher_node(
            text_llm_call,
            search_tool_fn=lambda query: "irrelevant search result",
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
            tracer=tracer,
        )

        with bind_execution_context(context):
            with tracer.span(SpanKind.WORKFLOW, "root", context=context):
                update = node(
                    {
                        "question": "q",
                        "task": {"task_id": "t1", "aspect": "aspect one", "reason": "test"},
                        "per_worker_timeout_seconds": 5.0,
                        "identity": {"user_id": "u1", "tenant_id": "t1", "roles": []},
                        "context": context.to_dict(),
                        "mode": "balanced",
                    }
                )

        result = update["worker_results"][0]
        assert result["status"] == "failed"
        assert result["task_id"] == "t1"
        assert result["error"]

    def test_researcher_reports_timeout_status_when_llm_call_itself_never_returns(self, researcher_env):
        registry, workflow_guardrail, tool_guardrail, tracer, context = researcher_env

        def never_returns(instructions: str, user_prompt: str) -> str:
            time.sleep(5.0)
            return "unreachable"

        node = make_researcher_node(
            never_returns,
            search_tool_fn=lambda query: "irrelevant search result",
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
            tracer=tracer,
        )


        with bind_execution_context(context):
            with tracer.span(SpanKind.WORKFLOW, "root", context=context):
                update = node(
                    {
                        "question": "q",
                        "task": {"task_id": "t1", "aspect": "aspect one", "reason": "test"},
                        "per_worker_timeout_seconds": 0.05,
                        "identity": {"user_id": "u1", "tenant_id": "t1", "roles": []},
                        "context": context.to_dict(),
                        "mode": "balanced",
                    }
                )
        result = update["worker_results"][0]
        assert result["status"] == "timeout"


class TestInvalidStructuredOutput:
    """Failure #12: the Planner's LLM reply is not usable JSON.

    ``_parse_aspects`` tolerates common near-misses (markdown fences,
    numbered lists) but total garbage yields an empty list, and
    ``make_planner_node`` then raises a bare, currently *uncaught*
    ``ValueError`` -- there is no fallback/retry for this case. This is a
    genuine, pre-existing gap, documented (not silently patched) in
    ``docs/09-FAILURE-MATRIX.md``.
    """

    def test_markdown_fenced_json_is_tolerated(self):
        raw = '```json\n["aspect one", "aspect two"]\n```'
        assert _parse_aspects(raw) == ["aspect one", "aspect two"]

    def test_numbered_list_fallback_is_tolerated(self):
        raw = "1. aspect one\n2. aspect two\n"
        assert _parse_aspects(raw) == ["aspect one", "aspect two"]

    def test_total_garbage_yields_empty_list(self):
        assert _parse_aspects("   \n\n   ") == []

    def test_planner_node_raises_uncaught_value_error_on_empty_aspects(self):
        def garbage_llm_call(system_prompt: str, user_prompt: str) -> str:
            return "   "

        tracer = Tracer()
        context = ExecutionContext(
            request_id="req-1", trace_id="trace-1", user_id="u1", session_id="s1", agent_version="test", environment="test"
        )
        planner_node = make_planner_node(garbage_llm_call, tracer=tracer)
        state = {
            "question": "q",
            "max_workers": 6,
            "iteration": 0,
            "review": None,
        }

        with bind_execution_context(context):
            with tracer.span(SpanKind.WORKFLOW, "root", context=context):
                with pytest.raises(ValueError, match="Planner produced no research aspects"):
                    planner_node(state)
