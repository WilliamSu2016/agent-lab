"""Failure #1 LLM timeout, #2 LLM 5xx, #7 Rate limit, #12 Invalid structured
output.

Every LLM-failure test here monkeypatches the *real* seam
(``openai.resources.chat.completions.completions.Completions.create``) so
the actual production call path --
``src.specialists.llm.build_openai_text_llm_call`` -- is exercised
end-to-end, not a reimplementation of ``retry_call``/``run_with_timeout`` in
isolation (that pipeline already has its own unit tests in
``tests/test_retry.py``/``tests/test_timeout.py``).

Expected behaviour (mirrors the user's own example):

    LLM timeout / LLM 5xx / Rate limit
        classify -> retry -> backoff -> retry limit -> structured failure
        (``ToolInvocationError``), never a bare/raw exception, never a
        silent hang.

    Invalid structured output (Planner)
        tolerate minor formatting deviations (markdown fences, numbered
        lists) -> otherwise a documented, currently-uncaught ``ValueError``
        that crashes the run (a real gap, not fixed here -- see
        ``docs/09-FAILURE-MATRIX.md``).
"""

from __future__ import annotations

import time

import openai
import pytest
from openai.resources.chat.completions.completions import Completions

from src.multi_agent_research.planner import _parse_aspects, make_planner_node
from src.multi_agent_research.research_worker import make_worker_node
from src.reliability.retry import ToolInvocationError
from src.specialists.llm import build_openai_text_llm_call
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
        # Initial attempt + 1 retry, exactly per policy.max_retries.
        assert call_count["n"] == 2

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
        """Not a "5xx" failure, but proves the classifier -- not just luck
        -- is what drives retry vs. immediate structured failure."""
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


class TestLLMFailureFeedsWorkerStructuredFallback:
    """Ties an exhausted LLM failure through to
    ``research_worker.make_worker_node``'s outer fallback -- the "fallback ->
    structured failure" step that lets the rest of the multi-agent graph
    continue despite one worker's total failure (see
    ``src/multi_agent_research/research_worker.py``)."""

    def test_worker_reports_failed_status_instead_of_raising(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=1, worker_timeout_seconds=1.0)

        def always_500(*args, **kwargs):
            raise openai_status_error(500, "down")

        monkeypatch.setattr(Completions, "create", always_500)
        text_llm_call = build_openai_text_llm_call(settings)
        worker_node = make_worker_node(text_llm_call)

        update = worker_node(
            {
                "question": "q",
                "task": {"task_id": "t1", "aspect": "aspect one", "reason": "test"},
                "per_worker_timeout_seconds": 5.0,
            }
        )

        result = update["worker_results"][0]
        assert result["status"] == "failed"
        assert result["task_id"] == "t1"
        assert "down" in result["error"] or "internal" in result["error"] or result["error"]

    def test_worker_reports_timeout_status_when_llm_call_itself_never_returns(self, make_settings):
        """The worker's own outer timeout fires (a raw ``TimeoutError`` ->
        ``status="timeout"``) when the inner call never returns at all --
        distinct from the "failed" case above where the inner pipeline
        itself already converted the failure into a structured
        ``ToolInvocationError``."""

        def never_returns(instructions: str, user_prompt: str) -> str:
            time.sleep(5.0)
            return "unreachable"

        worker_node = make_worker_node(never_returns)
        update = worker_node(
            {
                "question": "q",
                "task": {"task_id": "t1", "aspect": "aspect one", "reason": "test"},
                "per_worker_timeout_seconds": 0.05,
            }
        )
        result = update["worker_results"][0]
        assert result["status"] == "timeout"


class TestInvalidStructuredOutput:
    """Failure #12: the Planner's LLM reply is not usable JSON.

    ``_parse_aspects`` tolerates common near-misses (markdown fences,
    numbered lists) but total garbage yields an empty list, and
    ``make_planner_node`` then raises a bare, currently *uncaught*
    ``ValueError`` -- there is no fallback/retry for this case anywhere
    between the Planner and ``run_multi_agent_research``'s top level. This
    is a genuine, pre-existing gap; per this experiment's "no new business
    functionality" constraint, it is documented and pinned down here rather
    than silently patched. See ``docs/09-FAILURE-MATRIX.md``.
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

        planner_node = make_planner_node(garbage_llm_call)
        state = {
            "question": "q",
            "max_workers": 6,
            "iteration": 0,
            "review": None,
        }
        with pytest.raises(ValueError, match="Planner produced no research aspects"):
            planner_node(state)
