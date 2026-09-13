"""Failure #6: MCP unavailable.

**Disclaimer**: this repository has no real MCP (Model Context Protocol)
integration anywhere -- no MCP client, no MCP server config, no tool
registered over MCP. Per this experiment's "do not add new business
functionality" constraint, this file does not introduce one.

Instead, "MCP unavailable" is modeled the way any unreachable external
service is: a connection-level failure (``openai.APIConnectionError`` for
the LLM seam, or a generic ``ConnectionError`` for a hypothetical MCP tool
call), classified ``NETWORK`` -- which is retryable -- and routed through
the exact same retry/backoff/timeout/structured-failure pipeline already
proven for the LLM call and generic tool calls. If/when a real MCP
integration is added to this project, it should be wired through this same
``src.reliability`` pipeline rather than a bespoke handler, and these tests
extended to call the real client instead of this generic stand-in.
"""

from __future__ import annotations

import openai
import pytest
from openai.resources.chat.completions.completions import Completions

from src.multi_agent_research.research_worker import make_worker_node
from src.reliability.retry import RetryPolicy, ToolInvocationError, retry_call
from src.specialists.llm import build_openai_text_llm_call
from tests.failure_injection.conftest import openai_request


class TestMCPUnavailableViaLLMSeam:
    """The LLM call is this repository's one real outbound "external
    service" call; an MCP server being unreachable is, from the caller's
    point of view, indistinguishable from any other unreachable backend --
    a connection failure classified ``NETWORK``."""

    def test_connection_refused_is_retried_then_exhausts(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2)
        call_count = {"n": 0}

        def connection_refused(*args, **kwargs):
            call_count["n"] += 1
            raise openai.APIConnectionError(request=openai_request())

        monkeypatch.setattr(Completions, "create", connection_refused)
        text_llm_call = build_openai_text_llm_call(settings)

        with pytest.raises(ToolInvocationError) as exc_info:
            text_llm_call("system", "user")

        assert exc_info.value.tool_error.category.value == "network"
        assert exc_info.value.tool_error.retryable is True
        assert call_count["n"] == 3  # 1 initial + 2 retries

    def test_service_comes_back_within_retry_budget(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=2)
        call_count = {"n": 0}

        def _resp(content: str):
            class _Msg:
                pass

            class _Choice:
                pass

            class _Resp:
                pass

            msg = _Msg()
            msg.content = content
            choice = _Choice()
            choice.message = msg
            resp = _Resp()
            resp.choices = [choice]
            return resp

        def flaky_service(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise openai.APIConnectionError(request=openai_request())
            return _resp("mcp backend reachable again")

        monkeypatch.setattr(Completions, "create", flaky_service)
        text_llm_call = build_openai_text_llm_call(settings)

        assert text_llm_call("system", "user") == "mcp backend reachable again"
        assert call_count["n"] == 2


class TestMCPUnavailableGenericToolShape:
    """Same scenario, modeled with a plain ``ConnectionError`` -- the shape
    a hypothetical direct MCP client call (not going through the OpenAI
    client) would raise if its transport could not reach the server."""

    def test_generic_connection_error_is_network_and_retryable(self):
        call_count = {"n": 0}

        def mcp_server_unreachable() -> str:
            call_count["n"] += 1
            raise ConnectionError("MCP server unreachable: connection refused")

        policy = RetryPolicy(max_retries=2, backoff_base_seconds=0.001, backoff_max_seconds=0.005)
        with pytest.raises(ToolInvocationError) as exc_info:
            retry_call(mcp_server_unreachable, policy=policy, tool_name="mcp_tool")

        assert exc_info.value.tool_error.category.value == "network"
        assert call_count["n"] == 3


class TestMCPUnavailableFeedsWorkerStructuredFallback:
    """Same fan-out fallback proof as ``test_llm_failure.py``'s equivalent
    test: a persistent "MCP unavailable" failure must not crash the whole
    multi-agent run -- the worker node converts it into a structured
    ``WorkerResult(status="failed")`` instead."""

    def test_worker_reports_failed_status_instead_of_raising(self, monkeypatch, make_settings):
        settings = make_settings(max_retries=1, worker_timeout_seconds=1.0)

        def always_unreachable(*args, **kwargs):
            raise openai.APIConnectionError(request=openai_request())

        monkeypatch.setattr(Completions, "create", always_unreachable)
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
