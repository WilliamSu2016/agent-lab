"""Shared, minimal LLM call seam used by every Specialist Agent.

This is plain infrastructure (an HTTP client wrapper), not an Agent, not a
tool, and not a communication channel between agents. It exists only so each
Specialist Agent module can accept an injectable ``TextLLMCall`` for testing
(see ``tests/``) instead of hard-wiring the OpenAI client.

No agent in ``src/specialists/`` imports another agent's module, and no agent
calls another agent through this module. Each agent is wired independently
in its own ``main()``.

Configuration & Secrets: this module never reads ``os.environ`` and never
contains a hard-coded API key, model name, or endpoint. Every caller must
build the ``TextLLMCall`` from a validated ``config.Settings`` object (see
``config.load_settings()``), which is the *only* place secrets are read from
the environment.

Reliability: every call this module makes to the ``openai`` client (this
project's one real "external tool") goes through the unified
``src/reliability`` pipeline -- ``reliability.timeout.run_with_timeout`` (a
mandatory timeout on every external call) wrapping
``reliability.retry.retry_call`` (exponential backoff, classified
retry/non-retry, a structured ``ToolInvocationError`` instead of a bare
crash) -- rather than a bespoke retry loop private to this module. See
``docs/02-RELIABILITY.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from src.reliability.errors import classify_openai_exception
from src.reliability.retry import RetryPolicy as ReliabilityRetryPolicy
from src.reliability.retry import retry_call
from src.reliability.timeout import run_with_timeout

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

# A TextLLMCall takes (system_prompt, user_prompt) and returns the model's
# plain-text reply. Every Specialist Agent in this package uses this exact
# shape: one instructions block in, one text block out. No tool calling, no
# multi-turn state, no shared memory.
TextLLMCall = Callable[[str, str], str]


def build_openai_text_llm_call(settings: "Settings") -> TextLLMCall:
    """Build a ``TextLLMCall`` backed by the plain ``openai`` client.

    Uses the base Chat Completions client directly -- not the ``openai-agents``
    SDK, not LangChain's chat model wrappers -- so each Specialist Agent stays
    a single, plain LLM call wrapped in a one-node LangGraph graph.

    ``settings`` (a validated ``config.Settings``) supplies the API key,
    model name, base URL, retry policy, and worker timeout -- nothing here is
    hard-coded and nothing here reads the environment directly (Configuration
    & Secrets requirement).

    Reliability: each call is bounded by ``settings.limits.worker_timeout_seconds``
    (Reliability requirement: "每一个 external tool 必须有 timeout") and retried
    via ``reliability.retry.retry_call`` using ``classify_openai_exception``:
    retryable failures (timeout / network / rate limit / HTTP 5xx) are retried
    up to ``settings.retry.max_retries`` additional times with exponential
    backoff (capped at ``retry.backoff_max_seconds``) plus jitter;
    non-retryable failures (bad auth, bad request/validation, permission
    denied, etc.) raise immediately on the first attempt. Every failure --
    retried or not -- is logged as a structured trace line
    (``ToolError.to_trace_line()``) instead of only surfacing as a raw
    exception. On exhaustion this raises
    ``reliability.retry.ToolInvocationError`` (never the raw ``openai``
    exception, never an unhandled crash).
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    reliability_retry_policy = ReliabilityRetryPolicy(
        max_retries=settings.retry.max_retries,
        backoff_base_seconds=settings.retry.backoff_base_seconds,
        backoff_max_seconds=settings.retry.backoff_max_seconds,
    )
    timeout_seconds = settings.limits.worker_timeout_seconds

    def _trace(tool_error) -> None:
        logger.warning(tool_error.to_trace_line())

    def call(system_prompt: str, user_prompt: str) -> str:
        def _invoke() -> str:
            def _do_request() -> str:
                response = client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""

            return run_with_timeout(_do_request, timeout_seconds, tool_name="openai_chat_completion")

        return retry_call(
            _invoke,
            policy=reliability_retry_policy,
            tool_name="openai_chat_completion",
            classify=classify_openai_exception,
            on_error=_trace,
        )

    return call
