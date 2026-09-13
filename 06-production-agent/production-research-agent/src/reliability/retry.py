"""Unified retry policy: exponential backoff + max retry count + retryable
exception classification -- usable by *any* Agent Tool call (LLM calls
today; any future external HTTP/MCP/business tool call tomorrow), not just
one hand-rolled implementation per call site.

Design:

* :class:`RetryPolicy` is the configuration (max retries, backoff base/cap,
  jitter). It intentionally does not import ``config.RetryPolicy`` -- this
  package (``src/reliability``) has no dependency on this specific project's
  configuration system, so it stays reusable on its own. Call sites convert
  ``config.RetryPolicy`` to ``reliability.retry.RetryPolicy`` (see
  ``src/specialists/llm.py``).
* :func:`retry_call` drives one call through the full Error-handling
  pipeline required by this experiment: classify -> retry (if retryable and
  budget remains) -> otherwise raise a structured, classified failure
  (never a bare re-raise, never an uncaught crash) -> optional trace hook.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from src.reliability.errors import (
    ClassifierFunc,
    ToolError,
    classify_exception,
    make_tool_error,
)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration for one tool.

    ``max_retries=0`` disables retries entirely: the first attempt is made
    and any failure (retryable or not) is raised immediately -- useful for
    deterministic tests and for a ``test`` environment that wants fast
    failures instead of waiting through a backoff schedule.
    """

    max_retries: int = 2
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 20.0
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries!r}")
        if self.backoff_base_seconds <= 0:
            raise ValueError(f"backoff_base_seconds must be > 0, got {self.backoff_base_seconds!r}")
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError(
                "backoff_max_seconds must be >= backoff_base_seconds "
                f"(got max={self.backoff_max_seconds!r} < base={self.backoff_base_seconds!r})."
            )
        if self.jitter_ratio < 0:
            raise ValueError(f"jitter_ratio must be >= 0, got {self.jitter_ratio!r}")


def compute_backoff_seconds(policy: RetryPolicy, attempt: int, *, rand: Callable[[], float] = random.random) -> float:
    """Exponential backoff with a cap and proportional jitter.

    ``attempt`` is 1-based and means "the delay before making attempt number
    ``attempt``" (i.e. this is called after ``attempt - 1`` failures already
    happened). The base delay doubles every attempt and is capped at
    ``policy.backoff_max_seconds``; a random jitter of up to
    ``jitter_ratio`` of the capped delay is added on top, so many concurrent
    callers retrying the same transient failure do not all retry in
    lock-step (thundering herd).
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt!r}")
    base_delay = policy.backoff_base_seconds * (2 ** (attempt - 1))
    capped_delay = min(base_delay, policy.backoff_max_seconds)
    jitter = capped_delay * policy.jitter_ratio * rand() if policy.jitter_ratio else 0.0
    return capped_delay + jitter


class ToolInvocationError(RuntimeError):
    """Raised when a tool call ultimately fails -- either because the
    failure was classified non-retryable (raised on the very first
    attempt), or because every retry attempt was used up and the last one
    still failed. Always wraps a structured :class:`~src.reliability.errors.ToolError`
    (never a bare exception) so every caller gets the same shape regardless
    of the underlying failure type.

    ``__cause__`` is set to the original exception (via ``raise ... from``),
    so a full traceback is still available for debugging without making the
    structured error itself carry a giant, hard-to-serialize traceback.
    """

    def __init__(self, tool_error: ToolError):
        super().__init__(tool_error.message)
        self.tool_error = tool_error


def retry_call(
    func: Callable[[], T],
    *,
    policy: RetryPolicy,
    tool_name: str,
    classify: ClassifierFunc = classify_exception,
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[ToolError], None] | None = None,
) -> T:
    """Call ``func()``, retrying only classified-retryable failures.

    Behaviour (Error handling requirement: "Tool failure -> classify ->
    retry / fallback / terminate -> structured error -> trace"):

    1. Every exception ``func()`` raises is classified via ``classify`` and
       wrapped into a :class:`~src.reliability.errors.ToolError` immediately
       -- the raw exception never reaches the caller directly.
    2. ``on_error`` (if given) is invoked with that ``ToolError`` every time
       -- this is the trace/log hook. It runs *before* the retry/terminate
       decision, so every attempt is observable even if a later attempt
       succeeds.
    3. If the failure is retryable (per Requirement: timeout / temporary
       network error / rate limit / HTTP 5xx) and attempts remain under
       ``policy.max_retries``, sleep for an exponentially-increasing,
       jittered delay (:func:`compute_backoff_seconds`) and try again.
    4. Otherwise (non-retryable -- Requirement: invalid parameters /
       authentication failure / permission denied / business validation
       error -- or retries exhausted) raise :class:`ToolInvocationError`
       wrapping the last ``ToolError``. This function never lets the raw
       exception propagate and never returns a partial/undefined result.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 -- reclassified into ToolError immediately below
            tool_error = make_tool_error(exc, tool_name=tool_name, attempt=attempt, classify=classify)
            if on_error is not None:
                on_error(tool_error)
            if not tool_error.retryable or attempt > policy.max_retries:
                raise ToolInvocationError(tool_error) from exc
            delay = compute_backoff_seconds(policy, attempt)
            sleep(delay)
