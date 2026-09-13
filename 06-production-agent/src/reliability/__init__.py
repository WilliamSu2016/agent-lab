"""Unified reliability policy for every Agent Tool call in this project.

This package is the single place that answers, for *any* tool a Agent calls
(today: the LLM client; tomorrow: any external HTTP/MCP/business tool):

* ``errors``      -- classify a failure (retryable vs not) and produce a
                      structured ``ToolError`` instead of letting a raw
                      exception crash the caller; also the ``AgentLoopGuard``
                      / ``ControlledFailure`` primitives for bounding an
                      agent's own retry/review loop.
* ``retry``        -- exponential backoff + max retry count, driven by the
                      classification above.
* ``timeout``      -- one canonical "run this call with a deadline"
                      implementation, replacing the three near-duplicate
                      ad hoc versions that existed in
                      ``research_worker.py``/``graph.py``/
                      ``04_parallel_multi_agent.py`` before this module.
* ``idempotency``  -- deduplication strategy for side-effecting tools
                      (``send_email``, ``create_order``, ``publish_report``,
                      ...) so a retry, a duplicate call, or a replayed step
                      never re-executes an external side effect.

See ``docs/02-RELIABILITY.md`` for the full design write-up and
``tests/test_retry.py`` / ``tests/test_timeout.py`` / ``tests/test_errors.py``
/ ``tests/test_failure_injection.py`` for the automated test suite.
"""

from __future__ import annotations

from src.reliability.errors import (
    AgentLoopGuard,
    BusinessValidationError,
    ClassifierFunc,
    ControlledFailure,
    ControlledFailureError,
    ErrorCategory,
    NON_RETRYABLE_CATEGORIES,
    RETRYABLE_CATEGORIES,
    ToolError,
    classify_exception,
    classify_generic_exception,
    classify_openai_exception,
    is_retryable_category,
    make_tool_error,
)
from src.reliability.idempotency import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
    IdempotencyRecord,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    idempotent,
    make_create_order_tool,
    make_publish_report_tool,
    make_send_email_tool,
)
from src.reliability.retry import RetryPolicy, ToolInvocationError, compute_backoff_seconds, retry_call
from src.reliability.timeout import ToolTimeoutError, TimeoutPolicy, run_with_timeout

__all__ = [
    "AgentLoopGuard",
    "BusinessValidationError",
    "ClassifierFunc",
    "ControlledFailure",
    "ControlledFailureError",
    "ErrorCategory",
    "NON_RETRYABLE_CATEGORIES",
    "RETRYABLE_CATEGORIES",
    "ToolError",
    "classify_exception",
    "classify_generic_exception",
    "classify_openai_exception",
    "is_retryable_category",
    "make_tool_error",
    "IdempotencyConflictError",
    "IdempotencyInProgressError",
    "IdempotencyRecord",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "idempotent",
    "make_create_order_tool",
    "make_publish_report_tool",
    "make_send_email_tool",
    "RetryPolicy",
    "ToolInvocationError",
    "compute_backoff_seconds",
    "retry_call",
    "ToolTimeoutError",
    "TimeoutPolicy",
    "run_with_timeout",
]
