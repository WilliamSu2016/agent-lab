"""Unified error classification and structured failure types.

Error handling requirement enforced by this module: a tool failure must
never simply ``raise`` and crash the caller. Instead every tool boundary in
this project should follow the same pipeline:

    Tool failure -> classify -> retry / fallback / terminate
                 -> structured error -> trace

This module provides the *classify* and *structured error* steps (shared by
``src/reliability/retry.py``, which layers *retry* on top); the *trace* step
is a caller-supplied hook (``on_error`` in ``retry.retry_call`` / any log
line built from ``ToolError.to_trace_line()``).

It also provides ``AgentLoopGuard``/``ControlledFailure`` -- the "Agent loop"
requirement: an agent's own Planner<->Reviewer-style retry loop must have a
``max_iterations`` budget, and exceeding it must produce a deliberate,
structured *controlled failure*, not an unbounded loop and not a crash.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class ErrorCategory(str, enum.Enum):
    """Every failure a tool call can produce is classified into exactly one
    of these categories. New categories should be added here first, then
    wired into ``RETRYABLE_CATEGORIES``/the classifiers below -- never
    inferred ad hoc at a call site.
    """

    # Retryable: transient, likely to succeed on a later attempt.
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"  # HTTP 5xx

    # Non-retryable: retrying without changing the request cannot help.
    INVALID_PARAMETERS = "invalid_parameters"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    BUSINESS_VALIDATION = "business_validation"

    # Unclassified. Treated as non-retryable by default (Reliability
    # principle: fail safe -- never retry a failure we don't understand).
    UNKNOWN = "unknown"


# Requirement: Retryable = timeout / temporary network error / rate limit / HTTP 5xx.
RETRYABLE_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER_ERROR,
    }
)

# Requirement: Non-retryable = invalid parameters / authentication failure /
# permission denied / business validation error.
NON_RETRYABLE_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.INVALID_PARAMETERS,
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.PERMISSION_DENIED,
        ErrorCategory.BUSINESS_VALIDATION,
        ErrorCategory.UNKNOWN,
    }
)

assert RETRYABLE_CATEGORIES.isdisjoint(NON_RETRYABLE_CATEGORIES)
assert RETRYABLE_CATEGORIES | NON_RETRYABLE_CATEGORIES == set(ErrorCategory)


def is_retryable_category(category: ErrorCategory) -> bool:
    return category in RETRYABLE_CATEGORIES


ClassifierFunc = Callable[[BaseException], ErrorCategory]


def classify_generic_exception(exc: BaseException) -> ErrorCategory:
    """Fallback classifier for plain-Python exceptions, used when no
    richer, library-specific classifier applies (e.g. a non-HTTP tool, or in
    tests). Always returns a category -- never raises, never returns
    ``None`` -- so it can always be used as a safe default.
    """
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, PermissionError):
        return ErrorCategory.PERMISSION_DENIED
    if isinstance(exc, ConnectionError):
        return ErrorCategory.NETWORK
    if isinstance(exc, OSError):
        # Broad OS-level I/O failures (DNS, socket reset, etc.) are treated
        # as transient network errors.
        return ErrorCategory.NETWORK
    if isinstance(exc, (ValueError, TypeError, KeyError, LookupError)):
        return ErrorCategory.INVALID_PARAMETERS
    return ErrorCategory.UNKNOWN


def classify_openai_exception(exc: BaseException) -> ErrorCategory:
    """Classify exceptions raised by the ``openai`` client (the concrete
    "external tool" this project currently calls). Falls back to
    :func:`classify_generic_exception` for anything not from that library,
    so this can be used as the default classifier for the LLM tool
    regardless of which underlying error actually occurred.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a required dependency here
        return classify_generic_exception(exc)

    if isinstance(exc, openai.APITimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, openai.RateLimitError):
        return ErrorCategory.RATE_LIMIT
    if isinstance(exc, openai.AuthenticationError):
        return ErrorCategory.AUTHENTICATION
    if isinstance(exc, openai.PermissionDeniedError):
        return ErrorCategory.PERMISSION_DENIED
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError, openai.NotFoundError)):
        return ErrorCategory.INVALID_PARAMETERS
    if isinstance(exc, openai.APIConnectionError):
        # Covers APITimeoutError's own parent too, but the more specific
        # check above already handled that case.
        return ErrorCategory.NETWORK
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status is None:
            return ErrorCategory.UNKNOWN
        if 500 <= status < 600:
            return ErrorCategory.SERVER_ERROR
        if status == 429:
            return ErrorCategory.RATE_LIMIT
        if status == 401:
            return ErrorCategory.AUTHENTICATION
        if status == 403:
            return ErrorCategory.PERMISSION_DENIED
        if status in (400, 404, 409, 422):
            return ErrorCategory.INVALID_PARAMETERS
        return ErrorCategory.UNKNOWN
    return classify_generic_exception(exc)


class BusinessValidationError(RuntimeError):
    """Raise this from tool/business logic to explicitly signal a
    non-retryable business-rule failure (e.g. "requested quantity exceeds
    remaining stock") -- distinct from an invalid *shape* of input
    (``ValueError``/``TypeError``, classified as ``INVALID_PARAMETERS``).
    Classified as :attr:`ErrorCategory.BUSINESS_VALIDATION`; never retried.
    """


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Default, dependency-aware classifier: tries the ``openai``-specific
    classifier first (covers this project's one real external tool client),
    then a couple of project-specific exception types, then the generic
    fallback. Prefer passing an explicit ``classify=...`` to
    ``retry.retry_call`` for tools with a different failure taxonomy; this
    function is the sensible default when none is supplied.
    """
    if isinstance(exc, BusinessValidationError):
        return ErrorCategory.BUSINESS_VALIDATION
    category = classify_openai_exception(exc)
    if category is not ErrorCategory.UNKNOWN:
        return category
    return classify_generic_exception(exc)


@dataclass(frozen=True)
class ToolError:
    """Structured representation of a single tool-call failure.

    This is what a tool boundary produces instead of letting a raw exception
    propagate: every field a caller needs to decide retry/fallback/terminate
    and to write one useful trace/log line, without re-parsing an exception
    message string.
    """

    category: ErrorCategory
    retryable: bool
    message: str
    tool_name: str
    exception_type: str
    attempt: int
    occurred_at: float = field(default_factory=time.time)
    # Kept for callers that need the original traceback (e.g. re-raising
    # with ``raise ... from original_exception``); excluded from
    # equality/repr so two otherwise-identical ToolErrors compare equal in
    # tests regardless of exception identity.
    original_exception: BaseException | None = field(default=None, repr=False, compare=False)

    def to_trace_line(self) -> str:
        return (
            f"ToolError tool={self.tool_name!r} category={self.category.value} "
            f"retryable={self.retryable!r} attempt={self.attempt} message={self.message!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "retryable": self.retryable,
            "message": self.message,
            "tool_name": self.tool_name,
            "exception_type": self.exception_type,
            "attempt": self.attempt,
            "occurred_at": self.occurred_at,
        }


def make_tool_error(
    exc: BaseException,
    *,
    tool_name: str,
    attempt: int,
    classify: ClassifierFunc = classify_exception,
) -> ToolError:
    """Classify ``exc`` and build the structured :class:`ToolError` for it."""
    category = classify(exc)
    message = str(exc).strip() or exc.__class__.__name__
    return ToolError(
        category=category,
        retryable=is_retryable_category(category),
        message=message,
        tool_name=tool_name,
        exception_type=type(exc).__qualname__,
        attempt=attempt,
        original_exception=exc,
    )


# ---------------------------------------------------------------------------
# Agent loop guard: "Agent loop 设置 max_iterations，超过后进入 controlled
# failure" -- a deliberate, structured stop, never a crash and never an
# unbounded loop.
# ---------------------------------------------------------------------------


class ControlledFailureError(RuntimeError):
    """Raised by :meth:`AgentLoopGuard.enter_iteration` once the configured
    ``max_iterations`` budget has been exceeded.

    This is a *designed* stop, not a bug: catching this and turning it into
    a structured, user-facing "we could not complete this within budget"
    result is the expected, correct handling -- never let it propagate as an
    unhandled crash, and never suppress it to keep looping past the budget.
    """

    def __init__(self, controlled_failure: "ControlledFailure"):
        super().__init__(controlled_failure.to_trace_line())
        self.controlled_failure = controlled_failure


@dataclass(frozen=True)
class ControlledFailure:
    """Structured terminal state for an agent loop that exhausted its
    iteration budget without reaching its goal (e.g. Reviewer never
    approved). Distinguish this from an ordinary ``ToolError``: it is a
    *loop-level* outcome, not a single call's failure."""

    reason: str
    iterations_used: int
    max_iterations: int
    last_feedback: str = ""
    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: float = field(default_factory=time.time)

    def to_trace_line(self) -> str:
        return (
            f"ControlledFailure reason={self.reason!r} "
            f"iterations_used={self.iterations_used} max_iterations={self.max_iterations} "
            f"failure_id={self.failure_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "iterations_used": self.iterations_used,
            "max_iterations": self.max_iterations,
            "last_feedback": self.last_feedback,
            "failure_id": self.failure_id,
            "occurred_at": self.occurred_at,
        }


class AgentLoopGuard:
    """Enforces ``max_iterations`` on any Planner/Reviewer-style retry loop.

    Usage::

        guard = AgentLoopGuard(max_iterations=3)
        while True:
            guard.enter_iteration()   # raises ControlledFailureError past budget
            ... do one iteration of work ...
            if done:
                break

    Or, for loops (like this project's existing Planner<->Reviewer graph
    edge) that prefer to *check* the budget rather than have it raise::

        if guard.budget_exhausted():
            failure = guard.as_controlled_failure(reason="review never approved")
            ... record failure, stop looping ...
    """

    def __init__(self, max_iterations: int):
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations!r}")
        self.max_iterations = max_iterations
        self.iterations_used = 0

    def enter_iteration(self) -> int:
        """Consume one iteration of budget; raises once the budget is
        exceeded. Returns the 1-based iteration number on success."""
        self.iterations_used += 1
        if self.iterations_used > self.max_iterations:
            raise ControlledFailureError(
                self.as_controlled_failure(
                    reason=f"exceeded max_iterations={self.max_iterations}"
                )
            )
        return self.iterations_used

    def budget_exhausted(self) -> bool:
        return self.iterations_used >= self.max_iterations

    def as_controlled_failure(self, reason: str, last_feedback: str = "") -> ControlledFailure:
        return ControlledFailure(
            reason=reason,
            iterations_used=self.iterations_used,
            max_iterations=self.max_iterations,
            last_feedback=last_feedback,
        )
