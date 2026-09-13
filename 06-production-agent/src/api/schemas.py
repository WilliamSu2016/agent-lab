"""Requirement 2: request validation.

Every request/response body the API exchanges is a Pydantic model, so a
malformed body (missing field, wrong type, empty question, unknown risk
mode, ...) is rejected by FastAPI itself with a structured ``422`` before
any handler code -- let alone a graph node -- ever runs. This is the first
guardrail in the stack, upstream of ``src.security.guardrails`` (which
still runs on top, on the *content* of the validated fields -- validation
answers "is this shaped like a request?", not "is this safe?").
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RunStatus = Literal["running", "completed", "interrupted", "failed"]
ExecutionModeLiteral = Literal["fast", "balanced", "quality"]


class RunCreateRequest(BaseModel):
    """Body of ``POST /runs``."""

    question: str = Field(..., min_length=1, max_length=4000, description="The research question to run.")
    mode: ExecutionModeLiteral = Field(
        default="balanced",
        description="Cost/latency strategy (src.cost.policy.ExecutionMode) applied to this run.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque caller metadata, echoed back only.")


class RunSummary(BaseModel):
    """Response shape for ``POST /runs``, ``GET /runs/{id}`` and
    ``POST /runs/{id}/resume`` -- one consistent envelope for "what is the
    state of this run" regardless of which endpoint produced it."""

    run_id: str
    thread_id: str
    status: RunStatus
    mode: ExecutionModeLiteral
    user_id: str
    tenant_id: str
    created_at: float
    updated_at: float
    final_report: Optional[str] = None
    error: Optional[str] = None


class HistoryEntryResponse(BaseModel):
    step: int
    next_tasks: tuple[str, ...]
    trace_so_far: list[str]
    results_so_far: dict[str, Any]


class RunStateResponse(BaseModel):
    """Response shape for ``GET /runs/{id}/state``: the durable-execution
    view of a run -- exactly what a supervisor deciding whether/how to
    resume a run needs, independent of the lightweight ``RunSummary``."""

    run_id: str
    thread_id: str
    status: RunStatus
    pending_tasks: tuple[str, ...]
    history: list[HistoryEntryResponse]


class ErrorResponse(BaseModel):
    """Requirement 5: error handling. Every non-2xx response (validation
    errors aside, which use FastAPI's own body shape) is this shape --
    a client can always find ``request_id`` to correlate with server-side
    logs/traces, and ``message`` is always safe to display (never a raw
    stack trace or an internal exception's ``str()`` -- see
    ``src.api.errors``)."""

    request_id: str
    error_code: str
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


__all__ = [
    "RunStatus",
    "ExecutionModeLiteral",
    "RunCreateRequest",
    "RunSummary",
    "HistoryEntryResponse",
    "RunStateResponse",
    "ErrorResponse",
    "HealthResponse",
    "ReadyResponse",
]
