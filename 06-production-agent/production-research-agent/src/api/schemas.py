"""Request/response validation models (Requirement: request validation).

Every request/response body is a Pydantic model, rejected with a
structured 422 before any handler code runs if malformed. This is
upstream of ``src.security.guardrails`` (content-level checks), not a
replacement for it -- both layers run on every request that reaches a
handler.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RunStatus = Literal["running", "completed", "interrupted", "failed", "blocked"]
ExecutionModeLiteral = Literal["fast", "balanced", "quality"]


class RunCreateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="The research question to run.")
    mode: ExecutionModeLiteral = Field(default="balanced", description="Cost/latency strategy for this run.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque caller metadata, echoed back only.")


class RunSummary(BaseModel):
    run_id: str
    thread_id: str
    status: RunStatus
    mode: ExecutionModeLiteral
    user_id: str
    tenant_id: str
    created_at: float
    updated_at: float
    final_answer: Optional[str] = None
    error: Optional[str] = None


class HistoryEntryResponse(BaseModel):
    step: int
    next_tasks: tuple[str, ...]
    trace_so_far: list[str]
    iteration: int
    worker_result_count: int


class RunStateResponse(BaseModel):
    run_id: str
    thread_id: str
    status: RunStatus
    pending_tasks: tuple[str, ...]
    history: list[HistoryEntryResponse]


class ApprovalDecisionRequest(BaseModel):
    approved: bool


class ApprovalResponse(BaseModel):
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_by: str
    status: str
    approved_by: Optional[str] = None


class ErrorResponse(BaseModel):
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
    "ApprovalDecisionRequest",
    "ApprovalResponse",
    "ErrorResponse",
    "HealthResponse",
    "ReadyResponse",
]
