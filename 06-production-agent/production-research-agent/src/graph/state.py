"""Shared state schema for the Production Research Agent graph.

Extends the original Multi-Agent Research System's state
(question/tasks/worker_results/synthesis/review/final_answer/iteration/
trace) with everything the Final Review's P0 findings required to make
this the graph that actually runs in production, not a side experiment:

    identity        -- explicit Identity propagation (Security Layer 2/
                        Requirement 4), never ambient/thread-local state.
    mode            -- which cost/latency ExecutionPolicy this run uses
                        (Fast/Balanced/Quality).
    blocked /
    blocked_reason  -- set by the Supervisor's input guardrail (Layer 1)
                        or the Finalizer's output guardrail (Layer 2/8) --
                        a blocked run short-circuits straight to
                        ``finalizer`` without ever reaching the Planner/
                        Researcher/Synthesizer/Reviewer loop.
    tokens_used /
    cost_usd        -- accumulated (reducer: ``operator.add``) across every
                        LLM call in the run, so ``src.cost.budget.BudgetTracker``
                        can be rehydrated from state alone at any point.

Only JSON-serializable field types are used (str/int/float/bool/list/dict)
-- deliberately no ``Identity``/``ExecutionPolicy`` dataclasses stored
directly in state, so this state survives the production SQLite
checkpointer's serialization unmodified and can be inspected as plain data
from ``get_execution_history`` without any framework-specific
deserialization step.

Ownership convention (unchanged from the original experiment, enforced by
tests): every node returns a partial update containing only the key(s) it
owns.
"""

from __future__ import annotations

import operator
from typing import Literal, Optional

from typing_extensions import Annotated, TypedDict

DEFAULT_MAX_WORKERS = 6
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_WORKER_TIMEOUT_SECONDS = 30.0


class ResearchTask(TypedDict):
    task_id: str
    aspect: str
    reason: str


class WorkerResult(TypedDict):
    task_id: str
    aspect: str
    status: Literal["completed", "timeout", "failed", "blocked"]
    findings: str
    error: Optional[str]


class ReviewVerdict(TypedDict):
    approved: bool
    completeness: str
    factual_consistency: str
    evidence_quality: str
    logical_consistency: str
    missing_aspects: list[str]
    feedback: str


def empty_review_verdict() -> ReviewVerdict:
    return {
        "approved": False,
        "completeness": "",
        "factual_consistency": "",
        "evidence_quality": "",
        "logical_consistency": "",
        "missing_aspects": [],
        "feedback": "",
    }


class IdentityDict(TypedDict):
    """Plain-data mirror of ``src.security.authorization.Identity`` --
    kept as a TypedDict (not the dataclass itself) so it round-trips
    through the checkpointer's JSON-based serializer unchanged."""

    user_id: str
    tenant_id: str
    roles: list[str]


class ExecutionContextDict(TypedDict):
    """Plain-data mirror of ``src.observability.tracing.ExecutionContext``.

    Stored in state (not just held ambiently via ``contextvars``) because
    ``src.reliability.timeout.run_with_timeout`` runs its wrapped callable
    on a fresh raw thread (see that module's docstring), and Python's
    ``contextvars`` are *not* copied into a plain ``threading.Thread``
    started that way. The Researcher node's real work (tool call + LLM
    call, each its own tracer span) executes inside exactly such a thread
    (once directly via ``run_with_timeout``, and again per parallel
    ``Send`` branch), so it cannot rely on the ambient
    ``current_execution_context()``/``current_span()`` contextvars set by
    the outer ``bind_execution_context`` -- it must reconstruct its
    ``ExecutionContext`` from this field and pass it (plus an explicit
    ``parent`` span looked up via ``tracer.get_trace(trace_id)``) into
    every ``tracer.span(...)`` call it makes. See
    ``src.observability.tracing``'s module docstring ("Design note:
    parallel fan-out and ``contextvars``") for the underlying reason."""

    request_id: str
    trace_id: str
    user_id: str
    session_id: str
    agent_version: str
    environment: str


class ProductionResearchState(TypedDict):
    """The one channel every node communicates through."""

    question: str
    identity: IdentityDict
    context: ExecutionContextDict
    mode: str  # "fast" | "balanced" | "quality" -- src.cost.policy.ExecutionMode

    tasks: list[ResearchTask]
    worker_results: Annotated[list[WorkerResult], operator.add]

    synthesis: str
    review: ReviewVerdict
    final_answer: str

    iteration: int
    max_iterations: int
    max_workers: int
    per_worker_timeout_seconds: float

    tokens_used: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]

    blocked: bool
    blocked_reason: str

    # Optional HIGH-risk "notify by email" side effect, demonstrating the
    # mandatory Requirement 7 gate ("Agent -> approval -> Tool", not
    # "Agent -> Tool") reachable end-to-end from a real graph run. Empty
    # string (the default) means "do not notify" -- this never fires for
    # a normal research run; see ``src.agents.supervisor.make_notify_node``.
    notify_email: str
    notify_status: str  # "" | "sent" | "pending_approval" | "denied"

    trace: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    identity: IdentityDict,
    *,
    context: Optional[ExecutionContextDict] = None,
    mode: str = "balanced",
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    per_worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    notify_email: str = "",
) -> ProductionResearchState:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")

    if context is None:
        import uuid as _uuid

        trace_id = _uuid.uuid4().hex
        context = {
            "request_id": _uuid.uuid4().hex,
            "trace_id": trace_id,
            "user_id": identity["user_id"],
            "session_id": trace_id,
            "agent_version": "dev",
            "environment": "development",
        }

    return {
        "question": question,
        "identity": identity,
        "context": context,
        "mode": mode,
        "tasks": [],
        "worker_results": [],
        "synthesis": "",
        "review": empty_review_verdict(),
        "final_answer": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_workers": max_workers,
        "per_worker_timeout_seconds": per_worker_timeout_seconds,
        "tokens_used": 0,
        "cost_usd": 0.0,
        "blocked": False,
        "blocked_reason": "",
        "notify_email": notify_email,
        "notify_status": "",
        "trace": [
            f"Supervisor: received question={question!r} user_id={identity['user_id']!r} "
            f"tenant_id={identity['tenant_id']!r} mode={mode!r} "
            f"(max_workers={max_workers}, max_iterations={max_iterations})"
        ],
    }


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_WORKER_TIMEOUT_SECONDS",
    "ResearchTask",
    "WorkerResult",
    "ReviewVerdict",
    "empty_review_verdict",
    "IdentityDict",
    "ExecutionContextDict",
    "ProductionResearchState",
    "initial_state",
]
