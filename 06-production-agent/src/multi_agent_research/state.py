"""Shared state schema for the Multi-Agent Research System.

This is the single channel every node in ``graph.py`` communicates through
(Supervisor entry/exit, Planner, Research Workers, Synthesizer, Reviewer).
There is no other communication path between nodes: no globals, no shared
mutable objects, no direct function calls between agent modules.

Ownership convention (enforced by tests, mirroring experiment 5's Shared
State pattern): every node returns a *partial* update containing only the
key(s) it owns.

    node              owns (writes)
    ----              -------------
    supervisor_entry  trace only (bookkeeping, no domain field)
    planner           tasks, iteration
    research_worker   worker_results (reducer: each worker contributes one
                       item to the list; results accumulate across retries)
    synthesizer       synthesis
    reviewer          review
    finalizer         final_answer

``trace`` is contributed to by every node (it is the one deliberately
shared, append-only field -- see ``docs/14-MULTI-AGENT-RESEARCH.md``,
question 8, for why this single exception is safe).
"""

from __future__ import annotations

import operator
from typing import Literal, Optional

from typing_extensions import Annotated, TypedDict

# ---------------------------------------------------------------------------
# Reliability knobs (Requirement: max worker count, max review iterations,
# worker timeout -- all hard caps, independent of what any LLM proposes).
# ---------------------------------------------------------------------------

DEFAULT_MAX_WORKERS = 6
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_WORKER_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Structured task / worker-result / review types.
# ---------------------------------------------------------------------------


class ResearchTask(TypedDict):
    """One dynamically-planned unit of research work."""

    task_id: str
    aspect: str
    # Why this task exists: "initial decomposition" for iteration 1, or a
    # short note tying it back to the Reviewer's feedback for later
    # iterations. Purely informational (helps the Worker's own prompt and
    # helps humans read the trace); never used for control flow.
    reason: str


class WorkerResult(TypedDict):
    """Structured result every Research Worker must return -- never a raw
    string, and never anything that could be mistaken for another worker's
    result (Reliability requirement: structured worker result)."""

    task_id: str
    aspect: str
    status: Literal["completed", "timeout", "failed"]
    findings: str
    error: Optional[str]


class ReviewVerdict(TypedDict):
    """ReviewAgent's structured verdict, covering every dimension the
    Reviewer is required to check."""

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


# ---------------------------------------------------------------------------
# The shared state itself.
# ---------------------------------------------------------------------------


class MultiAgentResearchState(TypedDict):
    """The one and only channel every agent in this system communicates
    through (Requirement: structured state)."""

    question: str

    tasks: list[ResearchTask]
    # Reducer: every fanned-out research_worker instance appends exactly one
    # WorkerResult; results accumulate across review-loop iterations (a
    # retry does not erase earlier findings -- the Synthesizer re-synthesizes
    # from the full accumulated list every time).
    worker_results: Annotated[list[WorkerResult], operator.add]

    synthesis: str
    review: ReviewVerdict
    final_answer: str

    iteration: int
    max_iterations: int
    max_workers: int
    per_worker_timeout_seconds: float

    # Append-only audit log (Requirement: traceable state, mirrors
    # experiment 5's `trace` field). Every node contributes only its own
    # entries via the `operator.add` reducer.
    trace: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    per_worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> MultiAgentResearchState:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")

    return {
        "question": question,
        "tasks": [],
        "worker_results": [],
        "synthesis": "",
        "review": empty_review_verdict(),
        "final_answer": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_workers": max_workers,
        "per_worker_timeout_seconds": per_worker_timeout_seconds,
        "trace": [
            f"Supervisor: received question={question!r} "
            f"(max_workers={max_workers}, max_iterations={max_iterations}, "
            f"per_worker_timeout_seconds={per_worker_timeout_seconds})"
        ],
    }
