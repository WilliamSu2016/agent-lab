"""Crash simulation + restart/resume + execution-history helpers for the
real Production Research graph (adapted from the Durable Execution
experiment's ``src/durable/recovery.py`` -- same mechanics, applied to
``src.graph.state.ProductionResearchState`` instead of the toy demo
state).

    crash -> checkpoint -> restart -> resume

* **crash**: an uncaught exception escapes ``graph.invoke`` entirely
  (nothing inside a node catches it -- a real process crash is not a
  caught exception either).
* **checkpoint**: LangGraph's own checkpointer persists every completed
  superstep before the next one starts (see ``checkpointer.py``).
* **restart**: build a brand-new graph object against the same
  checkpointer + ``thread_id`` (a fresh ``StateGraph(...).compile()``
  call stands in for a fresh worker process).
* **resume**: ``graph.invoke(None, config)`` continues an existing thread
  from its last checkpoint -- every already-completed node (e.g. Planner,
  or an already-finished Researcher fan-out branch) is never re-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.observability.tracing import ExecutionContext, bind_execution_context


@dataclass
class RunOutcome:
    status: Literal["completed", "crashed"]
    thread_id: str
    crash: BaseException | None = None
    final_answer: str | None = None
    state: dict[str, Any] | None = None


def _config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}


def _context_from(values: dict[str, Any]) -> ExecutionContext | None:
    """Rebuilds the run's :class:`ExecutionContext` from state so
    ``run_or_crash``/``resume`` are self-sufficient -- they never require
    the caller to have already called ``bind_execution_context`` (a
    convenience; ``src.api.runs.RunRegistry`` still binds its own
    request-scoped context around the call too, which is harmless --
    nested ``bind_execution_context`` calls simply restore the previous
    value on exit)."""
    raw = values.get("context")
    if not raw:
        return None
    return ExecutionContext(**raw)


def run_or_crash(graph: Any, state: Any, thread_id: str) -> RunOutcome:
    """Start a brand-new run on ``thread_id``. Never raises: a crash is
    reported as a ``"crashed"`` :class:`RunOutcome`, exactly what a
    supervising process observes without any LangGraph-internal
    knowledge."""
    config = _config_for(thread_id)
    context = _context_from(state)
    try:
        if context is not None:
            with bind_execution_context(context):
                result = graph.invoke(state, config)
        else:
            result = graph.invoke(state, config)
    except BaseException as exc:  # noqa: BLE001 -- a real worker crash escapes everything
        return RunOutcome(status="crashed", thread_id=thread_id, crash=exc)
    return RunOutcome(status="completed", thread_id=thread_id, final_answer=result["final_answer"], state=dict(result))


def resume(graph: Any, thread_id: str) -> RunOutcome:
    """Resume a previously-started (and possibly crashed) run.
    ``graph`` must be a *newly constructed* graph object pointed at the
    same checkpointer/``thread_id`` -- never the same in-process object
    the crash happened on."""
    config = _config_for(thread_id)
    snapshot = graph.get_state(config)
    context = _context_from(snapshot.values or {})
    try:
        if context is not None:
            with bind_execution_context(context):
                result = graph.invoke(None, config)
        else:
            result = graph.invoke(None, config)
    except BaseException as exc:  # noqa: BLE001
        return RunOutcome(status="crashed", thread_id=thread_id, crash=exc)
    return RunOutcome(status="completed", thread_id=thread_id, final_answer=result["final_answer"], state=dict(result))


@dataclass
class HistoryEntry:
    step: int
    next_tasks: tuple[str, ...]
    trace_so_far: list[str]
    iteration: int
    worker_result_count: int


def get_execution_history(graph: Any, thread_id: str) -> list[HistoryEntry]:
    """Requirement: "支持查看 execution history." Every checkpoint recorded
    for ``thread_id``, newest first."""
    config = _config_for(thread_id)
    history: list[HistoryEntry] = []
    for snapshot in graph.get_state_history(config):
        values = snapshot.values or {}
        history.append(
            HistoryEntry(
                step=snapshot.metadata.get("step", -1) if snapshot.metadata else -1,
                next_tasks=tuple(snapshot.next),
                trace_so_far=list(values.get("trace", [])),
                iteration=int(values.get("iteration", 0)),
                worker_result_count=len(values.get("worker_results", [])),
            )
        )
    return history


def get_pending_tasks(graph: Any, thread_id: str) -> tuple[str, ...]:
    config = _config_for(thread_id)
    snapshot = graph.get_state(config)
    return tuple(snapshot.next)


__all__ = ["RunOutcome", "HistoryEntry", "run_or_crash", "resume", "get_execution_history", "get_pending_tasks"]
