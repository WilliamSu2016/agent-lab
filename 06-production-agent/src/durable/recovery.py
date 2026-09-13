"""Crash simulation + restart/resume + execution-history helpers.

This module implements the full loop the experiment asks for:

    crash -> checkpoint -> restart -> resume

* **crash**: :class:`CrashInjector` + :class:`WorkerCrash` simulate a
  worker process dying mid-graph (an uncaught exception that escapes
  ``graph.invoke`` entirely -- nothing "handles" it inside the graph, by
  design, because a real process crash is not a caught exception either).
* **checkpoint**: handled by whatever ``BaseCheckpointSaver`` the graph was
  compiled with (see ``checkpointer.py``) -- every completed superstep
  (i.e. every node that returned normally) is durably persisted before the
  next one starts. This module does not implement checkpointing itself; it
  only *relies on* LangGraph's own checkpointing plus a durable backing
  store to make the rest of this loop possible.
* **restart**: :func:`resume` builds a brand-new graph object (a fresh
  ``StateGraph(...).compile(...)`` call -- standing in for a fresh worker
  process) against the *same* checkpointer/thread_id.
* **resume**: calling ``graph.invoke(None, config)`` (input ``None``) tells
  LangGraph "continue this thread from its last checkpoint" instead of
  starting a new run -- LangGraph itself skips every node whose completion
  is already recorded in the checkpoint and starts again from the first
  node that had not yet completed.

:func:`get_execution_history` / :func:`get_pending_tasks` expose the
*execution history* requirement independently of any of the above --
useful for observability/debugging even on a thread that never crashed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class WorkerCrash(RuntimeError):
    """Simulates an abrupt worker crash (e.g. OOM-kill, node failure,
    ``os._exit``) -- an uncaught, unrecoverable exception that escapes
    ``graph.invoke`` entirely. Deliberately NOT caught anywhere inside
    ``graph.py``'s nodes: a real process crash does not give the process a
    chance to catch anything either, and catching it there would defeat
    the point of testing checkpoint-based recovery."""


CrashWhen = Literal["before", "after"]


class CrashInjectorProtocol(Protocol):
    def before(self, task_id: str) -> None: ...

    def after(self, task_id: str) -> None: ...


@dataclass
class CrashInjector:
    """Test/demo harness that arms a *single-shot* crash for a given
    ``task_id``, firing either ``"before"`` the task's side effect would
    run (simulating a crash the instant the worker picked up the task, so
    the side effect never happened at all) or ``"after"`` it already ran
    (simulating a crash after the side effect executed but before the
    graph could commit/return that node's result -- the case that actually
    exercises idempotency, since the replayed node re-invokes the same
    side effect call).

    Single-shot by design: once a crash fires for a task_id, it is
    disarmed, so calling the same graph again after "restart" runs the
    task for real instead of crashing forever.
    """

    _armed: dict[str, CrashWhen] = field(default_factory=dict)

    def arm(self, task_id: str, when: CrashWhen = "after") -> None:
        self._armed[task_id] = when

    def is_armed(self, task_id: str) -> bool:
        return task_id in self._armed

    def before(self, task_id: str) -> None:
        if self._armed.get(task_id) == "before":
            del self._armed[task_id]
            raise WorkerCrash(f"Simulated worker crash before executing task {task_id!r} (no side effect ran).")

    def after(self, task_id: str) -> None:
        if self._armed.get(task_id) == "after":
            del self._armed[task_id]
            raise WorkerCrash(
                f"Simulated worker crash after task {task_id!r}'s side effect ran, "
                "before the graph could commit the result."
            )


@dataclass
class RunOutcome:
    """Result of one attempt to run/resume a durable graph."""

    status: Literal["completed", "crashed"]
    thread_id: str
    crash: WorkerCrash | None = None
    final_report: str | None = None
    results: dict[str, Any] | None = None


def _config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def run_or_crash(graph: Any, state: Any, thread_id: str) -> RunOutcome:
    """Start a brand-new run on ``thread_id``. Returns a ``"completed"``
    outcome on success, or a ``"crashed"`` outcome (never raises) if a
    :class:`WorkerCrash` escaped the run -- exactly what a supervising
    process would observe (the worker died) without needing to know
    anything about LangGraph internals."""
    config = _config_for(thread_id)
    try:
        result = graph.invoke(state, config)
    except WorkerCrash as exc:
        return RunOutcome(status="crashed", thread_id=thread_id, crash=exc)
    return RunOutcome(
        status="completed",
        thread_id=thread_id,
        final_report=result["final_report"],
        results=result["results"],
    )


def resume(graph: Any, thread_id: str) -> RunOutcome:
    """Resume a previously-started (and possibly crashed) run on
    ``thread_id``.

    Passing ``None`` as the input (instead of a fresh initial state) is
    what tells LangGraph "continue an existing thread from its last
    checkpoint" rather than "start a new run" -- LangGraph re-derives the
    next node(s) to execute from the checkpoint's own bookkeeping
    (``channel_versions``/``next``), so any node whose completion was
    already durably checkpointed (Research A, Research B in this
    experiment's crash scenario) is never re-invoked; only the node(s)
    that had not yet completed when the crash happened run again.

    ``graph`` should be a *newly constructed* graph object (see this
    module's docstring: "restart" = a fresh ``StateGraph(...).compile()``
    call) pointed at the same checkpointer/``thread_id`` as the crashed
    run -- never the same in-process object the crash happened on, or the
    test would not actually be exercising cross-restart recovery.
    """
    config = _config_for(thread_id)
    try:
        result = graph.invoke(None, config)
    except WorkerCrash as exc:
        return RunOutcome(status="crashed", thread_id=thread_id, crash=exc)
    return RunOutcome(
        status="completed",
        thread_id=thread_id,
        final_report=result["final_report"],
        results=result["results"],
    )


@dataclass
class HistoryEntry:
    """One entry of a thread's execution history -- one durable checkpoint,
    newest first (mirrors ``graph.get_state_history``'s own order)."""

    step: int
    next_tasks: tuple[str, ...]
    trace_so_far: list[str]
    results_so_far: dict[str, Any]


def get_execution_history(graph: Any, thread_id: str) -> list[HistoryEntry]:
    """Requirement: "支持查看 execution history." Returns every checkpoint
    recorded for ``thread_id``, newest first -- i.e. the full durable
    audit trail of the run, independent of whether it ever crashed. Each
    entry's ``next_tasks`` shows which node(s) LangGraph would run next
    from that point (empty once the run has reached ``END``)."""
    config = _config_for(thread_id)
    history: list[HistoryEntry] = []
    for snapshot in graph.get_state_history(config):
        values = snapshot.values or {}
        history.append(
            HistoryEntry(
                step=snapshot.metadata.get("step", -1) if snapshot.metadata else -1,
                next_tasks=tuple(snapshot.next),
                trace_so_far=list(values.get("trace", [])),
                results_so_far=dict(values.get("results", {})),
            )
        )
    return history


def get_pending_tasks(graph: Any, thread_id: str) -> tuple[str, ...]:
    """The node(s) that would run if the thread were resumed right now --
    empty once the run has completed. Used to assert, right after a
    simulated crash, exactly which task the run is stuck on (e.g.
    ``("research_c",)``) before ever calling :func:`resume`."""
    config = _config_for(thread_id)
    snapshot = graph.get_state(config)
    return tuple(snapshot.next)


__all__ = [
    "WorkerCrash",
    "CrashInjector",
    "CrashInjectorProtocol",
    "RunOutcome",
    "HistoryEntry",
    "run_or_crash",
    "resume",
    "get_execution_history",
    "get_pending_tasks",
]
