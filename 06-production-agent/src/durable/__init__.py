"""Durable Execution experiment (LangGraph).

Requirement: an Agent graph must survive a worker crash without losing
progress and without re-executing side effects that already succeeded.
This package demonstrates the full loop:

    crash -> checkpoint -> restart -> resume

using a *production-capable* checkpointer (file-backed SQLite via
``langgraph.checkpoint.sqlite.SqliteSaver``) instead of
``langgraph.checkpoint.memory.InMemorySaver`` (which loses all state the
moment the process exits -- see ``docs/03-DURABLE-EXECUTION.md``).

Modules
-------
``state``
    The ``DurableResearchState`` schema shared by every node (mirrors
    ``src/multi_agent_research/state.py``'s ownership convention).
``graph``
    A small, deliberately linear 3-step "Research A -> Research B ->
    Research C -> Finalize" graph. Linear (not fanned out) on purpose: it
    makes "which steps already committed a checkpoint" trivial to reason
    about and verify in tests.
``checkpointer``
    Factory for the production-capable SQLite checkpointer, plus the
    documented reasons ``InMemorySaver`` is unacceptable in production.
``recovery``
    Crash simulation (``CrashInjector``, ``WorkerCrash``) and the
    restart/resume/execution-history helpers
    (``run_or_crash``, ``resume``, ``get_execution_history``).
"""

from src.durable.checkpointer import sqlite_checkpointer
from src.durable.graph import build_durable_graph, initial_state
from src.durable.recovery import (
    CrashInjector,
    WorkerCrash,
    get_execution_history,
    get_pending_tasks,
    resume,
    run_or_crash,
)

__all__ = [
    "sqlite_checkpointer",
    "build_durable_graph",
    "initial_state",
    "CrashInjector",
    "WorkerCrash",
    "get_execution_history",
    "get_pending_tasks",
    "resume",
    "run_or_crash",
]
