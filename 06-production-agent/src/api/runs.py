"""The run registry: ``thread_id``-per-run lifecycle on top of
``src.durable``.

This module is where the Deployment experiment's API contract meets the
Durable Execution experiment's machinery -- deliberately reused, not
reimplemented:

* one HTTP "run" == one durable-execution ``thread_id`` (Durable
  Execution's own "使用 thread_id 管理独立 Agent executions" requirement,
  now surfaced as ``POST /runs``'s ``run_id``);
* ``POST /runs`` == :func:`src.durable.recovery.run_or_crash`;
* ``POST /runs/{id}/resume`` == :func:`src.durable.recovery.resume`
  (skips every already-checkpointed node -- Research A/B are never
  re-run; only the interrupted node continues);
* ``GET /runs/{id}/state`` == :func:`src.durable.recovery.get_execution_history`
  + :func:`src.durable.recovery.get_pending_tasks`.

Execution happens on a background thread (``asyncio.to_thread``) bounded
by the chosen :class:`~src.cost.policy.ExecutionPolicy`'s
``budget.timeout_seconds`` (Cost & Latency experiment's mode selection,
reused verbatim rather than inventing a second timeout knob) via
``src.reliability.timeout.run_with_timeout``, so ``POST /runs`` itself
returns immediately (status ``"running"``) instead of blocking the HTTP
request for the whole run -- exactly the "interrupted execution" +
"support resume" combination the experiment asks for: a slow run is
reported as ``"interrupted"`` (recoverable) rather than the request simply
timing out with no trace of what happened.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.cost.policy import ExecutionMode, get_policy
from src.durable import recovery
from src.durable.state import initial_state
from src.reliability.timeout import ToolTimeoutError, run_with_timeout
from src.security.authorization import Identity


class RunNotFoundError(KeyError):
    def __init__(self, run_id: str):
        super().__init__(f"No run found with run_id={run_id!r}.")
        self.run_id = run_id


class InvalidRunStateError(RuntimeError):
    """Raised by :meth:`RunRegistry.resume` when a run is not currently in
    a resumable state (e.g. it already completed, or is still running)."""

    def __init__(self, run_id: str, status: str):
        super().__init__(f"Run {run_id!r} is not resumable from status {status!r}.")
        self.run_id = run_id
        self.status = status


@dataclass
class RunRecord:
    """One HTTP-visible run. ``run_id`` and ``thread_id`` are the same
    value (see module docstring) -- kept as two fields anyway so the
    schemas/response models never have to explain "these are secretly
    aliases" to an API consumer."""

    run_id: str
    thread_id: str
    user_id: str
    tenant_id: str
    mode: str
    status: str = "running"  # running | completed | interrupted | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    final_report: Optional[str] = None
    error: Optional[str] = None


GraphFactory = Callable[[], Any]


class RunRegistry:
    """In-memory index of :class:`RunRecord`. NOT the durable state itself
    -- that lives in the checkpointer (``src.durable.checkpointer``),
    which is what actually survives a real process restart. This registry
    is a fast, queryable cache of "which thread_ids exist and what is
    their last known status"; it is deliberately rebuildable from the
    checkpointer alone (a real production version would hydrate it from
    ``graph.get_state_history`` for every known ``thread_id`` on startup,
    matching ``ApprovalStore``/``RateLimiter``'s own documented
    in-memory-registry caveat).
    """

    def __init__(self, graph_factory: GraphFactory) -> None:
        # A fresh graph object per run/resume call mirrors
        # ``recovery.resume``'s own documented requirement ("never the same
        # in-process object the crash happened on") -- the factory is what
        # lets tests substitute a graph wired with a one-shot
        # ``CrashInjector`` for the very first call, then a "clean" graph
        # for every call after (a real restarted worker process would
        # naturally get a "clean" graph too: the crash injector, if any,
        # only ever lived in the dead process's memory).
        self._graph_factory = graph_factory
        self._records: dict[str, RunRecord] = {}
        self._background_tasks: set[asyncio.Task] = set()

    def _resolve_owned(self, run_id: str, identity: Identity) -> RunRecord:
        """Requirement 5 (Security experiment): tenant isolation. Returns
        the record only if it belongs to ``identity``'s tenant; otherwise
        raises the same :class:`RunNotFoundError` as a genuinely unknown
        run_id -- a cross-tenant lookup must be indistinguishable from
        "does not exist" to the caller, never a distinguishable 403 that
        would confirm another tenant's run_id is valid."""
        record = self._records.get(run_id)
        if record is None or record.tenant_id != identity.tenant_id:
            raise RunNotFoundError(run_id)
        return record

    async def create_run(self, question: str, identity: Identity, mode: str) -> RunRecord:
        thread_id = uuid.uuid4().hex
        record = RunRecord(
            run_id=thread_id,
            thread_id=thread_id,
            user_id=identity.user_id,
            tenant_id=identity.tenant_id,
            mode=mode,
        )
        self._records[thread_id] = record
        state = initial_state(question)
        self._launch(record, lambda: recovery.run_or_crash(self._graph_factory(), state, thread_id))
        return record

    async def get_run(self, run_id: str, identity: Identity) -> RunRecord:
        return self._resolve_owned(run_id, identity)

    async def get_state(self, run_id: str, identity: Identity) -> tuple[RunRecord, list[recovery.HistoryEntry], tuple[str, ...]]:
        record = self._resolve_owned(run_id, identity)
        graph = self._graph_factory()
        history = recovery.get_execution_history(graph, record.thread_id)
        pending = recovery.get_pending_tasks(graph, record.thread_id)
        return record, history, pending

    async def resume_run(self, run_id: str, identity: Identity) -> RunRecord:
        record = self._resolve_owned(run_id, identity)
        if record.status not in ("interrupted", "failed"):
            raise InvalidRunStateError(run_id, record.status)
        record.status = "running"
        record.error = None
        record.updated_at = time.time()
        self._launch(record, lambda: recovery.resume(self._graph_factory(), record.thread_id))
        return record

    def _launch(self, record: RunRecord, call: Callable[[], recovery.RunOutcome]) -> None:
        timeout_seconds = get_policy(ExecutionMode(record.mode)).budget.timeout_seconds

        def _blocking() -> None:
            try:
                outcome = run_with_timeout(call, timeout_seconds, tool_name=f"run:{record.run_id}")
            except ToolTimeoutError as exc:
                record.status = "interrupted"
                record.error = str(exc)
            except Exception as exc:  # noqa: BLE001 -- recorded, never crashes the server process
                record.status = "failed"
                record.error = str(exc)
            else:
                if outcome.status == "crashed":
                    record.status = "interrupted"
                    record.error = str(outcome.crash)
                else:
                    record.status = "completed"
                    record.final_report = outcome.final_report
            finally:
                record.updated_at = time.time()

        task = asyncio.create_task(asyncio.to_thread(_blocking))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain(self, timeout_seconds: float = 30.0) -> None:
        """Requirement 9: graceful shutdown. Awaits every in-flight run
        task (bounded by ``timeout_seconds``) so a shutdown never abandons
        a run mid-flight without recording its outcome -- called from
        ``app.py``'s lifespan shutdown handler after readiness has already
        started reporting ``not_ready`` (so no *new* traffic is routed in
        while this drain is in progress)."""
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        for task in still_pending:
            task.cancel()


__all__ = ["RunNotFoundError", "InvalidRunStateError", "RunRecord", "RunRegistry", "GraphFactory"]
