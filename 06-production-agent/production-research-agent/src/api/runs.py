"""The run registry: ``thread_id``-per-run lifecycle on top of the real
Production Research graph (``src.graph.graph``/``src.graph.recovery``).

One HTTP "run" == one durable-execution ``thread_id``. Execution happens
on a background thread bounded by the chosen
``src.cost.policy.ExecutionPolicy``'s ``timeout_seconds``, so ``POST
/runs`` returns immediately (status ``"running"``) instead of blocking the
whole HTTP request for the run's full duration -- a slow/crashed run is
reported as ``"interrupted"`` (recoverable via ``POST /runs/{id}/resume``)
rather than the request simply timing out with no trace of what happened.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.cost.policy import ExecutionMode, get_policy
from src.graph import recovery
from src.graph.state import IdentityDict, initial_state
from src.observability.metrics import MetricsRegistry
from src.observability.tracing import ExecutionContext, SpanKind, Tracer, bind_execution_context
from src.reliability.timeout import ToolTimeoutError, run_with_timeout
from src.security.authorization import Identity


class RunNotFoundError(KeyError):
    def __init__(self, run_id: str):
        super().__init__(f"No run found with run_id={run_id!r}.")
        self.run_id = run_id


class InvalidRunStateError(RuntimeError):
    def __init__(self, run_id: str, status: str):
        super().__init__(f"Run {run_id!r} is not resumable from status {status!r}.")
        self.run_id = run_id
        self.status = status


@dataclass
class RunRecord:
    run_id: str
    thread_id: str
    user_id: str
    tenant_id: str
    mode: str
    status: str = "running"  # running | completed | interrupted | failed | blocked
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    final_answer: Optional[str] = None
    error: Optional[str] = None


GraphFactory = Callable[[], Any]


class RunRegistry:
    """In-memory index of :class:`RunRecord`. NOT the durable state itself
    -- that lives in the checkpointer, which is what actually survives a
    real process restart (documented limitation, same as
    ``src.security.authorization.ApprovalStore``/``src.api.rate_limit
    .RateLimiter``: a real multi-replica deployment needs this rebuilt
    from / backed by a shared store)."""

    def __init__(
        self,
        graph_factory: GraphFactory,
        *,
        tracer: Optional[Tracer] = None,
        metrics: Optional[MetricsRegistry] = None,
        agent_version: str = "dev",
        environment: str = "development",
    ) -> None:
        self._graph_factory = graph_factory
        self._records: dict[str, RunRecord] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._tracer = tracer or Tracer()
        self._metrics = metrics or MetricsRegistry()
        self._agent_version = agent_version
        self._environment = environment

    def _resolve_owned(self, run_id: str, identity: Identity) -> RunRecord:
        record = self._records.get(run_id)
        if record is None or record.tenant_id != identity.tenant_id:
            raise RunNotFoundError(run_id)
        return record

    async def create_run(self, question: str, identity: Identity, mode: str) -> RunRecord:
        thread_id = uuid.uuid4().hex
        record = RunRecord(
            run_id=thread_id, thread_id=thread_id, user_id=identity.user_id, tenant_id=identity.tenant_id, mode=mode
        )
        self._records[thread_id] = record
        identity_dict: IdentityDict = {
            "user_id": identity.user_id,
            "tenant_id": identity.tenant_id,
            "roles": sorted(identity.roles),
        }
        policy = get_policy(ExecutionMode(mode))
        context = ExecutionContext(
            request_id=uuid.uuid4().hex,
            trace_id=thread_id,
            user_id=identity.user_id,
            session_id=thread_id,
            agent_version=self._agent_version,
            environment=self._environment,
        )
        state = initial_state(
            question,
            identity_dict,
            context=context.to_dict(),
            mode=mode,
            max_workers=policy.max_workers,
            max_iterations=policy.max_agent_iterations,
        )
        self._launch(record, context, lambda: recovery.run_or_crash(self._graph_factory(), state, thread_id))
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
        context = ExecutionContext(
            request_id=uuid.uuid4().hex,
            trace_id=record.thread_id,
            user_id=identity.user_id,
            session_id=record.thread_id,
            agent_version=self._agent_version,
            environment=self._environment,
        )
        self._launch(record, context, lambda: recovery.resume(self._graph_factory(), record.thread_id))
        return record

    def _launch(self, record: RunRecord, context: ExecutionContext, call: Callable[[], recovery.RunOutcome]) -> None:
        timeout_seconds = get_policy(ExecutionMode(record.mode)).budget.timeout_seconds

        def _blocking() -> None:
            try:
                with bind_execution_context(context):
                    with self._tracer.span(SpanKind.WORKFLOW, "api_run", context=context) as root:
                        outcome = run_with_timeout(call, timeout_seconds, tool_name=f"run:{record.run_id}")
                        root.attributes["run_id"] = record.run_id
                self._metrics.record_from_trace(root)
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
                elif outcome.state and outcome.state.get("blocked"):
                    record.status = "blocked"
                    record.final_answer = outcome.final_answer
                else:
                    record.status = "completed"
                    record.final_answer = outcome.final_answer
            finally:
                record.updated_at = time.time()

        task = asyncio.create_task(asyncio.to_thread(_blocking))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def drain(self, timeout_seconds: float = 30.0) -> None:
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        _done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        for task in still_pending:
            task.cancel()


__all__ = ["RunNotFoundError", "InvalidRunStateError", "RunRecord", "RunRegistry", "GraphFactory"]
