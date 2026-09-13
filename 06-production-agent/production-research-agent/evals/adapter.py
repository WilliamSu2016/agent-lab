"""``evals.adapter``: wraps the real graph (``src.graph.graph.build_graph``)
as a :data:`~evals.runner.SystemUnderTest`.

Uses deterministic stub LLM callables (never a real network call) so
``run_offline_evaluation``/CI runs are fast, free, and 100% reproducible
-- exactly the same "swap the LLM call, keep everything else real"
pattern every failure-injection/security test in this package already
uses. Every other component in the path is the REAL production code:
the real ``StateGraph``, the real three-layer guardrail pipeline
(``src.security.guardrails``), the real HIGH-risk approval gate
(``src.security.authorization.ApprovalStore``), and a real file-backed
SQLite checkpointer (never ``InMemorySaver``, per this package's own
Durable Execution rule -- even for evaluation runs).

Stub LLM design (why the stubs echo their input verbatim): the Planner
stub emits the raw question as its one research aspect, the Researcher
stub echoes that aspect back into its "finding", and the Synthesizer stub
echoes the full accumulated prompt into the synthesis. The net effect: any
substring of a task's ``input`` question that the Researcher/Synthesizer
handled correctly is guaranteed to reach ``final_answer`` verbatim -- so
``evals.dataset.tasks``'s ``key_points`` genuinely test "did the pipeline's
plumbing preserve and forward content end-to-end" (state field ownership,
fan-out/fan-in, output-guardrail sanitization), not "is any specific LLM
factually correct" (a real deployment would additionally run this same
dataset through a second adapter backed by a real model, on a slower
cadence, to catch model-quality regressions -- out of scope for this
fast/offline/CI-safe adapter).
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from evals.dataset.tasks import EvalTask
from evals.runner import AgentRunResult, SystemUnderTest, ToolCallRecord
from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import build_graph
from src.graph.state import IdentityDict, initial_state
from src.observability.tracing import ExecutionContext, bind_execution_context
from src.security.authorization import ApprovalRequiredError, ApprovalStore, AuthorizationError

_URL_RE = re.compile(r"https?://\S+")

# Tasks that must run as a non-admin identity, to exercise the Layer 2
# role-authorization rejection path for a HIGH-risk tool.
_INSUFFICIENT_ROLE_TASKS = frozenset({"attack-insufficient-role-notify"})
# Tasks that set ``notify_email`` (fire the optional HIGH-risk
# ``notify_action`` node at all).
_NOTIFY_TASKS = frozenset({"notify-approved-send-email", "notify-denied-send-email", "attack-insufficient-role-notify"})
_DENIED_NOTIFY_TASKS = frozenset({"notify-denied-send-email"})


def _planner_llm(system: str, user: str) -> str:
    return json.dumps([user])


def _researcher_llm(system: str, user: str) -> str:
    return f"Finding: {user}"


def _synth_llm(system: str, user: str) -> str:
    return f"Synthesized report. {user}"


def _review_llm(system: str, user: str) -> str:
    return "APPROVE\nComplete and internally consistent."


def make_adapter(*, agent_version: str = "eval-1.0.0", environment: str = "evaluation") -> SystemUnderTest:
    """Builds one :data:`SystemUnderTest` closure. A fresh, file-backed
    SQLite checkpointer directory is created once per adapter (not once
    per task) and every task runs on its own ``thread_id`` within it --
    mirroring how a real deployment shares one checkpointer across many
    concurrent runs."""
    db_dir = Path(tempfile.mkdtemp(prefix="eval_checkpoints_"))
    db_path = db_dir / "eval_checkpoints.sqlite3"

    def run(task: EvalTask) -> AgentRunResult:
        search_calls: list[dict] = []
        sent_emails: list[tuple[str, str, str]] = []
        approval_store = ApprovalStore()

        def search_tool_fn(query: str) -> str:
            index = len(search_calls) + 1
            search_calls.append({"query": query, "url": f"https://example.com/search?doc={index}"})
            return f"[search result #{index}] {query}"

        def send_email_tool_fn(to: str, subject: str, body: str) -> str:
            sent_emails.append((to, subject, body))
            return f"sent to {to}"

        roles = ["researcher"] if task.task_id in _INSUFFICIENT_ROLE_TASKS else ["admin"]
        identity: IdentityDict = {"user_id": "eval-user", "tenant_id": "eval-tenant", "roles": roles}
        notify_email = "ops@example.com" if task.task_id in _NOTIFY_TASKS else ""

        thread_id = f"eval-{task.task_id}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

        route = "completed"
        terminated = True
        termination_reason = "completed"
        safety_blocked = False
        final_state: Optional[dict] = None

        started = time.perf_counter()
        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(
                _planner_llm,
                _researcher_llm,
                _synth_llm,
                _review_llm,
                checkpointer=checkpointer,
                approval_store=approval_store,
                search_tool_fn=search_tool_fn,
                send_email_tool_fn=send_email_tool_fn,
            )
            context = ExecutionContext(
                request_id=uuid.uuid4().hex,
                trace_id=thread_id,
                user_id=identity["user_id"],
                session_id=thread_id,
                agent_version=agent_version,
                environment=environment,
            )
            state = initial_state(
                task.input,
                identity,
                context=context.to_dict(),
                mode="balanced",
                max_workers=1,
                max_iterations=1,
                notify_email=notify_email,
            )

            try:
                with bind_execution_context(context):
                    final_state = graph.invoke(state, config)
            except ApprovalRequiredError:
                # A human decides right now (this adapter runs one task
                # synchronously end-to-end; multi-step "crash -> approve
                # -> resume over HTTP" is covered separately in
                # tests/integration/test_approval_workflow.py).
                request_id = next(iter(approval_store._requests))
                approve = task.task_id not in _DENIED_NOTIFY_TASKS
                approval_store.decide(request_id, approved=approve, approved_by="eval-approver")
                try:
                    with bind_execution_context(context):
                        final_state = graph.invoke(None, config)
                except ApprovalRequiredError:
                    terminated, termination_reason, route = False, "pending_human_approval", "pending_approval"
            except AuthorizationError as exc:
                safety_blocked, route, termination_reason = True, "blocked_authorization", f"authorization_denied: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        final_answer, cost_usd, iteration = "", 0.0, 0
        if final_state is not None:
            final_answer = final_state.get("final_answer", "")
            cost_usd = final_state.get("cost_usd", 0.0)
            iteration = final_state.get("iteration", 0)
            if final_state.get("blocked") and route == "completed":
                route, safety_blocked = "blocked_input_guardrail", True
            notify_status = final_state.get("notify_status", "")
            if notify_status == "sent":
                route = "notify_sent"
            elif notify_status == "denied":
                route = "notify_denied"  # a legitimate denial, not a safety block

        tool_calls = tuple(
            ToolCallRecord(name="search_web", arguments={"query": call["query"]}, approved=True) for call in search_calls
        )
        tool_calls += tuple(
            ToolCallRecord(name="send_email", arguments={"to": to, "subject": subject, "body": body}, approved=True)
            for to, subject, body in sent_emails
        )

        evidence = tuple(f"{call['query']}" for call in search_calls)
        citations = tuple(call["url"] for call in search_calls)

        return AgentRunResult(
            final_answer=final_answer,
            route=route,
            tool_calls=tool_calls,
            terminated=terminated,
            termination_reason=termination_reason,
            retry_count=max(0, iteration - 1),
            safety_blocked=safety_blocked,
            evidence=evidence,
            citations=citations,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    return run


__all__ = ["make_adapter"]
