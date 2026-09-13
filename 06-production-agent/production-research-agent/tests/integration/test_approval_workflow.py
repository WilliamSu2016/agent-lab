"""End-to-end proof that a HIGH-risk tool reachable from the real graph
(``send_email`` via the optional ``notify_action`` node) goes
``Agent -> approval -> Tool``, never ``Agent -> Tool`` directly -- closing
the Final Review's finding that the HIGH-risk approval path was
implemented (``ApprovalStore``) but never wired into anything an actual
request could reach.

Scenario:

    1. A run is started with ``notify_email`` set -> the full
       Planner/Researcher/Synthesizer/Reviewer pipeline completes
       normally, then ``notify_action`` fires, finds no approval decided
       yet, and raises ``ApprovalRequiredError`` *uncaught* -- the run
       "crashes" (mirroring exactly how a worker crash is reported) and
       is durably checkpointed with ``notify_action`` as the sole
       pending task.
    2. Nobody has approved it yet: resuming immediately keeps raising the
       *same* ``ApprovalRequiredError`` (idempotent -- no duplicate email,
       no duplicate approval request).
    3. An operator approves via ``ApprovalStore.decide(...)``.
    4. Resuming now actually invokes the ``send_email`` tool function
       exactly once and the run completes with ``notify_status="sent"``.

A parallel test proves the "denied" branch never invokes the tool at all.
"""

from __future__ import annotations

import json

import pytest

from src.graph import recovery
from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import build_graph
from src.graph.state import initial_state
from src.security.authorization import ApprovalRequiredError, ApprovalStore

IDENTITY = {"user_id": "u1", "tenant_id": "t1", "roles": ["admin"]}


def _planner_llm(system: str, user: str) -> str:
    return json.dumps(["Research A"])


def _researcher_llm(system: str, user: str) -> str:
    return "Finding for A."


def _synth_llm(system: str, user: str) -> str:
    return "Synthesized report."


def _review_llm(system: str, user: str) -> str:
    return "APPROVE\nComplete."


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "approval_checkpoints.sqlite3"


def _build(checkpointer, approval_store, sent_emails):
    return build_graph(
        _planner_llm,
        _researcher_llm,
        _synth_llm,
        _review_llm,
        checkpointer=checkpointer,
        approval_store=approval_store,
        send_email_tool_fn=lambda to, subject, body: sent_emails.append((to, subject, body)) or "ok",
    )


class TestApprovalWorkflow:
    def test_high_risk_notify_blocks_until_approved_then_sends_exactly_once(self, db_path):
        thread_id = "approval-flow-1"
        sent_emails: list = []
        # Not durable across "restart" in this test suite either (same
        # documented caveat as InMemoryIdempotencyStore) -- the caller
        # (here: the test, standing in for a real approval-queue service)
        # must keep passing the SAME store across "restarts".
        approval_store = ApprovalStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = _build(checkpointer, approval_store, sent_emails)
            state = initial_state(
                "Research topic", IDENTITY, mode="balanced", max_workers=1, max_iterations=1, notify_email="ops@example.com"
            )
            outcome = recovery.run_or_crash(graph, state, thread_id)

            assert outcome.status == "crashed"
            assert isinstance(outcome.crash, ApprovalRequiredError)
            assert sent_emails == []  # never invoked before approval
            assert recovery.get_pending_tasks(graph, thread_id) == ("notify_action",)

        # Resuming again before anyone approves: still blocked, still no
        # email sent, and (thanks to the idempotency key) still the SAME
        # pending approval request -- not a second one.
        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = _build(checkpointer2, approval_store, sent_emails)
            resumed = recovery.resume(graph2, thread_id)
            assert resumed.status == "crashed"
            assert isinstance(resumed.crash, ApprovalRequiredError)
            assert sent_emails == []
            pending_ids = [r.request_id for r in approval_store._requests.values()]
            assert len(pending_ids) == 1

        # An operator approves the pending request out-of-band.
        request_id = next(iter(approval_store._requests))
        approval_store.decide(request_id, approved=True, approved_by="ops-admin")

        with sqlite_checkpointer(db_path) as checkpointer3:
            graph3 = _build(checkpointer3, approval_store, sent_emails)
            resumed2 = recovery.resume(graph3, thread_id)

            assert resumed2.status == "completed"
            assert resumed2.state["notify_status"] == "sent"
            assert len(sent_emails) == 1
            to, subject, body = sent_emails[0]
            assert to == "ops@example.com"
            assert subject == "Research complete: Research topic"
            assert "Synthesized report." in body
            assert recovery.get_pending_tasks(graph3, thread_id) == ()

    def test_high_risk_notify_never_calls_the_tool_when_denied(self, db_path):
        thread_id = "approval-flow-2"
        sent_emails: list = []
        approval_store = ApprovalStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = _build(checkpointer, approval_store, sent_emails)
            state = initial_state(
                "Research topic", IDENTITY, mode="balanced", max_workers=1, max_iterations=1, notify_email="ops@example.com"
            )
            outcome = recovery.run_or_crash(graph, state, thread_id)
            assert isinstance(outcome.crash, ApprovalRequiredError)

        request_id = next(iter(approval_store._requests))
        approval_store.decide(request_id, approved=False, approved_by="ops-admin")

        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = _build(checkpointer2, approval_store, sent_emails)
            resumed = recovery.resume(graph2, thread_id)

            assert resumed.status == "completed"
            assert resumed.state["notify_status"] == "denied"
            assert sent_emails == []  # the tool function is NEVER invoked when denied

    def test_normal_runs_without_notify_email_never_touch_the_approval_gate(self, db_path):
        """A regular research run (no ``notify_email``) must be entirely
        unaffected -- ``notify_action`` is a true no-op node in the
        default path."""
        thread_id = "approval-flow-noop"
        sent_emails: list = []
        approval_store = ApprovalStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = _build(checkpointer, approval_store, sent_emails)
            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=1, max_iterations=1)
            outcome = recovery.run_or_crash(graph, state, thread_id)

            assert outcome.status == "completed"
            assert outcome.state["notify_status"] == ""
            assert sent_emails == []
            assert approval_store._requests == {}
