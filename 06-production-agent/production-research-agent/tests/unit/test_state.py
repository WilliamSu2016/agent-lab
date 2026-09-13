"""Unit tests: ``ProductionResearchState`` shape and ``initial_state``."""

from __future__ import annotations

import pytest

from src.graph.state import initial_state

IDENTITY = {"user_id": "u1", "tenant_id": "t1", "roles": ["researcher"]}


def test_initial_state_has_all_required_fields_with_sane_defaults():
    state = initial_state("What is LangGraph?", IDENTITY)

    assert state["question"] == "What is LangGraph?"
    assert state["identity"] == IDENTITY
    assert state["mode"] == "balanced"
    assert state["tasks"] == []
    assert state["worker_results"] == []
    assert state["synthesis"] == ""
    assert state["final_answer"] == ""
    assert state["iteration"] == 0
    assert state["tokens_used"] == 0
    assert state["cost_usd"] == 0.0
    assert state["blocked"] is False
    assert state["blocked_reason"] == ""
    assert state["notify_email"] == ""
    assert state["notify_status"] == ""
    assert len(state["trace"]) == 1


def test_initial_state_generates_a_full_execution_context_when_none_supplied():
    state = initial_state("q", IDENTITY)
    context = state["context"]
    for key in ("request_id", "trace_id", "user_id", "session_id", "agent_version", "environment"):
        assert context[key]
    assert context["user_id"] == "u1"


def test_initial_state_accepts_a_caller_supplied_context_verbatim():
    context = {
        "request_id": "r1",
        "trace_id": "t1",
        "user_id": "u1",
        "session_id": "s1",
        "agent_version": "v1.2.3",
        "environment": "production",
    }
    state = initial_state("q", IDENTITY, context=context)
    assert state["context"] == context


def test_initial_state_accepts_notify_email():
    state = initial_state("q", IDENTITY, notify_email="ops@example.com")
    assert state["notify_email"] == "ops@example.com"
    assert state["notify_status"] == ""


@pytest.mark.parametrize("question", ["", "   "])
def test_initial_state_rejects_empty_question(question):
    with pytest.raises(ValueError):
        initial_state(question, IDENTITY)


def test_initial_state_rejects_non_positive_workers_or_iterations():
    with pytest.raises(ValueError):
        initial_state("q", IDENTITY, max_workers=0)
    with pytest.raises(ValueError):
        initial_state("q", IDENTITY, max_iterations=0)
