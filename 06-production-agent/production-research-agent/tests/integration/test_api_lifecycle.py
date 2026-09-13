"""End-to-end HTTP integration test: ``POST /runs -> GET /runs/{id} -> GET
/runs/{id}/state -> POST /runs/{id}/resume``, plus ``/health``/``/ready``,
against a real FastAPI app wired to the real graph (stub LLMs, real
guardrails, real SQLite checkpointer)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthConfig
from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import build_graph

HEADERS = {
    "Authorization": "Bearer test-token",
    "X-User-Id": "alice",
    "X-Tenant-Id": "acme",
    "X-User-Roles": "researcher",
}


def _planner_llm(system: str, user: str) -> str:
    return json.dumps(["Research A"])


def _researcher_llm(system: str, user: str) -> str:
    return "Finding for A."


def _synth_llm(system: str, user: str) -> str:
    return "Synthesized report."


def _review_llm(system: str, user: str) -> str:
    return "APPROVE\nComplete."


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "api_checkpoints.sqlite3"
    with sqlite_checkpointer(db_path) as checkpointer:

        def graph_factory():
            return build_graph(_planner_llm, _researcher_llm, _synth_llm, _review_llm, checkpointer=checkpointer)

        app = create_app(
            graph_factory=graph_factory,
            auth_config=AuthConfig(valid_tokens=frozenset({"test-token"})),
            agent_version="test-1.0.0",
            environment="test",
        )
        with TestClient(app) as test_client:
            yield test_client


def _wait_until_terminal(client: TestClient, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/runs/{run_id}", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] not in ("running",):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached a terminal state")


class TestHealthAndReadiness:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_endpoint_reports_ready_while_serving(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["accepting_traffic"] is True


class TestAuthenticationAndValidation:
    def test_missing_bearer_token_is_rejected(self, client):
        resp = client.post("/runs", json={"question": "hi", "mode": "balanced"}, headers={"X-User-Id": "alice"})
        assert resp.status_code == 401

    def test_invalid_request_body_is_rejected(self, client):
        headers = dict(HEADERS)
        resp = client.post("/runs", json={"mode": "balanced"}, headers=headers)  # missing required "question"
        assert resp.status_code == 422


class TestRunLifecycle:
    def test_full_run_lifecycle_completes_successfully(self, client):
        create_resp = client.post("/runs", json={"question": "What is LangGraph?", "mode": "balanced"}, headers=HEADERS)
        assert create_resp.status_code == 202
        run_id = create_resp.json()["run_id"]
        assert create_resp.json()["status"] == "running"

        final = _wait_until_terminal(client, run_id)
        assert final["status"] == "completed"
        assert "Synthesized report" in final["final_answer"]

        state_resp = client.get(f"/runs/{run_id}/state", headers=HEADERS)
        assert state_resp.status_code == 200
        state_body = state_resp.json()
        assert state_body["pending_tasks"] == []
        assert len(state_body["history"]) >= 1

    def test_a_different_tenant_cannot_see_another_tenants_run(self, client):
        create_resp = client.post("/runs", json={"question": "q", "mode": "balanced"}, headers=HEADERS)
        run_id = create_resp.json()["run_id"]
        _wait_until_terminal(client, run_id)

        other_tenant_headers = {**HEADERS, "X-Tenant-Id": "other-co"}
        resp = client.get(f"/runs/{run_id}", headers=other_tenant_headers)
        assert resp.status_code == 404

    def test_resume_on_an_already_completed_run_is_rejected(self, client):
        create_resp = client.post("/runs", json={"question": "q", "mode": "balanced"}, headers=HEADERS)
        run_id = create_resp.json()["run_id"]
        _wait_until_terminal(client, run_id)

        resume_resp = client.post(f"/runs/{run_id}/resume", headers=HEADERS)
        assert resume_resp.status_code >= 400

    def test_unknown_run_id_is_a_404(self, client):
        resp = client.get("/runs/does-not-exist", headers=HEADERS)
        assert resp.status_code == 404
