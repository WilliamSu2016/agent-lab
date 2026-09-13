"""Tests for the Deployment experiment's production API layer
(``src/api``).

Exercises, against a real FastAPI ``TestClient`` (never mocked at the HTTP
layer) wired to a real temp-file SQLite checkpointer (never
``InMemorySaver``/``:memory:`` -- see ``src/durable/checkpointer.py``):

* the exact scenario the whole session has been building toward --
  Research A succeeds, Research B succeeds, Research C crashes; after
  resume, A/B are never re-executed and C completes -- now driven entirely
  through ``POST /runs`` / ``GET /runs/{id}/state`` / ``POST
  /runs/{id}/resume``;
* Requirement 1 (authentication): missing/invalid bearer token rejected;
* Requirement 2 (request validation): an empty question is rejected
  (``422``) before ever reaching the graph;
* Requirement 3 (rate limiting): the Nth request in a window is ``429``
  with a ``Retry-After`` header;
* tenant isolation: a different tenant's identity gets a ``404`` for
  someone else's run_id, not a ``403`` (see ``src/api/runs.py``'s
  docstring on why that distinction matters);
* Requirement 5 (error handling): every error response is the documented
  ``{request_id, error_code, message}`` shape;
* Requirement 6 (request id): every response (including errors) carries
  ``X-Request-ID``;
* Requirements 7/8 (health/ready): both endpoints respond, unauthenticated.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthConfig
from src.api.rate_limit import RateLimiter
from src.durable import recovery
from src.durable.checkpointer import sqlite_checkpointer
from src.durable.graph import build_durable_graph
from src.durable.state import TASK_IDS
from src.reliability.idempotency import InMemoryIdempotencyStore

VALID_TOKEN = "test-token-123"
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}"}


def _identity_headers(user_id: str, tenant_id: str = "acme") -> dict[str, str]:
    return {**AUTH_HEADERS, "X-User-Id": user_id, "X-Tenant-Id": tenant_id}


class _Harness:
    """Builds one app wired to a real temp-file checkpointer, one shared
    idempotency store, and side-effect call counters -- so a test can
    assert exactly how many times each research task's side effect ran,
    across a crash + resume, the same way ``tests/test_recovery.py`` does
    for the durable graph directly."""

    def __init__(self, tmp_path: Path, *, rate_limiter: RateLimiter | None = None):
        self.db_path = tmp_path / "api_test_checkpoints.sqlite3"
        self._ctx = sqlite_checkpointer(self.db_path)
        self.saver = self._ctx.__enter__()
        self.idempotency_store = InMemoryIdempotencyStore()
        self.call_counts: dict[str, int] = {task_id: 0 for task_id in TASK_IDS}
        self.crash_injector = recovery.CrashInjector()

        def make_side_effect(task_id: str):
            def _side_effect(*, task_id: str, content: str) -> dict[str, str]:
                self.call_counts[task_id] += 1
                return {"task_id": task_id, "findings": content}

            return _side_effect

        self.side_effects = {task_id: make_side_effect(task_id) for task_id in TASK_IDS}

        def graph_factory():
            return build_durable_graph(
                self.idempotency_store,
                side_effects=self.side_effects,
                crash_injector=self.crash_injector,
                checkpointer=self.saver,
            )

        self.app = create_app(
            graph_factory=graph_factory,
            auth_config=AuthConfig(valid_tokens=frozenset({VALID_TOKEN})),
            rate_limiter=rate_limiter,
            agent_version="test",
            environment="test",
        )

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


@pytest.fixture()
def harness(tmp_path):
    h = _Harness(tmp_path)
    yield h
    h.close()


def _wait_for_status(client: TestClient, run_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/runs/{run_id}", headers=_identity_headers("alice"))
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not leave 'running' within {timeout}s")


# ---------------------------------------------------------------------------
# Core scenario: crash -> checkpoint -> restart -> resume, over HTTP.
# ---------------------------------------------------------------------------


def test_crash_then_resume_does_not_rerun_completed_tasks(harness):
    harness.crash_injector.arm("C", "after")
    with TestClient(harness.app) as client:
        created = client.post(
            "/runs", json={"question": "why do systems crash?"}, headers=_identity_headers("alice")
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        interrupted = _wait_for_status(client, run_id)
        assert interrupted["status"] == "interrupted"
        assert interrupted["final_report"] is None

        state = client.get(f"/runs/{run_id}/state", headers=_identity_headers("alice")).json()
        assert state["pending_tasks"] == ["research_c"]
        assert harness.call_counts == {"A": 1, "B": 1, "C": 1}  # C's side effect DID run before the crash

        resumed = client.post(f"/runs/{run_id}/resume", headers=_identity_headers("alice"))
        assert resumed.status_code == 200
        completed = _wait_for_status(client, run_id)
        assert completed["status"] == "completed"
        assert completed["final_report"] is not None

    # Requirement: side effects with side effects are never re-executed.
    assert harness.call_counts == {"A": 1, "B": 1, "C": 1}


def test_run_without_any_crash_completes_directly(harness):
    with TestClient(harness.app) as client:
        created = client.post("/runs", json={"question": "plain run"}, headers=_identity_headers("alice"))
        run_id = created.json()["run_id"]
        completed = _wait_for_status(client, run_id)
        assert completed["status"] == "completed"
    assert harness.call_counts == {"A": 1, "B": 1, "C": 1}


def test_resume_on_a_completed_run_is_rejected(harness):
    with TestClient(harness.app) as client:
        created = client.post("/runs", json={"question": "plain run"}, headers=_identity_headers("alice"))
        run_id = created.json()["run_id"]
        _wait_for_status(client, run_id)

        response = client.post(f"/runs/{run_id}/resume", headers=_identity_headers("alice"))
        assert response.status_code == 409
        body = response.json()
        assert body["error_code"] == "invalid_run_state"
        assert "request_id" in body


# ---------------------------------------------------------------------------
# Requirement 1: authentication.
# ---------------------------------------------------------------------------


def test_missing_bearer_token_is_rejected(harness):
    with TestClient(harness.app) as client:
        response = client.post("/runs", json={"question": "x"}, headers={"X-User-Id": "alice"})
        assert response.status_code == 401
        assert "X-Request-ID" in response.headers


def test_invalid_bearer_token_is_rejected(harness):
    with TestClient(harness.app) as client:
        headers = {"Authorization": "Bearer wrong-token", "X-User-Id": "alice"}
        response = client.post("/runs", json={"question": "x"}, headers=headers)
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Requirement 2: request validation.
# ---------------------------------------------------------------------------


def test_empty_question_is_rejected_before_reaching_the_graph(harness):
    with TestClient(harness.app) as client:
        response = client.post("/runs", json={"question": ""}, headers=_identity_headers("alice"))
        assert response.status_code == 422
    assert harness.call_counts == {"A": 0, "B": 0, "C": 0}


def test_invalid_mode_is_rejected(harness):
    with TestClient(harness.app) as client:
        response = client.post(
            "/runs", json={"question": "x", "mode": "ultra"}, headers=_identity_headers("alice")
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Requirement 3: rate limiting.
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    h = _Harness(tmp_path, rate_limiter=RateLimiter(max_requests=2, window_seconds=60))
    try:
        with TestClient(h.app) as client:
            headers = _identity_headers("bob")
            r1 = client.post("/runs", json={"question": "one"}, headers=headers)
            r2 = client.post("/runs", json={"question": "two"}, headers=headers)
            r3 = client.post("/runs", json={"question": "three"}, headers=headers)
            assert r1.status_code == 202
            assert r2.status_code == 202
            assert r3.status_code == 429
            assert "Retry-After" in r3.headers
            assert r3.json()["error_code"] == "rate_limited"
    finally:
        h.close()


# ---------------------------------------------------------------------------
# Tenant isolation.
# ---------------------------------------------------------------------------


def test_another_tenant_gets_404_not_403(harness):
    with TestClient(harness.app) as client:
        created = client.post(
            "/runs", json={"question": "secret"}, headers=_identity_headers("alice", tenant_id="acme")
        )
        run_id = created.json()["run_id"]

        response = client.get(f"/runs/{run_id}", headers=_identity_headers("mallory", tenant_id="other-corp"))
        assert response.status_code == 404
        assert response.json()["error_code"] == "run_not_found"


def test_unknown_run_id_is_404(harness):
    with TestClient(harness.app) as client:
        response = client.get("/runs/does-not-exist", headers=_identity_headers("alice"))
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Requirement 6: request id on every response.
# ---------------------------------------------------------------------------


def test_every_response_carries_a_request_id_header(harness):
    with TestClient(harness.app) as client:
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

        error_response = client.get("/runs/nope", headers=_identity_headers("alice"))
        assert "X-Request-ID" in error_response.headers
        assert error_response.json()["request_id"] == error_response.headers["X-Request-ID"]


def test_incoming_request_id_is_propagated(harness):
    with TestClient(harness.app) as client:
        response = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
        assert response.headers["X-Request-ID"] == "caller-supplied-id"


# ---------------------------------------------------------------------------
# Requirements 7/8: health + readiness.
# ---------------------------------------------------------------------------


def test_health_never_requires_auth(harness):
    with TestClient(harness.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_ready_reports_checkpointer_health(harness):
    with TestClient(harness.app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["accepting_traffic"] is True


def test_ready_uses_injected_probe(tmp_path):
    def failing_probe():
        return {"checkpointer": False}

    h = _Harness(tmp_path)
    try:
        h.app = create_app(
            graph_factory=lambda: build_durable_graph(h.idempotency_store, checkpointer=h.saver),
            auth_config=AuthConfig(valid_tokens=frozenset({VALID_TOKEN})),
            readiness_probe=failing_probe,
        )
        with TestClient(h.app) as client:
            response = client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
    finally:
        h.close()
