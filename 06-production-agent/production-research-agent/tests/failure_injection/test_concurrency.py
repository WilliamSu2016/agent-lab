"""Failure #11: concurrent requests.

Exercises real thread-level concurrency (``concurrent.futures.ThreadPoolExecutor``,
never simulated sequentially) against the production primitives:

    * ``src.reliability.idempotency.InMemoryIdempotencyStore`` /
      ``idempotent()`` -- N concurrent callers with the *same* idempotency
      key must result in exactly one real side-effect execution.
    * ``src.api.rate_limit.RateLimiter`` -- concurrent callers sharing one
      key must never be let through more than ``max_requests`` times in
      the window.
    * the full HTTP API (``src.api.app``) -- concurrent ``POST /runs``
      calls from different identities must never leak state across
      requests/threads (tenant isolation holds even under real
      concurrency, not just sequential calls).

Expected behaviour: correctness under concurrency is preserved by each
primitive's own lock/atomic-claim, and a caller that loses a race gets an
explicit, structured signal (``IdempotencyInProgressError``,
``RateLimitExceededError``) rather than silent corruption or a hang.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import AuthConfig
from src.api.rate_limit import RateLimitExceededError, RateLimiter
from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import build_graph
from src.reliability.idempotency import (
    IdempotencyInProgressError,
    InMemoryIdempotencyStore,
    idempotent,
)

VALID_TOKEN = "concurrency-test-token"


class TestConcurrentIdempotentCalls:
    """N threads racing the same idempotency key must produce exactly one
    real side-effect execution; every other thread either gets the cached
    result or a structured "in progress" signal -- never a second
    execution, never a hang, never a crash."""

    def test_only_one_of_many_concurrent_callers_actually_runs_the_side_effect(self):
        store = InMemoryIdempotencyStore()
        call_count = {"n": 0}
        call_lock = threading.Lock()
        start_barrier = threading.Barrier(20)

        def slow_side_effect(*, idempotency_key: str) -> str:
            with call_lock:
                call_count["n"] += 1
            time.sleep(0.05)  # widen the race window so threads genuinely overlap
            return "the one true result"

        wrapped = idempotent(
            slow_side_effect,
            store=store,
            key_fn=lambda **kwargs: kwargs["idempotency_key"],
            fingerprint_fn=lambda **kwargs: "fp",
        )

        results = []
        errors = []

        def worker():
            start_barrier.wait()
            try:
                results.append(wrapped(idempotency_key="shared-key"))
            except IdempotencyInProgressError as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(worker) for _ in range(20)]
            for future in as_completed(futures):
                future.result()

        assert call_count["n"] == 1
        assert len(results) + len(errors) == 20
        assert all(r == "the one true result" for r in results)
        assert len(results) >= 1


class TestConcurrentRateLimiting:
    """The rate limiter must enforce its cap correctly even when every
    ``check()`` call races every other one for the same key."""

    def test_at_most_max_requests_succeed_under_concurrent_load(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)
        start_barrier = threading.Barrier(30)
        allowed = []
        rejected = []
        lock = threading.Lock()

        def worker():
            start_barrier.wait()
            try:
                limiter.check("shared-caller")
            except RateLimitExceededError:
                with lock:
                    rejected.append(1)
            else:
                with lock:
                    allowed.append(1)

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = [pool.submit(worker) for _ in range(30)]
            for future in as_completed(futures):
                future.result()

        assert len(allowed) == 10
        assert len(rejected) == 20

    def test_different_keys_are_independent_under_concurrent_load(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        start_barrier = threading.Barrier(10)
        results: dict[str, int] = {"user-a": 0, "user-b": 0}
        lock = threading.Lock()

        def worker(user_id: str):
            start_barrier.wait()
            try:
                limiter.check(user_id)
            except RateLimitExceededError:
                pass
            else:
                with lock:
                    results[user_id] += 1

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker, "user-a" if i % 2 == 0 else "user-b") for i in range(10)]
            for future in as_completed(futures):
                future.result()

        assert results == {"user-a": 5, "user-b": 5}


def _stub_llm(system: str, user: str) -> str:
    if "研究方面" in system or "aspect" in system.lower():
        return json.dumps(["aspect one"])
    return "stub response"


class _ApiHarness:
    """Minimal API harness for exercising real concurrent HTTP requests
    against the real Production Research graph (bounded to ``mode="fast"``
    and a stub LLM so these tests run in milliseconds without a real
    OpenAI key)."""

    def __init__(self, tmp_path: Path):
        self.db_path = tmp_path / "concurrency_api.sqlite3"
        self._ctx = sqlite_checkpointer(self.db_path)
        self.saver = self._ctx.__enter__()

        def graph_factory():
            return build_graph(_stub_llm, _stub_llm, _stub_llm, _stub_llm, checkpointer=self.saver)

        self.app = create_app(
            graph_factory=graph_factory,
            auth_config=AuthConfig(valid_tokens=frozenset({VALID_TOKEN})),
            agent_version="test",
            environment="test",
        )

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


@pytest.fixture()
def api_harness(tmp_path):
    h = _ApiHarness(tmp_path)
    yield h
    h.close()


def _headers(user_id: str, tenant_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {VALID_TOKEN}", "X-User-Id": user_id, "X-Tenant-Id": tenant_id}


class TestConcurrentHTTPRequestsPreserveTenantIsolation:
    """Multiple tenants issuing concurrent ``POST /runs`` calls from
    different threads must never see each other's runs -- tenant
    isolation must hold under genuine concurrency, not only sequential
    requests."""

    def test_concurrent_requests_from_different_tenants_never_cross_contaminate(self, api_harness):
        with TestClient(api_harness.app) as client:
            created_run_ids: dict[str, str] = {}
            lock = threading.Lock()
            start_barrier = threading.Barrier(6)

            def create_for_tenant(tenant_id: str, index: int):
                start_barrier.wait()
                response = client.post(
                    "/runs",
                    json={"question": f"question from {tenant_id} #{index}", "mode": "fast"},
                    headers=_headers(f"user-{tenant_id}-{index}", tenant_id),
                )
                assert response.status_code == 202
                with lock:
                    created_run_ids[f"{tenant_id}-{index}"] = response.json()["run_id"]

            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = [
                    pool.submit(create_for_tenant, tenant_id, index)
                    for tenant_id in ("tenant-x", "tenant-y", "tenant-z")
                    for index in range(2)
                ]
                for future in as_completed(futures):
                    future.result()

            assert len(created_run_ids) == 6
            assert len(set(created_run_ids.values())) == 6  # every run_id unique

            for key, run_id in created_run_ids.items():
                owning_tenant = key.rsplit("-", 1)[0]
                own_lookup = client.get(f"/runs/{run_id}", headers=_headers("checker", owning_tenant))
                assert own_lookup.status_code == 200

                for other_tenant in ("tenant-x", "tenant-y", "tenant-z"):
                    if other_tenant == owning_tenant:
                        continue
                    cross_lookup = client.get(f"/runs/{run_id}", headers=_headers("checker", other_tenant))
                    assert cross_lookup.status_code == 404
