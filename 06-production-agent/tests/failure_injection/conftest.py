"""Shared fixtures for the chaos-testing suite.

``make_settings`` builds a real, validated ``config.Settings`` with a tiny
retry/backoff/timeout budget so every test that exercises the *real*
``retry_call``/``run_with_timeout`` pipeline (rather than a reimplementation
of it) runs in milliseconds instead of waiting through production-sized
backoff delays.
"""

from __future__ import annotations

import httpx2
import openai
import pytest

from config.settings import AgentLimits, RetryPolicy, Settings


@pytest.fixture
def make_settings():
    def _make(*, max_retries: int = 2, worker_timeout_seconds: float = 0.2) -> Settings:
        return Settings(
            environment="test",
            api_key="test-key",
            model_name="test-model",
            base_url="https://example.invalid/v1",
            limits=AgentLimits(
                max_workers=6,
                max_iterations=3,
                worker_timeout_seconds=worker_timeout_seconds,
                run_timeout_seconds=30.0,
            ),
            retry=RetryPolicy(
                max_retries=max_retries,
                backoff_base_seconds=0.001,
                backoff_max_seconds=0.005,
            ),
        )

    return _make


def openai_request() -> httpx2.Request:
    return httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


def openai_status_error(status: int, message: str = "boom") -> openai.APIStatusError:
    """Build a real ``openai.APIStatusError`` the same way
    ``tests/test_errors.py`` does, so classification (``classify_openai_exception``)
    behaves identically to production."""
    req = openai_request()
    resp = httpx2.Response(status, request=req, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=resp, body=None)
