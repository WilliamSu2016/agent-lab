"""Shared fixtures for the Failure Injection / Chaos Testing suite.

``make_settings`` builds a real, validated ``src.config.settings.Settings``
with a tiny retry/backoff/timeout budget so tests that exercise the real
``retry_call``/``run_with_timeout`` pipeline run in milliseconds instead
of production-sized backoff delays.
"""

from __future__ import annotations

import json

import httpx2
import openai
import pytest

from src.config.settings import AgentLimits, RetryPolicy, Settings


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
    """Build a real ``openai.APIStatusError`` the same way production code
    encounters it, so classification (``src.reliability.retry``) behaves
    identically to a real OpenAI outage."""
    req = openai_request()
    resp = httpx2.Response(status, request=req, json={"error": {"message": message}})
    return openai.APIStatusError(message, response=resp, body=None)


def stub_llm_factory(*, plan_aspects=None):
    """Builds a small family of deterministic stub ``TextLLMCall``s (one
    per node role) usable for graph-level chaos tests that must not
    depend on a real OpenAI key. ``plan_aspects`` (a list[str]) controls
    what the planner "decomposes" the question into."""
    plan_aspects = plan_aspects or ["aspect one", "aspect two"]

    def planner_llm(system: str, user: str) -> str:
        return json.dumps(plan_aspects)

    def researcher_llm(system: str, user: str) -> str:
        return f"Finding for: {user[:60]}"

    def synthesizer_llm(system: str, user: str) -> str:
        return "Synthesized report."

    def reviewer_llm(system: str, user: str) -> str:
        return "APPROVE\nGood enough."

    return planner_llm, researcher_llm, synthesizer_llm, reviewer_llm
