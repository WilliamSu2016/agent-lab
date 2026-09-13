"""``search_web`` -- the one LOW-risk (read-only) tool the Researcher agent
calls before every LLM synthesis step.

This closes a gap the Final Review flagged: the previous experiment's
"Multi-Agent Research" pipeline never actually called a *tool* -- every
"research" step was a bare LLM call with no external action, so
``src/security/tool_policy.py``'s risk tiers had nothing real to classify.
Here, every researcher invocation goes through
:func:`search_web` -- a synchronous function with a small, injectable
backend (so tests never hit the real network) -- via
``src.security.guardrails.secure_tool_call`` (Layer 2 authorization/tenant
isolation + Layer 3 argument validation/output sanitization), and every
call is wrapped in an observability TOOL span (see
``src/graph/graph.py``).

Real deployments should swap :data:`DEFAULT_BACKEND` for a real search API
client; the function signature (``query: str -> str``) is the only
contract callers depend on.
"""

from __future__ import annotations

from typing import Callable

SearchBackend = Callable[[str], str]


def _reference_backend(query: str) -> str:
    """Deterministic, offline stand-in for a real search API. Returns a
    short, clearly-labeled synthetic snippet so tests and local runs never
    depend on network access or an API key just to exercise the tool
    plumbing. Real deployments inject a real backend via
    :func:`build_search_tool`."""
    return (
        f"[reference_backend] Top result for query={query!r}: "
        "no live web search is configured; this is a placeholder search "
        "result used only when no real search backend is injected."
    )


def build_search_tool(backend: SearchBackend | None = None) -> SearchBackend:
    """Returns a ``search_web(query) -> str`` callable bound to ``backend``
    (defaults to the offline reference backend). This is the function
    registered as the ``search_web`` tool in
    ``src.security.tool_policy.build_default_registry`` (LOW risk)."""
    resolved = backend or _reference_backend

    def search_web(query: str) -> str:
        return resolved(query)

    return search_web


__all__ = ["SearchBackend", "build_search_tool"]
