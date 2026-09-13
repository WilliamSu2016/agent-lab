"""Module-level graph entrypoint for the LangGraph CLI's local development
server (``langgraph dev``, configured by ``deployment/langgraph.json``).

Requirement: "Development: local LangGraph server." The LangGraph CLI
imports a plain module-level variable pointing at a graph -- it cannot
call a factory function that takes arguments the way
``src.durable.graph.build_durable_graph`` does (it needs an
``IdempotencyStore`` and, in ``src.api.main``, a real SQLite connection).
This module exists purely to give the CLI something importable: a
compiled graph using a fresh, process-local in-memory idempotency store
and **no explicit checkpointer** -- ``langgraph dev`` supplies its own
development-mode persistence layer for whatever graph it loads, so this
module must not open a competing SQLite connection itself.

This is intentionally the *development* wiring only. The production
FastAPI app (``src/api/main.py``) does not import this module -- it builds
its own graph via ``build_durable_graph`` bound to the real,
production-configured SQLite checkpointer (or, for a genuinely
multi-replica deployment, a Postgres-backed one -- see
``src/durable/checkpointer.py``). See ``docs/08-DEPLOYMENT.md`` for the
full Development-vs-Production comparison.
"""

from __future__ import annotations

from src.durable.graph import build_durable_graph
from src.reliability.idempotency import InMemoryIdempotencyStore

graph = build_durable_graph(InMemoryIdempotencyStore())

__all__ = ["graph"]
