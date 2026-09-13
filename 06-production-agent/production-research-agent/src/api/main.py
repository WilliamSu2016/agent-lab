"""Process entry point: ``python -m src.api.main`` (or the Dockerfile's
CMD/uvicorn invocation -- see ``deployment/Dockerfile``).

Wires the API layer to the **real** Production Research graph
(``src.graph.graph.build_graph``, Planner/Researcher/Synthesizer/Reviewer
with real LLM calls) bound to a production SQLite checkpointer -- this is
the change that closes the Final Review's single most important finding:
the previous experiment's ``src/api/main.py`` served a placeholder,
non-LLM demo graph instead of the real Multi-Agent Research Agent.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from src.api.app import create_app
from src.api.auth import AuthConfig
from src.api.rate_limit import RateLimiter
from src.config import ConfigurationError, load_settings
from src.graph.checkpointer import DEFAULT_CHECKPOINT_DB_PATH, open_sqlite_checkpointer
from src.graph.graph import build_graph
from src.observability.metrics import MetricsRegistry
from src.observability.tracing import Tracer
from src.agents.llm import build_openai_text_llm_call
from src.security.authorization import ApprovalStore
from src.security.tool_policy import build_default_registry
from src.tools.search import build_search_tool

AGENT_VERSION_ENV = "AGENT_VERSION"
API_HOST_ENV = "API_HOST"
API_PORT_ENV = "API_PORT"
API_GRACEFUL_TIMEOUT_ENV = "API_GRACEFUL_SHUTDOWN_SECONDS"
CHECKPOINT_DB_PATH_ENV = "CHECKPOINT_DB_PATH"
RATE_LIMIT_MAX_REQUESTS_ENV = "RATE_LIMIT_MAX_REQUESTS"
RATE_LIMIT_WINDOW_SECONDS_ENV = "RATE_LIMIT_WINDOW_SECONDS"


def build_app_and_connection():
    """Builds the FastAPI app plus the raw sqlite3 connection backing its
    checkpointer, so the caller can close the connection on shutdown.
    Split out from ``main()`` so tests can build the exact same wiring
    against a temp-file database instead of the real ``.env``-configured
    one."""
    load_dotenv()
    settings = load_settings()

    db_path = Path(os.environ.get(CHECKPOINT_DB_PATH_ENV, str(DEFAULT_CHECKPOINT_DB_PATH)))
    saver, connection = open_sqlite_checkpointer(db_path)

    text_llm_call = build_openai_text_llm_call(settings)
    tracer = Tracer()
    metrics = MetricsRegistry()
    approval_store = ApprovalStore()
    tool_registry = build_default_registry()
    search_tool_fn = build_search_tool()

    def graph_factory():
        # A fresh compiled graph object per call, sharing the one durable
        # checkpointer connection, tracer, metrics registry, and approval
        # store for the whole process -- see src/graph/recovery.py's
        # docstring on why "restart" means a fresh graph object, never a
        # fresh checkpointer.
        return build_graph(
            text_llm_call,
            text_llm_call,
            text_llm_call,
            text_llm_call,
            checkpointer=saver,
            tracer=tracer,
            tool_registry=tool_registry,
            approval_store=approval_store,
            search_tool_fn=search_tool_fn,
        )

    def readiness_probe() -> dict[str, bool]:
        try:
            connection.execute("SELECT 1").fetchone()
            checkpointer_ok = True
        except Exception:  # noqa: BLE001
            checkpointer_ok = False
        return {"checkpointer": checkpointer_ok, "configuration": True}

    auth_config = AuthConfig.from_env()
    rate_limiter = RateLimiter(
        max_requests=int(os.environ.get(RATE_LIMIT_MAX_REQUESTS_ENV, "60")),
        window_seconds=float(os.environ.get(RATE_LIMIT_WINDOW_SECONDS_ENV, "60")),
    )

    app = create_app(
        graph_factory=graph_factory,
        auth_config=auth_config,
        approval_store=approval_store,
        rate_limiter=rate_limiter,
        readiness_probe=readiness_probe,
        tracer=tracer,
        metrics=metrics,
        agent_version=os.environ.get(AGENT_VERSION_ENV, "dev"),
        environment=settings.environment,
        shutdown_drain_timeout_seconds=float(os.environ.get(API_GRACEFUL_TIMEOUT_ENV, "30")),
    )
    return app, connection


def main() -> None:
    try:
        app, connection = build_app_and_connection()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error, refusing to start: {exc}") from exc

    host = os.environ.get(API_HOST_ENV, "0.0.0.0")
    port = int(os.environ.get(API_PORT_ENV, "8000"))
    graceful_timeout = float(os.environ.get(API_GRACEFUL_TIMEOUT_ENV, "30"))
    try:
        uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=int(graceful_timeout))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
