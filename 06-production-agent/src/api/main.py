"""Process entry point: ``python -m src.api.main`` (or the Dockerfile's
``CMD``/``uvicorn`` invocation -- see ``deployment/Dockerfile``).

Wires the API layer (``src.api.app.create_app``) to real, production-grade
dependencies:

* **Production checkpointer** -- ``src.durable.checkpointer.sqlite_checkpointer``,
  opened once for the whole process lifetime against a path that must be a
  mounted persistent volume in Production (``deployment/docker-compose.yml``'s
  ``checkpoints`` volume) -- see docs/08-DEPLOYMENT.md and
  docs/03-DURABLE-EXECUTION.md for why an ``InMemorySaver`` is never used
  here.
* **Environment variables** -- read once at start-up via
  ``config.settings.load_settings`` (fails loudly and immediately, before
  ``uvicorn`` ever binds a socket, if a required secret/setting is
  missing -- never a confusing failure on the first real request) plus the
  API-specific variables this module itself reads directly (below).
* **Graceful shutdown** -- ``uvicorn``'s own SIGTERM handling drives
  ``create_app``'s lifespan shutdown hook (``RunRegistry.drain``); this
  module additionally passes ``--timeout-graceful-shutdown`` so Docker's
  ``docker stop`` (SIGTERM, then SIGKILL after a grace period) has time to
  let in-flight runs finish recording their outcome before the process
  actually exits.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from config.settings import ConfigurationError, load_settings
from src.api.app import create_app
from src.api.auth import AuthConfig
from src.api.rate_limit import RateLimiter
from src.durable.checkpointer import DEFAULT_CHECKPOINT_DB_PATH
from src.durable.graph import build_durable_graph
from src.reliability.idempotency import InMemoryIdempotencyStore

AGENT_VERSION_ENV = "AGENT_VERSION"
API_HOST_ENV = "API_HOST"
API_PORT_ENV = "API_PORT"
API_GRACEFUL_TIMEOUT_ENV = "API_GRACEFUL_SHUTDOWN_SECONDS"
CHECKPOINT_DB_PATH_ENV = "CHECKPOINT_DB_PATH"
RATE_LIMIT_MAX_REQUESTS_ENV = "RATE_LIMIT_MAX_REQUESTS"
RATE_LIMIT_WINDOW_SECONDS_ENV = "RATE_LIMIT_WINDOW_SECONDS"


def build_app_and_connection():
    """Builds the FastAPI app plus the raw sqlite3 connection backing its
    checkpointer, so the caller (``main()``) can close the connection on
    shutdown. Split out from ``main()`` so ``tests/test_api.py`` can build
    the exact same wiring against a temp-file database instead of the real
    ``.env``-configured one."""
    load_dotenv()
    settings = load_settings()

    db_path = Path(os.environ.get(CHECKPOINT_DB_PATH_ENV, str(DEFAULT_CHECKPOINT_DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)

    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(connection)
    saver.setup()

    idempotency_store = InMemoryIdempotencyStore()

    def graph_factory():
        # A fresh compiled graph object per call, sharing the one durable
        # checkpointer connection and the one idempotency store for the
        # whole process -- see src/durable/recovery.py's docstring on why
        # "restart" means a fresh graph object, never a fresh checkpointer.
        return build_durable_graph(idempotency_store, checkpointer=saver)

    def readiness_probe() -> dict[str, bool]:
        try:
            connection.execute("SELECT 1").fetchone()
            checkpointer_ok = True
        except sqlite3.Error:
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
        rate_limiter=rate_limiter,
        readiness_probe=readiness_probe,
        agent_version=os.environ.get(AGENT_VERSION_ENV, "dev"),
        environment=settings.environment,
        shutdown_drain_timeout_seconds=float(os.environ.get(API_GRACEFUL_TIMEOUT_ENV, "30")),
    )
    return app, connection


def main() -> None:
    try:
        app, connection = build_app_and_connection()
    except ConfigurationError as exc:
        # Fail loudly before uvicorn ever binds a socket -- never a process
        # that "starts" but 500s on every request because config was bad.
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
