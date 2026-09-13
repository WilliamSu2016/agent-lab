"""FastAPI app factory: routes, middleware, lifespan.

Wires together every cross-cutting requirement the Deployment experiment
lists, on top of the domain modules in this package:

    1. Authentication         -> src.api.auth
    2. Request validation     -> src.api.schemas (Pydantic)
    3. Rate limiting          -> src.api.rate_limit
    4. Timeout                -> src.api.runs (per-run, via src.cost.policy)
    5. Error handling         -> src.api.errors
    6. Request ID             -> RequestContextMiddleware (this module)
    7. Health check           -> GET /health
    8. Readiness check        -> GET /ready
    9. Graceful shutdown      -> lifespan shutdown handler (this module)

``create_app`` takes every external dependency as an explicit argument
(graph factory, auth config, rate limiter, checkpointer readiness probe)
rather than importing/constructing them at module import time -- the same
dependency-injection convention used throughout this project (e.g.
``TextLLMCall`` injection in ``src/multi_agent_research``) so
``tests/test_api.py`` can build a fully isolated app (fake graph, no real
OpenAI key, temp-file checkpointer) without monkeypatching anything.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from src.api.auth import AuthConfig, get_identity, make_verify_bearer_token
from src.api.errors import EXCEPTION_HANDLERS
from src.api.rate_limit import RateLimiter
from src.api.runs import GraphFactory, RunRegistry
from src.api.schemas import (
    HealthResponse,
    HistoryEntryResponse,
    ReadyResponse,
    RunCreateRequest,
    RunStateResponse,
    RunSummary,
)
from src.observability.logging import configure_json_logging, log_event
from src.observability.tracing import bind_execution_context, new_execution_context
from src.security.authorization import Identity

_logger = configure_json_logging(logger_name="agent.api")


def _record_to_summary(record) -> RunSummary:  # noqa: ANN001 - RunRecord, avoided import cycle noise
    return RunSummary(
        run_id=record.run_id,
        thread_id=record.thread_id,
        status=record.status,  # type: ignore[arg-type]
        mode=record.mode,  # type: ignore[arg-type]
        user_id=record.user_id,
        tenant_id=record.tenant_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        final_report=record.final_report,
        error=record.error,
    )


def create_app(
    *,
    graph_factory: GraphFactory,
    auth_config: AuthConfig,
    rate_limiter: Optional[RateLimiter] = None,
    readiness_probe: Optional[Callable[[], dict[str, bool]]] = None,
    agent_version: str = "unknown",
    environment: str = "development",
    shutdown_drain_timeout_seconds: float = 30.0,
) -> FastAPI:
    """Builds one fully configured FastAPI application.

    ``readiness_probe`` returns a ``{check_name: healthy}`` mapping (e.g.
    "checkpointer" -> can we open/query it) -- ``GET /ready`` is ``503``
    if any value is ``False``. Defaults to "always ready" only for tests
    that do not care about readiness semantics; ``main.py`` always passes
    a real probe backed by the production checkpointer.
    """
    registry = RunRegistry(graph_factory)
    rate_limiter = rate_limiter or RateLimiter()
    verify_bearer_token = make_verify_bearer_token(auth_config)
    shutting_down = {"value": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_event(_logger, logging.INFO, "api_startup", environment=environment, agent_version=agent_version)
        yield
        # Requirement 9: graceful shutdown. Readiness must flip to
        # "not_ready" the instant shutdown begins (see /ready below) so an
        # orchestrator stops routing new traffic here *before* we start
        # waiting for in-flight runs to finish -- never the other way
        # around, or new requests could race the drain.
        shutting_down["value"] = True
        log_event(_logger, logging.INFO, "api_shutdown_draining", timeout_seconds=shutdown_drain_timeout_seconds)
        await registry.drain(shutdown_drain_timeout_seconds)
        log_event(_logger, logging.INFO, "api_shutdown_complete")

    app = FastAPI(title="Production Agent API", version=agent_version, lifespan=lifespan)
    for exc_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_type, handler)

    # ------------------------------------------------------------------
    # Requirement 6: request id + basic access logging, wraps every route
    # (including /health and /ready, and even a 404 for an unknown path).
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        context = new_execution_context(
            user_id=request.headers.get("X-User-Id", "anonymous"),
            session_id=request.headers.get("X-Session-Id") or uuid.uuid4().hex,
            agent_version=agent_version,
            environment=environment,
            request_id=request_id,
        )
        started = time.time()
        with bind_execution_context(context):
            response = await call_next(request)
        duration_ms = (time.time() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        log_event(
            _logger,
            logging.INFO,
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    # ------------------------------------------------------------------
    # Requirement 3: rate limiting -- one dependency shared by every
    # authenticated route, keyed by the caller's identity.
    # ------------------------------------------------------------------
    def enforce_rate_limit(identity: Identity = Depends(get_identity)) -> Identity:
        rate_limiter.check(identity.user_id)
        return identity

    protected = [Depends(verify_bearer_token)]

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        # Requirement 7: liveness only -- must always answer fast and
        # never depend on the checkpointer/downstream services; an
        # orchestrator restarts the process on a failing liveness probe,
        # which would be the wrong response to "the database is briefly
        # unreachable" (that is what /ready is for).
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(response: Response) -> ReadyResponse:
        # Requirement 8: readiness -- checks real dependencies, and is the
        # first thing to fail during graceful shutdown (Requirement 9).
        checks: dict[str, bool] = {"accepting_traffic": not shutting_down["value"]}
        if readiness_probe is not None:
            checks.update(readiness_probe())
        healthy = all(checks.values())
        response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="ready" if healthy else "not_ready", checks=checks)

    @app.post("/runs", response_model=RunSummary, status_code=status.HTTP_202_ACCEPTED, dependencies=protected)
    async def create_run(body: RunCreateRequest, identity: Identity = Depends(enforce_rate_limit)) -> RunSummary:
        record = await registry.create_run(body.question, identity, body.mode)
        return _record_to_summary(record)

    @app.get("/runs/{run_id}", response_model=RunSummary, dependencies=protected)
    async def get_run(run_id: str, identity: Identity = Depends(enforce_rate_limit)) -> RunSummary:
        record = await registry.get_run(run_id, identity)
        return _record_to_summary(record)

    @app.get("/runs/{run_id}/state", response_model=RunStateResponse, dependencies=protected)
    async def get_run_state(run_id: str, identity: Identity = Depends(enforce_rate_limit)) -> RunStateResponse:
        record, history, pending = await registry.get_state(run_id, identity)
        return RunStateResponse(
            run_id=record.run_id,
            thread_id=record.thread_id,
            status=record.status,  # type: ignore[arg-type]
            pending_tasks=pending,
            history=[
                HistoryEntryResponse(
                    step=entry.step,
                    next_tasks=entry.next_tasks,
                    trace_so_far=entry.trace_so_far,
                    results_so_far=entry.results_so_far,
                )
                for entry in history
            ],
        )

    @app.post("/runs/{run_id}/resume", response_model=RunSummary, dependencies=protected)
    async def resume_run(run_id: str, identity: Identity = Depends(enforce_rate_limit)) -> RunSummary:
        record = await registry.resume_run(run_id, identity)
        return _record_to_summary(record)

    app.state.registry = registry
    return app


__all__ = ["create_app"]
