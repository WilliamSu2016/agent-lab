"""FastAPI app factory: routes, middleware, lifespan.

Wires: 1) Authentication, 2) Request validation, 3) Rate limiting,
4) Timeout, 5) Error handling, 6) Request ID, 7) Health check,
8) Readiness check, 9) Graceful shutdown -- plus the HIGH-risk tool
approval endpoints (``/approvals/*``) that the Final Review flagged as
implemented (``src.security.authorization.ApprovalStore``) but
unreachable from any route in the previous experiment.
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
    ApprovalDecisionRequest,
    ApprovalResponse,
    HealthResponse,
    HistoryEntryResponse,
    ReadyResponse,
    RunCreateRequest,
    RunStateResponse,
    RunSummary,
)
from src.observability.logging import configure_json_logging, log_event
from src.observability.metrics import MetricsRegistry
from src.observability.tracing import Tracer, bind_execution_context, new_execution_context
from src.security.authorization import ApprovalStore, Identity

_logger = configure_json_logging(logger_name="agent.api")


def _record_to_summary(record) -> RunSummary:  # noqa: ANN001
    return RunSummary(
        run_id=record.run_id,
        thread_id=record.thread_id,
        status=record.status,  # type: ignore[arg-type]
        mode=record.mode,  # type: ignore[arg-type]
        user_id=record.user_id,
        tenant_id=record.tenant_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        final_answer=record.final_answer,
        error=record.error,
    )


def create_app(
    *,
    graph_factory: GraphFactory,
    auth_config: AuthConfig,
    approval_store: Optional[ApprovalStore] = None,
    rate_limiter: Optional[RateLimiter] = None,
    readiness_probe: Optional[Callable[[], dict[str, bool]]] = None,
    tracer: Optional[Tracer] = None,
    metrics: Optional[MetricsRegistry] = None,
    agent_version: str = "unknown",
    environment: str = "development",
    shutdown_drain_timeout_seconds: float = 30.0,
) -> FastAPI:
    tracer = tracer or Tracer()
    metrics = metrics or MetricsRegistry()
    approval_store = approval_store or ApprovalStore()
    registry = RunRegistry(
        graph_factory, tracer=tracer, metrics=metrics, agent_version=agent_version, environment=environment
    )
    rate_limiter = rate_limiter or RateLimiter()
    verify_bearer_token = make_verify_bearer_token(auth_config)
    shutting_down = {"value": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_event(_logger, logging.INFO, "api_startup", environment=environment, agent_version=agent_version)
        yield
        shutting_down["value"] = True
        log_event(_logger, logging.INFO, "api_shutdown_draining", timeout_seconds=shutdown_drain_timeout_seconds)
        await registry.drain(shutdown_drain_timeout_seconds)
        log_event(_logger, logging.INFO, "api_shutdown_complete")

    app = FastAPI(title="Production Research Agent API", version=agent_version, lifespan=lifespan)
    for exc_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_type, handler)

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

    def enforce_rate_limit(identity: Identity = Depends(get_identity)) -> Identity:
        rate_limiter.check(identity.user_id)
        return identity

    protected = [Depends(verify_bearer_token)]

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(response: Response) -> ReadyResponse:
        checks: dict[str, bool] = {"accepting_traffic": not shutting_down["value"]}
        if readiness_probe is not None:
            checks.update(readiness_probe())
        healthy = all(checks.values())
        response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="ready" if healthy else "not_ready", checks=checks)

    @app.get("/metrics", tags=["ops"], dependencies=protected)
    async def get_metrics() -> dict:
        # Requirement (Observability): dashboard-ready summary of every
        # metric this process has recorded so far.
        return metrics.summary()

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
                    iteration=entry.iteration,
                    worker_result_count=entry.worker_result_count,
                )
                for entry in history
            ],
        )

    @app.post("/runs/{run_id}/resume", response_model=RunSummary, dependencies=protected)
    async def resume_run(run_id: str, identity: Identity = Depends(enforce_rate_limit)) -> RunSummary:
        record = await registry.resume_run(run_id, identity)
        return _record_to_summary(record)

    # ------------------------------------------------------------------
    # HIGH-risk tool approval workflow (Security experiment Requirement 7:
    # "Agent -> approval -> Tool"). Closes the Final Review's finding that
    # ApprovalStore existed but had no reachable API surface.
    # ------------------------------------------------------------------
    @app.get("/approvals/{request_id}", response_model=ApprovalResponse, dependencies=protected)
    async def get_approval(request_id: str) -> ApprovalResponse:
        request = approval_store.get(request_id)
        return ApprovalResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            requested_by=request.requested_by.user_id,
            status=request.status.value,
            approved_by=request.approved_by,
        )

    @app.post("/approvals/{request_id}/decide", response_model=ApprovalResponse, dependencies=protected)
    async def decide_approval(
        request_id: str, body: ApprovalDecisionRequest, identity: Identity = Depends(get_identity)
    ) -> ApprovalResponse:
        request = approval_store.decide(request_id, approved=body.approved, approved_by=identity.user_id)
        return ApprovalResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            requested_by=request.requested_by.user_id,
            status=request.status.value,
            approved_by=request.approved_by,
        )

    app.state.registry = registry
    app.state.approval_store = approval_store
    app.state.tracer = tracer
    app.state.metrics = metrics
    return app


__all__ = ["create_app"]
