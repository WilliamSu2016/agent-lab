"""Requirement 5: error handling.

One exception -> one structured JSON response mapping, registered once in
``app.py``. Every handler here does two things, always in this order:

1. Log the *full* detail server-side (via
   ``src.observability.logging.log_event``, which itself redacts
   sensitive data -- see that module) -- so an operator can always
   correlate a returned ``request_id`` back to the real cause.
2. Return a *sanitized*, generic-enough message to the client -- never a
   raw ``str(exc)`` for an unexpected/internal error, which could leak
   internal paths, stack frames, or (worse) a secret interpolated into an
   exception message somewhere deep in a dependency.

Domain-specific exceptions (rate limit, not-found, invalid state, auth)
already carry a safe, specific message, so those are passed straight
through -- only the catch-all ``Exception`` handler substitutes a generic
message.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Request, status
from fastapi.responses import JSONResponse

from src.api.rate_limit import RateLimitExceededError
from src.api.runs import InvalidRunStateError, RunNotFoundError
from src.observability.logging import configure_json_logging, log_event
from src.reliability.timeout import ToolTimeoutError
from src.security.authorization import AuthorizationError

_logger = configure_json_logging(logger_name="agent.api")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def _error(request: Request, status_code: int, error_code: str, message: str, **headers: str) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={"request_id": request_id, "error_code": error_code, "message": message},
        headers={"X-Request-ID": request_id, **headers},
    )


async def handle_rate_limit_exceeded(request: Request, exc: RateLimitExceededError) -> JSONResponse:
    log_event(_logger, logging.WARNING, "rate_limit_exceeded", key=exc.key, retry_after_seconds=exc.retry_after_seconds)
    return _error(
        request,
        status.HTTP_429_TOO_MANY_REQUESTS,
        "rate_limited",
        "Too many requests. Please retry later.",
        **{"Retry-After": str(int(exc.retry_after_seconds) + 1)},
    )


async def handle_run_not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
    return _error(request, status.HTTP_404_NOT_FOUND, "run_not_found", f"No run found with id {exc.run_id!r}.")


async def handle_invalid_run_state(request: Request, exc: InvalidRunStateError) -> JSONResponse:
    return _error(
        request,
        status.HTTP_409_CONFLICT,
        "invalid_run_state",
        f"Run {exc.run_id!r} cannot be resumed from status {exc.status!r}.",
    )


async def handle_authorization_error(request: Request, exc: AuthorizationError) -> JSONResponse:
    log_event(_logger, logging.WARNING, "authorization_denied", reason=str(exc))
    return _error(request, status.HTTP_403_FORBIDDEN, "forbidden", "You are not authorized to perform this action.")


async def handle_timeout(request: Request, exc: ToolTimeoutError) -> JSONResponse:
    log_event(_logger, logging.ERROR, "request_timeout", reason=str(exc))
    return _error(request, status.HTTP_504_GATEWAY_TIMEOUT, "timeout", "The request exceeded its configured timeout.")


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id(request)
    # Full detail server-side only; the client never sees exc's own message.
    log_event(
        _logger,
        logging.ERROR,
        "unhandled_exception",
        request_id=request_id,
        exception_type=type(exc).__name__,
        detail=str(exc),
    )
    return _error(request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "An unexpected error occurred.")


EXCEPTION_HANDLERS = {
    RateLimitExceededError: handle_rate_limit_exceeded,
    RunNotFoundError: handle_run_not_found,
    InvalidRunStateError: handle_invalid_run_state,
    AuthorizationError: handle_authorization_error,
    ToolTimeoutError: handle_timeout,
    Exception: handle_unexpected_error,
}

__all__ = ["EXCEPTION_HANDLERS"]
