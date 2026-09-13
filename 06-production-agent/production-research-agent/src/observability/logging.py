"""Structured JSON logging with mandatory identifiers and sensitive-data
redaction by default.

Requirement: "敏感数据不得默认进入日志" (sensitive data must not enter logs
by default). This module enforces that at the formatter level -- every
record's rendered ``message`` *and* every structured ``extra`` field is
passed through redaction before being serialized, so a caller cannot
accidentally leak a secret by forgetting to redact it themselves at the
call site. Reuses ``src/security/sanitization.py::redact_sensitive_data``
rather than a second, divergent pattern list (the same category of match
should redact the same way whether it shows up in a tool response or a
log line).

Every log record also always carries the six mandatory identifiers
(request_id, trace_id, user_id, session_id, agent_version, environment),
pulled automatically from ``observability.tracing.current_execution_context()``
when one is bound -- callers do not need to pass them explicitly at every
``logger.info(...)`` call site.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from src.observability.tracing import current_execution_context
from src.security.sanitization import redact_sensitive_data

# ---------------------------------------------------------------------------
# Defense in depth: beyond regex value-matching (redact_sensitive_data
# looks at the *content* of a string), also redact by *field name* -- an
# ``extra={"password":??? "hunter2"}`` should never appear even if "hunter2"
# itself doesn't match any of the value-pattern regexes.
# ---------------------------------------------------------------------------

SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "authorization",
        "auth_token",
        "credit_card",
        "ssn",
        "private_key",
    }
)

_REDACTED_FIELD_PLACEHOLDER = "[REDACTED:field-name]"

# Attributes already present on every stdlib LogRecord -- excluded when
# harvesting caller-supplied ``extra`` fields so we don't re-serialize
# logging internals as if they were application data.
_STANDARD_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys())


def _redact_value(key: str, value: Any) -> Any:
    if key.lower() in SENSITIVE_FIELD_NAMES:
        return _REDACTED_FIELD_PLACEHOLDER
    if isinstance(value, str):
        return redact_sensitive_data(value)
    if isinstance(value, dict):
        return {k: _redact_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(key, item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Renders every log record as one JSON object per line.

    Always includes: ``timestamp``, ``level``, ``logger``, ``message``,
    and the six mandatory identifiers (as ``null`` if no
    :class:`~src.observability.tracing.ExecutionContext` is currently
    bound -- e.g. during early process startup, before the first request
    arrives). Any additional ``extra={...}`` fields a call site passes are
    included as well, after redaction.
    """

    def format(self, record: logging.LogRecord) -> str:
        context = current_execution_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_data(record.getMessage()),
            "request_id": context.request_id if context else None,
            "trace_id": context.trace_id if context else None,
            "user_id": context.user_id if context else None,
            "session_id": context.session_id if context else None,
            "agent_version": context.agent_version if context else None,
            "environment": context.environment if context else None,
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_")
        }
        for key, value in extras.items():
            payload[key] = _redact_value(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_json_logging(
    *, logger_name: str = "agent", level: int = logging.INFO, stream: Any = None
) -> logging.Logger:
    """Sets up (idempotently) a logger named ``logger_name`` that emits
    :class:`JsonFormatter`-rendered JSON lines to ``stream`` (default
    ``sys.stdout``, matching container/production log-collector
    conventions -- write to stdout, let the platform ship it elsewhere,
    rather than this process owning log files directly)."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Idempotent: calling this twice (e.g. once per test) must not stack
    # duplicate handlers and double-emit every line.
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Convenience wrapper: ``log_event(logger, logging.INFO, "tool_call_complete", tool_name="search_web", duration_ms=42)``.
    Equivalent to ``logger.log(level, message, extra=fields)`` but avoids
    every call site needing to know that ``extra=`` is the mechanism."""
    logger.log(level, message, extra=fields)


__all__ = [
    "SENSITIVE_FIELD_NAMES",
    "JsonFormatter",
    "configure_json_logging",
    "log_event",
]
