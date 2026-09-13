"""Requirement 3: rate limiting.

A plain in-memory sliding-window counter, keyed by caller identity
(``user_id``) rather than by IP -- IP-based limiting punishes every user
behind a shared NAT/proxy equally and lets a single misbehaving *user* hop
IPs to evade it; keying by the identity this API already authenticates is
both more precise and free (no extra lookup).

Known limitation, stated up front rather than discovered in an incident:
this is **per-process, in-memory** state, exactly like
``src.reliability.idempotency.InMemoryIdempotencyStore`` and
``src.security.authorization.ApprovalStore``. Running more than one API
replica means each replica enforces its own independent limit -- a caller
effectively gets ``limit * replica_count`` requests per window. A real
multi-replica deployment must move this to a shared store (e.g. Redis
``INCR``+``EXPIRE``) so the limit is enforced globally; the interface
below (``RateLimiter.check``) is intentionally the only integration point
``app.py`` depends on, so that swap requires no changes outside this file.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class RateLimitExceededError(RuntimeError):
    """Raised by :meth:`RateLimiter.check`. Carries ``retry_after_seconds``
    so the HTTP layer (``app.py``) can set a ``Retry-After`` header instead
    of leaving the caller to guess when to try again."""

    def __init__(self, key: str, retry_after_seconds: float):
        super().__init__(f"Rate limit exceeded for {key!r}; retry after {retry_after_seconds:.1f}s.")
        self.key = key
        self.retry_after_seconds = retry_after_seconds


@dataclass
class RateLimiter:
    """Fixed-size sliding window: at most ``max_requests`` calls to
    :meth:`check` for the same ``key`` in any trailing ``window_seconds``
    interval. Thread-safe (guarded by one lock) since the API server may
    serve concurrent requests from a thread pool."""

    max_requests: int = 60
    window_seconds: float = 60.0
    _hits: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ValueError(f"max_requests must be > 0, got {self.max_requests!r}.")
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds!r}.")

    def check(self, key: str, *, now: float | None = None) -> None:
        """Records one hit for ``key`` and raises
        :class:`RateLimitExceededError` if that pushes ``key`` over the
        limit within the current window. Call once per incoming request,
        before any expensive work happens."""
        current_time = now if now is not None else time.time()
        cutoff = current_time - self.window_seconds
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(hits) >= self.max_requests:
                oldest = min(hits)
                retry_after = max(0.0, (oldest + self.window_seconds) - current_time)
                self._hits[key] = hits
                raise RateLimitExceededError(key, retry_after)
            hits.append(current_time)
            self._hits[key] = hits


__all__ = ["RateLimitExceededError", "RateLimiter"]
