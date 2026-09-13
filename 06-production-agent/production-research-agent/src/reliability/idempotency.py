"""Idempotency strategies for side-effecting Agent Tools.

Requirement: identify every Tool with a real external side effect (the
requirement's own examples: ``send_email``, ``create_order``,
``publish_report``) and design an idempotency strategy for each -- a retry
(from ``src/reliability/retry.py``), a duplicate agent call, or a replayed
step after a crash/restart must never re-execute the side effect or bill/act
twice.

This project's current Agent Tool layer (see
``docs/00-PRODUCTION-ARCHITECTURE.md``, section 1, row 4 / gap P2-01) has no
real side-effecting tool yet -- every existing "tool" call is a plain LLM
text completion, which has no external side effect to deduplicate. This
module therefore provides:

1. A small, reusable idempotency *framework* (:class:`IdempotencyStore`
   protocol, :func:`idempotent` wrapper) that any future side-effecting tool
   should be wired through.
2. Reference implementations of the three named example tools
   (``send_email``/``create_order``/``publish_report``) demonstrating the
   pattern end-to-end, with a working (non-durable, in-memory) store. These
   are illustrative reference tools, not wired into any production pipeline
   in this repository -- copy the pattern once a real side-effecting tool is
   added, backed by a durable store (see the module-level warning below and
   ``docs/02-RELIABILITY.md``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class IdempotencyConflictError(RuntimeError):
    """Raised when the same idempotency key is reused with materially
    different input -- e.g. retrying ``create_order`` with the same
    ``idempotency_key`` but a different ``amount_cents``. Must never
    silently return the previous, unrelated result, and must never perform
    the side effect a second time for the new input either.
    """


class IdempotencyInProgressError(RuntimeError):
    """Raised when a concurrent duplicate call arrives while the first
    call for the same key is still executing. This mirrors how idempotency
    keys behave in typical external APIs (e.g. a 409-style "still
    processing, try again" response) -- the correct caller behavior is to
    wait/poll or retry later, never to re-invoke the side effect."""

    def __init__(self, key: str):
        super().__init__(f"Idempotency key {key!r} is already in progress; retry later instead of re-executing.")
        self.key = key


@dataclass
class IdempotencyRecord:
    key: str
    fingerprint: str
    status: str  # "in_progress" | "completed"
    result: Any = None


class IdempotencyStore(Protocol):
    """Minimal storage contract for idempotent tool execution.

    A production implementation MUST be durable and shared across every
    process/replica that can invoke the tool -- e.g. a database table with a
    unique constraint on ``key``, so ``put_in_progress`` is an atomic
    "insert if not exists" even across concurrent processes. The in-memory
    implementation below is for tests and single-process demos ONLY: it is
    not durable across a process restart and does not coordinate across
    machines (see ``docs/02-RELIABILITY.md``).
    """

    def get(self, key: str) -> IdempotencyRecord | None: ...

    def put_in_progress(self, key: str, fingerprint: str) -> bool:
        """Atomically claim ``key`` for in-progress execution.

        Returns ``True`` if this call won the claim (i.e. it should proceed
        to actually run the side effect), ``False`` if another caller
        already claimed it first (a concurrent duplicate).
        """
        ...

    def complete(self, key: str, result: Any) -> None: ...

    def fail(self, key: str) -> None:
        """Release a claimed-but-failed key so a later call may retry it."""
        ...


class InMemoryIdempotencyStore:
    """Thread-safe, single-process idempotency store.

    NOT durable across process restarts and NOT shared across multiple
    processes/replicas -- a real deployment with more than one worker
    process, or any restart tolerance requirement, MUST replace this with a
    database-backed store (e.g. a table keyed by ``key`` with a unique
    constraint, `INSERT ... ON CONFLICT DO NOTHING` for
    ``put_in_progress``). See docs/02-RELIABILITY.md and
    docs/00-PRODUCTION-ARCHITECTURE.md's Idempotency section.
    """

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> IdempotencyRecord | None:
        with self._lock:
            return self._records.get(key)

    def put_in_progress(self, key: str, fingerprint: str) -> bool:
        with self._lock:
            if key in self._records:
                return False
            self._records[key] = IdempotencyRecord(key=key, fingerprint=fingerprint, status="in_progress")
            return True

    def complete(self, key: str, result: Any) -> None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                raise KeyError(f"No in-progress idempotency record for key={key!r}")
            record.status = "completed"
            record.result = result

    def fail(self, key: str) -> None:
        with self._lock:
            self._records.pop(key, None)


def idempotent(
    func: Callable[..., T],
    *,
    store: IdempotencyStore,
    key_fn: Callable[..., str],
    fingerprint_fn: Callable[..., str] | None = None,
) -> Callable[..., T]:
    """Wrap a side-effecting ``func`` with idempotent-execution semantics.

    * ``key_fn(*args, **kwargs)`` computes the idempotency key from the
      call's arguments. It must be a **caller-supplied, stable** identifier
      (e.g. a client-generated ``idempotency_key``, or a business key such
      as ``f"{report_id}:{version}"``) -- never a value freshly generated
      *inside* the tool (e.g. ``uuid4()`` on every call), which would defeat
      deduplication entirely.
    * ``fingerprint_fn`` (defaults to a repr of the arguments) captures
      enough of the call's input to detect a *conflicting* reuse of the same
      key with different arguments; a mismatch raises
      :class:`IdempotencyConflictError` instead of silently returning an
      unrelated cached result or silently re-running the side effect.
    * First call for a given key: claims the key, runs ``func``, stores the
      result, and returns it.
    * A later call with the *same* key and the *same* fingerprint, made
      after the first call already completed, returns the cached result
      immediately without invoking ``func`` again -- this is the actual
      deduplication.
    * A concurrent duplicate call (same key, first call still running)
      raises :class:`IdempotencyInProgressError` rather than either
      blocking indefinitely or running the side effect twice.
    * If ``func`` raises, the claim is released (:meth:`IdempotencyStore.fail`)
      so a legitimate retry with the same key can try again later.
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = key_fn(*args, **kwargs)
        fingerprint = fingerprint_fn(*args, **kwargs) if fingerprint_fn else f"{args!r}|{kwargs!r}"

        existing = store.get(key)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    f"Idempotency key {key!r} was already used with different arguments "
                    f"(existing fingerprint={existing.fingerprint!r}, new fingerprint={fingerprint!r}); "
                    "refusing to re-execute the side effect or return an unrelated cached result."
                )
            if existing.status == "completed":
                return existing.result
            raise IdempotencyInProgressError(key)

        claimed = store.put_in_progress(key, fingerprint)
        if not claimed:
            # Lost a race against a concurrent duplicate call that claimed
            # the key between our `get` and `put_in_progress` above.
            record = store.get(key)
            if record is not None and record.fingerprint == fingerprint and record.status == "completed":
                return record.result
            raise IdempotencyInProgressError(key)

        try:
            result = func(*args, **kwargs)
        except Exception:
            store.fail(key)
            raise
        store.complete(key, result)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Reference tools: the pattern applied to the three named side-effecting
# tool examples. NOT wired into any production pipeline in this repository
# (see module docstring) -- these exist so a future real integration has a
# concrete strategy to copy, and so the strategy itself is exercised by
# tests (see tests/test_failure_injection.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmailSendResult:
    message_id: str
    to: str
    idempotency_key: str


def _send_email_side_effect(*, to: str, subject: str, body: str, idempotency_key: str) -> EmailSendResult:
    """Placeholder for a real email-sending side effect (e.g. an SMTP/API
    call). Not a real integration -- reference implementation only."""
    import uuid

    return EmailSendResult(message_id=f"msg-{uuid.uuid4().hex[:12]}", to=to, idempotency_key=idempotency_key)


def make_send_email_tool(store: IdempotencyStore) -> Callable[..., EmailSendResult]:
    """``send_email`` idempotency strategy.

    Dedup key: the caller-supplied ``idempotency_key`` (e.g. derived from
    the originating business event, such as "welcome-email-for-order-123"),
    never generated inside the tool. Fingerprint covers every field that
    defines "the same email" (recipient/subject/body) so reusing the key
    with different content is rejected instead of silently sending the
    wrong email once and then serving a stale cached result for the rest.
    Without this, a retried Agent Tool call (e.g. after a network timeout
    whose response was lost) would send the same email twice.
    """
    return idempotent(
        _send_email_side_effect,
        store=store,
        key_fn=lambda **kwargs: kwargs["idempotency_key"],
        fingerprint_fn=lambda **kwargs: f"{kwargs['to']}|{kwargs['subject']}|{kwargs['body']}",
    )


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    customer_id: str
    amount_cents: int


def _create_order_side_effect(*, customer_id: str, amount_cents: int, idempotency_key: str) -> OrderResult:
    """Placeholder for a real order-creation side effect (e.g. a payment/
    order-service API call). Not a real integration -- reference
    implementation only."""
    import uuid

    return OrderResult(order_id=f"order-{uuid.uuid4().hex[:12]}", customer_id=customer_id, amount_cents=amount_cents)


def make_create_order_tool(store: IdempotencyStore) -> Callable[..., OrderResult]:
    """``create_order`` idempotency strategy.

    Dedup key: a caller-supplied ``idempotency_key`` (e.g. derived from the
    client's checkout session). This is the canonical "must never
    double-charge" case: a retry after the order was actually created
    server-side, but before the success response reached the caller (a lost
    response, not a lost request), must return the *same* ``order_id`` on
    retry, never create a second order for the same checkout.
    """
    return idempotent(
        _create_order_side_effect,
        store=store,
        key_fn=lambda **kwargs: kwargs["idempotency_key"],
        fingerprint_fn=lambda **kwargs: f"{kwargs['customer_id']}|{kwargs['amount_cents']}",
    )


@dataclass(frozen=True)
class ReportPublishResult:
    report_id: str
    version: int


def _publish_report_side_effect(*, report_id: str, version: int, content: str) -> ReportPublishResult:
    """Placeholder for a real report-publication side effect (e.g. writing
    to a public artifact store and notifying subscribers). Not a real
    integration -- reference implementation only."""
    return ReportPublishResult(report_id=report_id, version=version)


def make_publish_report_tool(store: IdempotencyStore) -> Callable[..., ReportPublishResult]:
    """``publish_report`` idempotency strategy.

    Dedup key: the natural business key ``f"{report_id}:{version}"`` --
    publishing the *same* report version twice (e.g. because a Reviewer
    retry, a crash-recovery replay, or a duplicate scheduler trigger
    re-ran the publish step) must be a no-op that returns the original
    publish result, never a second publish event or a duplicate
    notification to subscribers. Publishing a *new* version of the same
    report is a different key (different ``version``) and is correctly
    allowed to proceed as a new side effect.
    """
    return idempotent(
        _publish_report_side_effect,
        store=store,
        key_fn=lambda **kwargs: f"{kwargs['report_id']}:{kwargs['version']}",
        fingerprint_fn=lambda **kwargs: kwargs["content"],
    )
