"""Failure #10: duplicate request.

Expected behaviour:

    Duplicate request
        idempotency key -> detect duplicate -> return existing result,
        never re-run the side effect.

This project's HTTP layer (``src.api``) intentionally does *not* accept a
caller-supplied idempotency key on ``POST /runs`` -- every call mints a
brand-new ``run_id``/``thread_id`` (see ``src/api/runs.py``).

The actual, already-implemented idempotency mechanism in this codebase is
``src.reliability.idempotency`` (``InMemoryIdempotencyStore`` +
``idempotent()`` + the reference ``send_email``/``create_order``/
``publish_report`` tools). This file exercises the reusable primitive
directly against its documented reference tools. The graph-level
equivalent -- a crashed Researcher task replayed by LangGraph after a
restart never duplicates its already-committed checkpoint write -- is
exercised in ``test_worker_crash.py``.
"""

from __future__ import annotations

import pytest

from src.reliability.idempotency import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
    InMemoryIdempotencyStore,
    make_create_order_tool,
    make_publish_report_tool,
    make_send_email_tool,
)


class TestDuplicateSendEmail:
    def test_retrying_the_same_idempotency_key_does_not_send_twice(self):
        store = InMemoryIdempotencyStore()
        send_email = make_send_email_tool(store)

        first = send_email(to="user@example.com", subject="hi", body="hello", idempotency_key="welcome-42")
        second = send_email(to="user@example.com", subject="hi", body="hello", idempotency_key="welcome-42")

        assert first == second
        assert first.message_id == second.message_id

    def test_reusing_the_key_with_different_content_is_rejected_not_silently_resent(self):
        store = InMemoryIdempotencyStore()
        send_email = make_send_email_tool(store)

        send_email(to="user@example.com", subject="hi", body="hello", idempotency_key="welcome-42")
        with pytest.raises(IdempotencyConflictError):
            send_email(to="user@example.com", subject="DIFFERENT", body="hello", idempotency_key="welcome-42")


class TestDuplicateCreateOrder:
    """The canonical "must never double-charge" duplicate-request case."""

    def test_duplicate_checkout_retry_returns_the_same_order_never_a_second_one(self):
        store = InMemoryIdempotencyStore()
        create_order = make_create_order_tool(store)

        first = create_order(customer_id="cust-1", amount_cents=2500, idempotency_key="checkout-session-abc")
        second = create_order(customer_id="cust-1", amount_cents=2500, idempotency_key="checkout-session-abc")

        assert first.order_id == second.order_id

    def test_two_different_checkout_sessions_create_two_different_orders(self):
        store = InMemoryIdempotencyStore()
        create_order = make_create_order_tool(store)

        first = create_order(customer_id="cust-1", amount_cents=2500, idempotency_key="checkout-session-abc")
        different = create_order(customer_id="cust-1", amount_cents=2500, idempotency_key="checkout-session-xyz")

        assert first.order_id != different.order_id


class TestDuplicatePublishReport:
    def test_republishing_the_same_report_version_is_a_no_op(self):
        store = InMemoryIdempotencyStore()
        publish_report = make_publish_report_tool(store)

        first = publish_report(report_id="report-1", version=3, content="final content")
        second = publish_report(report_id="report-1", version=3, content="final content")

        assert first == second


class TestConcurrentDuplicateArrivesWhileFirstStillInFlight:
    """A duplicate request that arrives while the *first* call for the
    same key is still executing must never itself re-run the side effect
    or block indefinitely -- it must be told "in progress, retry later"
    (see ``IdempotencyInProgressError``'s own docstring)."""

    def test_second_call_while_first_in_progress_raises_in_progress_error(self):
        store = InMemoryIdempotencyStore()
        store.put_in_progress("in-flight-key", fingerprint="fp")

        from src.reliability.idempotency import idempotent

        call_count = {"n": 0}

        def side_effect(*, idempotency_key: str) -> str:
            call_count["n"] += 1
            return "done"

        wrapped = idempotent(
            side_effect,
            store=store,
            key_fn=lambda **kwargs: kwargs["idempotency_key"],
            fingerprint_fn=lambda **kwargs: "fp",
        )

        with pytest.raises(IdempotencyInProgressError):
            wrapped(idempotency_key="in-flight-key")
        assert call_count["n"] == 0  # never actually invoked the side effect
