"""``update_record`` (MEDIUM) and ``send_email`` (HIGH) -- reference
tools demonstrating the two risk tiers above LOW, and the mandatory
``Agent -> approval -> Tool`` gate for HIGH-risk calls.

Nothing in the Researcher/Synthesizer/Reviewer pipeline needs to call
these during a normal research run -- they exist so this package has a
real, testable MEDIUM tool (a reversible write) and a real, testable
HIGH tool (an irreversible external side effect) wired end to end through
``src.security.guardrails.secure_tool_call``, closing the Final Review's
finding that the HIGH-risk approval path (``ApprovalStore``) was
implemented but never reachable from anywhere in the real request path.
The Supervisor's ``notify_action`` node (see ``src/agents/supervisor.py``)
optionally invokes ``send_email`` when a run is created with
``notify=True`` -- exercising exactly this path.
"""

from __future__ import annotations

from typing import Callable

RecordStore = dict[str, dict[str, str]]


def build_update_record_tool(store: RecordStore | None = None) -> Callable[..., str]:
    """MEDIUM risk: writes one field of one record, but the previous value
    is retained (reversible -- a caller can always write it back)."""
    backing_store: RecordStore = store if store is not None else {}

    def update_record(record_id: str, field: str, value: str) -> str:
        record = backing_store.setdefault(record_id, {})
        previous = record.get(field)
        record[field] = value
        return f"updated record_id={record_id!r} field={field!r} previous={previous!r} new={value!r}"

    return update_record


def build_send_email_tool(sink: Callable[[str, str, str], None] | None = None) -> Callable[..., str]:
    """HIGH risk: an external side effect (an email) that cannot be
    recalled once sent. ``sink`` defaults to a no-op so tests never send a
    real email; production wiring supplies a real mail-sending callable."""
    send = sink or (lambda to, subject, body: None)

    def send_email(to: str, subject: str, body: str) -> str:
        send(to, subject, body)
        return f"sent email to={to!r} subject={subject!r}"

    return send_email


__all__ = ["RecordStore", "build_update_record_tool", "build_send_email_tool"]
