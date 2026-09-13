"""Short-term, per-tenant episodic memory.

Honest scope note (do not oversell this as RAG/long-term memory): this is
a small, process-local cache mapping ``(tenant_id, normalized question)``
-> the most recent final answer for that question, used only to let the
Supervisor short-circuit an exact repeat of a question a tenant already
asked and got a completed answer for (a cheap, safe win for cost/latency
that is *not* semantic retrieval, has no embedding/vector store, and is
never treated as a source of truth for the LLM -- it only ever returns a
previous run's own final answer, never injects arbitrary retrieved text
into a prompt).

For real long-term memory / RAG (semantic recall across many distinct
past questions, embeddings, a vector store), a dedicated component is
required and is explicitly out of scope here -- see
``docs/architecture.md``'s Memory section for the reasoning.

Tenant isolation: every read/write is keyed by ``tenant_id`` (an explicit
argument, never ambient state -- same discipline as
``src.security.authorization.Identity``), so one tenant's cached answers
are never visible to another tenant.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


@dataclass(frozen=True)
class MemoryEntry:
    question: str
    answer: str
    thread_id: str
    created_at: float


class SessionMemoryStore:
    """In-memory, per-tenant cache of (question -> last completed answer).

    NOT durable across a process restart (same documented caveat as
    ``src.security.authorization.ApprovalStore`` /
    ``src.reliability.idempotency.InMemoryIdempotencyStore``): a real
    deployment wanting this cache to survive a restart would back it with
    a real key-value store, keyed the same way.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], MemoryEntry] = {}
        self._lock = threading.Lock()

    def remember(self, tenant_id: str, question: str, answer: str, thread_id: str) -> None:
        key = (tenant_id, _normalize_question(question))
        with self._lock:
            self._entries[key] = MemoryEntry(
                question=question, answer=answer, thread_id=thread_id, created_at=time.time()
            )

    def recall(self, tenant_id: str, question: str) -> Optional[MemoryEntry]:
        key = (tenant_id, _normalize_question(question))
        with self._lock:
            return self._entries.get(key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__ = ["MemoryEntry", "SessionMemoryStore"]
