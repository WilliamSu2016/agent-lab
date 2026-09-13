"""Full Agent execution tree tracing.

This module answers, for any one request, the tree-shaped questions from
the requirement:

    1. 一个 Request 经过了哪些 Agent?        -> agents_visited(root)
    2. 调用了哪些 Tools?                       -> tools_called(root)
    3. 每个 Tool 花了多久?                     -> tool_durations_ms(root)
    7. Agent loop 执行了多少次?                -> count_loop_iterations(root, agent_name)
    8. 为什么最终失败?                          -> find_root_cause_failure(root)

by building an explicit, in-memory **span tree** per request -- one root
span (the whole request) with nested child spans (one per Agent
invocation, per Tool call, per LLM call) -- and exporting it as a single
nested JSON document. Questions 4/5/6 (tokens, cost, most-expensive-agent,
least-reliable-tool) are answered by ``metrics.py``, which is fed *from*
these same span trees (:func:`MetricsRegistry.record_from_trace` in
``metrics.py``) rather than duplicating any of this bookkeeping.

Every span belongs to exactly one :class:`ExecutionContext`, which is
where the five/six mandatory identifiers the requirement lists live:

    request_id, trace_id, user_id, session_id, agent_version, environment

Design note: parallel fan-out and ``contextvars``
--------------------------------------------------
This module offers an implicit-parent convenience (``tracer.span(...)``
infers its parent from a ``contextvars.ContextVar`` "current span", so a
normal, sequential call chain -- Supervisor -> Planner -> Worker ->
Synthesizer, exactly this project's ``multi_agent_research`` graph -- does
not need to thread span objects through every function signature).
``contextvars`` are copied into new threads/tasks by Python's own
``contextvars.copy_context()``/``asyncio`` machinery, but a plain
``threading.Thread`` started without going through that copy (as, for
example, ``reliability.timeout.run_with_timeout`` does with a raw daemon
thread) will NOT see the parent's current span. For genuine parallel
fan-out (this project's own ``research_worker`` fan-out in experiment 4/14),
pass the parent span explicitly via ``tracer.span(..., parent=parent_span)``
instead of relying on the contextvar -- the API supports both.
"""

from __future__ import annotations

import enum
import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


# ---------------------------------------------------------------------------
# ExecutionContext: the mandatory per-execution identifiers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionContext:
    """The six identifiers the requirement mandates every execution must
    carry. Immutable and always explicit -- the same "explicit identity,
    never ambient global state" principle as
    ``src/security/authorization.py::Identity`` (see that module's
    docstring for why: concurrent, multi-tenant fan-out is exactly where
    ambient/thread-local state silently leaks one request's identifiers
    into another's).
    """

    request_id: str
    trace_id: str
    user_id: str
    session_id: str
    agent_version: str
    environment: str

    def to_dict(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_version": self.agent_version,
            "environment": self.environment,
        }


def new_execution_context(
    *,
    user_id: str,
    session_id: str,
    agent_version: str,
    environment: str,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> ExecutionContext:
    """Builds a fresh :class:`ExecutionContext`. ``request_id``/``trace_id``
    default to newly generated UUIDs -- one request normally has its own
    ``request_id`` and its own ``trace_id`` (a 1:1 request:trace
    relationship in this module), but both may be supplied explicitly
    (e.g. propagated in from an upstream HTTP request header) so this
    process's trace stitches into a larger distributed trace."""
    return ExecutionContext(
        request_id=request_id or uuid.uuid4().hex,
        trace_id=trace_id or uuid.uuid4().hex,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        environment=environment,
    )


_current_context: ContextVar[Optional[ExecutionContext]] = ContextVar("_current_context", default=None)
_current_span: ContextVar[Optional["Span"]] = ContextVar("_current_span", default=None)


def current_execution_context() -> Optional[ExecutionContext]:
    """Used by ``logging.py`` to automatically attach request_id/trace_id/
    user_id/session_id/agent_version/environment to every log line without
    every call site having to pass them in manually."""
    return _current_context.get()


def current_span() -> Optional["Span"]:
    return _current_span.get()


@contextmanager
def bind_execution_context(context: ExecutionContext) -> Iterator[ExecutionContext]:
    """Binds ``context`` as the current execution context for the duration
    of the ``with`` block (and, via ``contextvars``, for any code called
    from within it on the same thread/task). Call this once per request,
    at the outermost entry point."""
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


class SpanKind(str, enum.Enum):
    """What a span represents. Matches the vocabulary the requirement's
    questions use directly (agent / tool / llm), plus ``WORKFLOW`` for the
    one root span wrapping an entire request."""

    WORKFLOW = "workflow"
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"


@dataclass
class Span:
    """One node in the execution tree. ``attributes`` is a free-form bag
    for whatever a given span kind needs to record -- e.g. a TOOL span
    stores ``retry_count``; an LLM span stores ``prompt_tokens``/
    ``completion_tokens``/``cost_usd``; an AGENT span may store
    ``iteration`` (which pass of a Planner<->Reviewer loop this was)."""

    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    request_id: str
    kind: SpanKind
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"  # "ok" | "error"
    error: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "kind": self.kind.value,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": dict(self.attributes),
            "children": [child.to_dict() for child in self.children],
        }


def walk(root: Span) -> Iterator[Span]:
    """Pre-order traversal of an execution tree -- the shared primitive
    every query helper below (and ``metrics.MetricsRegistry.record_from_trace``)
    is built on."""
    yield root
    for child in root.children:
        yield from walk(child)


# ---------------------------------------------------------------------------
# Tracer: builds and stores one execution tree per trace_id.
# ---------------------------------------------------------------------------

SpanSink = Callable[[dict[str, Any]], None]


class Tracer:
    """Builds one execution tree (one root :class:`Span`) per
    ``trace_id``. Keeps every completed root tree in memory
    (:meth:`get_trace`) and optionally forwards each completed *root* span
    (the full tree, nested) to a ``sink`` callback -- e.g.
    :func:`jsonl_file_sink` to persist every trace as one line of a
    ``.jsonl`` file, mirroring ``src/tracing.py``'s existing
    JSON-lines-file pattern for the OpenAI Agents SDK experiment, but for
    this framework-agnostic tree instead.
    """

    def __init__(self, sink: Optional[SpanSink] = None) -> None:
        self._roots: dict[str, Span] = {}
        self._sink = sink

    @contextmanager
    def span(
        self,
        kind: SpanKind,
        name: str,
        *,
        context: Optional[ExecutionContext] = None,
        parent: Optional[Span] = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Start a new span of the given ``kind``/``name``.

        ``context`` defaults to :func:`current_execution_context` (must be
        bound via :func:`bind_execution_context` if not passed explicitly).
        ``parent`` defaults to :func:`current_span` (the nearest enclosing
        ``with tracer.span(...)`` on this thread/task); pass it explicitly
        for genuine parallel fan-out where ``contextvars`` propagation
        cannot be relied on (see module docstring).

        On a raised exception inside the ``with`` block, the span is
        marked ``status="error"`` with ``error=str(exc)`` and the
        exception is re-raised unchanged (this never swallows a failure --
        it only records it, mirroring
        ``src/reliability/errors.py``'s "classify, never silently
        suppress" principle).
        """
        ctx = context or current_execution_context()
        if ctx is None:
            raise RuntimeError(
                "No ExecutionContext is bound. Call bind_execution_context(...) "
                "before starting a span, or pass context=... explicitly."
            )
        parent_span = parent if parent is not None else current_span()

        span = Span(
            span_id=uuid.uuid4().hex,
            parent_span_id=parent_span.span_id if parent_span else None,
            trace_id=ctx.trace_id,
            request_id=ctx.request_id,
            kind=kind,
            name=name,
            start_time=time.time(),
            attributes=dict(attributes),
        )
        if parent_span is not None:
            parent_span.children.append(span)
        else:
            self._roots[ctx.trace_id] = span

        span_token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record, then always re-raise
            span.status = "error"
            span.error = str(exc)
            raise
        finally:
            span.end_time = time.time()
            _current_span.reset(span_token)
            if parent_span is None and self._sink is not None:
                self._sink(span.to_dict())

    def get_trace(self, trace_id: str) -> Optional[Span]:
        return self._roots.get(trace_id)

    def all_traces(self) -> tuple[Span, ...]:
        return tuple(self._roots.values())


def jsonl_file_sink(path: Path) -> SpanSink:
    """A ready-to-use sink that appends each completed trace tree as one
    JSON line -- one line per request, the whole nested tree in that line,
    so a single ``grep``/``jq`` pass over the file can answer any of the
    tree-shaped questions for any past request without needing the
    in-memory ``Tracer`` that produced it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sink(trace_dict: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_dict, default=str) + "\n")

    return _sink


# ---------------------------------------------------------------------------
# Query helpers -- pure functions over a Span tree, directly answering the
# requirement's tree-shaped questions. Deliberately pure (take a `root:
# Span`, return a value) so they work identically on a tree built moments
# ago in-memory, or one re-loaded from a `jsonl_file_sink` line -- see
# `Span`-shaped dict -> dataclass hydration not needed for these helpers
# since they only need attribute access, not the dataclass type itself
# (tests exercise both in-memory Span trees and plain dict reconstructions).
# ---------------------------------------------------------------------------


def agents_visited(root: Span) -> list[str]:
    """Requirement Q1: "一个 Request 经过了哪些 Agent?" -- every AGENT span
    name, in the order first encountered (pre-order / chronological for a
    non-parallel run), without duplicates."""
    seen: list[str] = []
    for span in walk(root):
        if span.kind is SpanKind.AGENT and span.name not in seen:
            seen.append(span.name)
    return seen


def tools_called(root: Span) -> list[str]:
    """Requirement Q2: "调用了哪些 Tools?" -- every distinct TOOL span name."""
    seen: list[str] = []
    for span in walk(root):
        if span.kind is SpanKind.TOOL and span.name not in seen:
            seen.append(span.name)
    return seen


def tool_durations_ms(root: Span) -> dict[str, list[float]]:
    """Requirement Q3: "每个 Tool 花了多久?" -- every individual call's
    duration (a tool called more than once in one request keeps every
    call's duration, not just the last)."""
    durations: dict[str, list[float]] = {}
    for span in walk(root):
        if span.kind is SpanKind.TOOL and span.duration_ms is not None:
            durations.setdefault(span.name, []).append(span.duration_ms)
    return durations


def llm_calls(root: Span) -> list[Span]:
    """Every LLM-call span in the tree, in order -- the raw material
    ``metrics.py`` uses to answer Q4 (tokens) and Q5 (most expensive
    agent)."""
    return [span for span in walk(root) if span.kind is SpanKind.LLM]


def count_loop_iterations(root: Span, agent_name: str) -> int:
    """Requirement Q7: "Agent loop 执行了多少次?" -- counts how many times
    ``agent_name`` appears as an AGENT span anywhere in the tree (e.g. a
    Planner<->Reviewer loop that ran 3 times has 3 ``planner`` AGENT
    spans -- mirrors ``src/multi_agent_research``'s own ``iteration``
    counter, but derived independently from the trace instead of trusting
    the agent's self-reported count)."""
    return sum(1 for span in walk(root) if span.kind is SpanKind.AGENT and span.name == agent_name)


def find_root_cause_failure(root: Span) -> Optional[Span]:
    """Requirement Q8: "为什么最终失败?" -- returns the *deepest* (most
    specific) failed span, on the theory that the root cause of a failure
    is whichever leaf-most operation actually raised, not the outer
    Agent/workflow span that merely propagated it. Returns ``None`` if
    nothing in the tree failed."""
    failed = [span for span in walk(root) if span.status == "error"]
    if not failed:
        return None

    depth_by_span_id = {span.span_id: depth for depth, span in _spans_with_depth(root)}
    return max(failed, key=lambda span: depth_by_span_id[span.span_id])


def _spans_with_depth(root: Span, depth: int = 0) -> Iterator[tuple[int, Span]]:
    yield depth, root
    for child in root.children:
        yield from _spans_with_depth(child, depth + 1)


__all__ = [
    "ExecutionContext",
    "new_execution_context",
    "bind_execution_context",
    "current_execution_context",
    "current_span",
    "SpanKind",
    "Span",
    "walk",
    "Tracer",
    "jsonl_file_sink",
    "agents_visited",
    "tools_called",
    "tool_durations_ms",
    "llm_calls",
    "count_loop_iterations",
    "find_root_cause_failure",
]
