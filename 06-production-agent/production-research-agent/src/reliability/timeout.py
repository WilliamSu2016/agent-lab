"""Unified timeout enforcement for external tool calls.

Requirement: "每一个 external tool 必须有 timeout." This module provides one
canonical implementation, replacing the three nearly-identical ad hoc
``_run_with_timeout`` helpers that previously existed independently in
``src/multi_agent_research/research_worker.py``,
``src/multi_agent_research/graph.py``, and ``src/04_parallel_multi_agent.py``
-- every tool call in this project now goes through the same, single,
well-understood timeout mechanism.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ToolTimeoutError(TimeoutError):
    """Raised when a tool call exceeds its configured timeout.

    Carries ``tool_name``/``timeout_seconds`` so a caller building a
    structured ``ToolError`` (see ``src/reliability/errors.py``) has that
    context directly, without re-parsing the exception message.
    """

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(f"Tool {tool_name!r} exceeded its {timeout_seconds}s timeout.")
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


def run_with_timeout(
    func: Callable[[], T],
    timeout_seconds: float | None,
    *,
    tool_name: str = "tool",
) -> T:
    """Run ``func()`` bounded by ``timeout_seconds`` using a daemon thread.

    Deliberately not ``concurrent.futures.ThreadPoolExecutor``: that module
    keeps a *global* registry of every worker thread it has ever created and
    registers an ``atexit`` hook (``concurrent.futures.thread._python_exit``)
    that joins *all* of them before the interpreter is allowed to exit --
    even threads belonging to an executor you already called
    ``shutdown(wait=False)`` on. In practice a single hung/slow call (e.g. a
    real network request past its timeout) still blocks the whole *process*
    from exiting, long after the timeout you configured has already fired
    (this was observed empirically in this project's earlier experiments --
    see ``src/04_parallel_multi_agent.py``'s docstring). A plain
    ``threading.Thread(daemon=True)`` has no such hook: if it is still
    running when ``timeout_seconds`` elapses, :class:`ToolTimeoutError` is
    raised immediately, and the OS simply reclaims the abandoned thread
    whenever the process eventually exits, without blocking on it.

    ``timeout_seconds=None`` disables the timeout entirely -- kept as a
    single code path (rather than a separate branch at every call site) for
    "no deadline configured".

    IMPORTANT limitation: this bounds how long the *caller* waits; it does
    not, in general, *cancel* the underlying work. A real network request
    already in flight when the timeout fires keeps running server-side and
    can still complete, error, or bill after this function has already
    raised. Callers that need true cancellation should prefer a client with
    native timeout/cancellation support (e.g. an async HTTP client with a
    request-level deadline) instead of relying on this wrapper alone -- see
    ``docs/02-RELIABILITY.md`` for the full discussion and
    ``docs/00-PRODUCTION-ARCHITECTURE.md``'s Reliability Q6.
    """
    if timeout_seconds is None:
        return func()
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0 or None, got {timeout_seconds!r}")

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = func()
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's side, unmodified
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise ToolTimeoutError(tool_name, timeout_seconds)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")  # type: ignore[return-value]


@dataclass(frozen=True)
class TimeoutPolicy:
    """A named timeout budget for one tool.

    Bundling ``tool_name`` with ``timeout_seconds`` makes call sites
    self-documenting (``policy.run(call_llm)`` instead of a bare float
    passed positionally) and makes it trivial to build one of these per
    tool from ``config.Settings.limits`` at process start-up.
    """

    tool_name: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds!r}")

    def run(self, func: Callable[[], T]) -> T:
        return run_with_timeout(func, self.timeout_seconds, tool_name=self.tool_name)
