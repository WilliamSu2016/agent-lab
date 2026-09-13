"""Local, privacy-safe tracing setup built entirely on the OpenAI Agents SDK.

The SDK's tracing system (see ``agents.tracing``) is enabled by default and, out
of the box, exports every trace/span to OpenAI's public tracing backend
(``https://api.openai.com/v1/traces/ingest``) via ``BackendSpanExporter``. This
project talks to a private, OpenAI-compatible gateway (``OPENAI_BASE_URL``), so
sending request/response content to OpenAI's public backend would be an
unintended data-exfiltration path -- regardless of which model provider actually
served the request.

Instead of disabling tracing outright, this module keeps the SDK's tracing
*fully enabled* (agent runs, LLM generations, tool calls, tool results, and the
final output are all still recorded) but replaces the default processor with
one that writes trace/span data to a local JSON-lines file. No third-party
observability framework, no OpenTelemetry, and no database are involved --
only the SDK's own ``TracingProcessor``/``TracingExporter`` extension points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.tracing import Span, Trace, set_trace_processors
from agents.tracing.processor_interface import TracingExporter
from agents.tracing.processors import BatchTraceProcessor

DEFAULT_TRACE_FILE = Path("traces") / "trace.jsonl"


class LocalFileExporter(TracingExporter):
    """Writes exported traces/spans to a local JSON-lines file.

    This is the SDK's own ``TracingExporter`` extension point (see
    ``agents.tracing.processor_interface.TracingExporter``); it is not a
    third-party observability integration. Each line is one trace or span,
    serialized via the SDK's own ``.export()`` method on ``Trace``/``Span``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, items: list[Trace | Span[Any]]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item.export(), default=str) + "\n")


def enable_local_tracing(trace_file: Path = DEFAULT_TRACE_FILE) -> Path:
    """Route the SDK's built-in tracing to a local file instead of OpenAI's backend.

    Tracing itself stays enabled (``RunConfig.tracing_disabled=False``, the SDK
    default); only the *destination* of the exported traces/spans changes. Call
    this once at process startup, before any ``Runner.run`` / ``run_sync`` call.
    """
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    set_trace_processors([BatchTraceProcessor(LocalFileExporter(trace_file))])
    return trace_file
