"""Metrics: latency, token usage, tool/agent failure rates, retry counts,
cost, and success rate -- aggregated across many requests (unlike
``tracing.py``, which is scoped to one request's execution tree).

:class:`MetricsRegistry` is fed in one of two ways:

* :meth:`MetricsRegistry.record_from_trace` -- the normal path. Walks a
  completed execution tree from ``tracing.py`` (one ``Span`` per Agent
  invocation / Tool call / LLM call) and derives every metric below from
  it, so instrumenting one call with ``tracer.span(...)`` automatically
  produces both a trace *and* metrics, with no separate bookkeeping to keep
  in sync.
* :meth:`MetricsRegistry.record_tool_call` / ``record_llm_call`` /
  ``record_agent_run`` directly, for callers that want to feed metrics
  without going through the full tracing tree (e.g. a lightweight
  test, or a component that only cares about metrics).

Answers the requirement's aggregate questions:

    4. 每次 LLM call 消耗多少 tokens?    -> LLMCallRecord.total_tokens (per call)
                                             + token_usage_total() (aggregated)
    5. 哪个 Agent 最昂贵?                -> most_expensive_agent()
    6. 哪个 Tool 最容易失败?             -> least_reliable_tool()

plus the seven metrics the requirement explicitly names: latency,
token_usage, tool_failure_rate, agent_failure_rate, retry_count, cost,
success_rate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from src.observability.tracing import Span, SpanKind, walk

# ---------------------------------------------------------------------------
# Illustrative token pricing. NOT real, current provider pricing -- a real
# deployment must source this from its actual billing agreement (and it
# varies by model and by input/output token type). Kept as a simple,
# override-able default purely so ``cost_usd`` has *some* deterministic
# value to aggregate/rank by in this experiment and its tests.
# ---------------------------------------------------------------------------

DEFAULT_PRICE_PER_1K_TOKENS_USD: dict[str, float] = {"prompt": 0.0015, "completion": 0.002}


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    price_per_1k_tokens: dict[str, float] = DEFAULT_PRICE_PER_1K_TOKENS_USD,
) -> float:
    return (prompt_tokens / 1000.0) * price_per_1k_tokens["prompt"] + (
        completion_tokens / 1000.0
    ) * price_per_1k_tokens["completion"]


# ---------------------------------------------------------------------------
# Individual event records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    duration_ms: float
    success: bool
    retry_count: int = 0


@dataclass(frozen=True)
class LLMCallRecord:
    agent_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    success: bool
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class AgentRunRecord:
    agent_name: str
    duration_ms: float
    success: bool
    cost_usd: float = 0.0
    retry_count: int = 0


@dataclass(frozen=True)
class LatencyStats:
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _latency_stats(durations_ms: list[float]) -> LatencyStats:
    if not durations_ms:
        return LatencyStats(count=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0)
    ordered = sorted(durations_ms)
    return LatencyStats(
        count=len(ordered),
        avg_ms=sum(ordered) / len(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1],
    )


# ---------------------------------------------------------------------------
# The registry itself.
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Accumulates records across as many requests/traces as are fed into
    it (typically the lifetime of one process, or one reporting window --
    a real deployment would instead push each record to a real metrics
    backend such as Prometheus/StatsD; this in-memory registry is the
    aggregation logic those integrations would sit behind, kept separate
    and independently testable per this project's existing convention of
    deferring "which real backend" decisions -- see
    ``reliability.idempotency.InMemoryIdempotencyStore``'s equivalent
    caveat)."""

    def __init__(self) -> None:
        self._tool_calls: list[ToolCallRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._agent_runs: list[AgentRunRecord] = []

    # -- ingestion -----------------------------------------------------

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.append(record)

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._llm_calls.append(record)

    def record_agent_run(self, record: AgentRunRecord) -> None:
        self._agent_runs.append(record)

    def record_from_trace(self, root: Span) -> None:
        """Bridges ``tracing.py`` execution trees into this registry.
        Every TOOL span becomes a :class:`ToolCallRecord`, every LLM span a
        :class:`LLMCallRecord`, every AGENT span an :class:`AgentRunRecord`
        -- ``attributes`` supplies the kind-specific fields (see each
        span-producing call site for which attributes it sets, e.g.
        ``docs/05-OBSERVABILITY.md``'s worked example)."""
        for span in walk(root):
            if span.duration_ms is None:
                continue  # an unfinished span (should not happen once the trace is complete)
            success = span.status == "ok"
            if span.kind is SpanKind.TOOL:
                self.record_tool_call(
                    ToolCallRecord(
                        tool_name=span.name,
                        duration_ms=span.duration_ms,
                        success=success,
                        retry_count=int(span.attributes.get("retry_count", 0)),
                    )
                )
            elif span.kind is SpanKind.LLM:
                prompt_tokens = int(span.attributes.get("prompt_tokens", 0))
                completion_tokens = int(span.attributes.get("completion_tokens", 0))
                cost_usd = span.attributes.get("cost_usd")
                if cost_usd is None:
                    cost_usd = estimate_cost_usd(prompt_tokens, completion_tokens)
                self.record_llm_call(
                    LLMCallRecord(
                        agent_name=str(span.attributes.get("agent_name", span.name)),
                        model=str(span.attributes.get("model", "unknown")),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration_ms=span.duration_ms,
                        success=success,
                        cost_usd=float(cost_usd),
                    )
                )
            elif span.kind is SpanKind.AGENT:
                self.record_agent_run(
                    AgentRunRecord(
                        agent_name=span.name,
                        duration_ms=span.duration_ms,
                        success=success,
                        cost_usd=float(span.attributes.get("cost_usd", 0.0)),
                        retry_count=int(span.attributes.get("retry_count", 0)),
                    )
                )

    # -- latency ---------------------------------------------------------

    def latency_stats(self, name: str, *, kind: SpanKind) -> LatencyStats:
        """Requirement metric: latency (per tool/agent/llm-call name)."""
        if kind is SpanKind.TOOL:
            durations = [r.duration_ms for r in self._tool_calls if r.tool_name == name]
        elif kind is SpanKind.AGENT:
            durations = [r.duration_ms for r in self._agent_runs if r.agent_name == name]
        elif kind is SpanKind.LLM:
            durations = [r.duration_ms for r in self._llm_calls if r.agent_name == name]
        else:
            durations = []
        return _latency_stats(durations)

    # -- token usage (Requirement Q4) -------------------------------------

    def token_usage_total(self, *, agent_name: Optional[str] = None) -> dict[str, int]:
        calls = self._llm_calls if agent_name is None else [c for c in self._llm_calls if c.agent_name == agent_name]
        prompt = sum(c.prompt_tokens for c in calls)
        completion = sum(c.completion_tokens for c in calls)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    # -- failure rates -----------------------------------------------------

    def tool_failure_rate(self, tool_name: str) -> float:
        """Requirement metric: tool_failure_rate."""
        calls = [c for c in self._tool_calls if c.tool_name == tool_name]
        if not calls:
            return 0.0
        return sum(1 for c in calls if not c.success) / len(calls)

    def agent_failure_rate(self, agent_name: str) -> float:
        """Requirement metric: agent_failure_rate."""
        runs = [r for r in self._agent_runs if r.agent_name == agent_name]
        if not runs:
            return 0.0
        return sum(1 for r in runs if not r.success) / len(runs)

    def success_rate(self, *, kind: Optional[SpanKind] = None, name: Optional[str] = None) -> float:
        """Requirement metric: success_rate. ``kind=None`` (default)
        computes the overall success rate across every recorded event of
        every kind; narrow with ``kind=``/``name=`` for a specific
        tool/agent/model."""
        records: list[bool]
        if kind is SpanKind.TOOL:
            records = [c.success for c in self._tool_calls if name is None or c.tool_name == name]
        elif kind is SpanKind.AGENT:
            records = [r.success for r in self._agent_runs if name is None or r.agent_name == name]
        elif kind is SpanKind.LLM:
            records = [c.success for c in self._llm_calls if name is None or c.agent_name == name]
        else:
            records = (
                [c.success for c in self._tool_calls]
                + [r.success for r in self._agent_runs]
                + [c.success for c in self._llm_calls]
            )
        if not records:
            return 1.0  # no data yet -- vacuously "all good", never divides by zero
        return sum(1 for ok in records if ok) / len(records)

    # -- retry count ---------------------------------------------------

    def retry_count(self, name: str) -> int:
        """Requirement metric: retry_count (summed across every recorded
        tool call and agent run with this name)."""
        return sum(c.retry_count for c in self._tool_calls if c.tool_name == name) + sum(
            r.retry_count for r in self._agent_runs if r.agent_name == name
        )

    # -- cost ------------------------------------------------------------

    def total_cost_usd(self, *, agent_name: Optional[str] = None) -> float:
        """Requirement metric: cost."""
        llm_cost = sum(c.cost_usd for c in self._llm_calls if agent_name is None or c.agent_name == agent_name)
        agent_cost = sum(r.cost_usd for r in self._agent_runs if agent_name is None or r.agent_name == agent_name)
        return llm_cost + agent_cost

    # -- derived rankings (Requirement Q5/Q6) -----------------------------

    def most_expensive_agent(self) -> Optional[tuple[str, float]]:
        """Requirement Q5: "哪个 Agent 最昂贵?" -- ranks by total attributed
        cost (LLM-call cost billed to that agent, plus any cost recorded
        directly on the agent-run span itself)."""
        totals: dict[str, float] = defaultdict(float)
        for call in self._llm_calls:
            totals[call.agent_name] += call.cost_usd
        for run in self._agent_runs:
            totals[run.agent_name] += run.cost_usd
        if not totals:
            return None
        return max(totals.items(), key=lambda item: item[1])

    def least_reliable_tool(self) -> Optional[tuple[str, float]]:
        """Requirement Q6: "哪个 Tool 最容易失败?" -- ranks by failure rate;
        ties broken by call volume (a tool with more observed calls is a
        more trustworthy signal than one with a single failed call)."""
        names = {c.tool_name for c in self._tool_calls}
        if not names:
            return None
        ranked = sorted(
            names,
            key=lambda name: (self.tool_failure_rate(name), len([c for c in self._tool_calls if c.tool_name == name])),
            reverse=True,
        )
        top = ranked[0]
        return top, self.tool_failure_rate(top)

    # -- dashboard-ready summary ------------------------------------------

    def summary(self) -> dict:
        """One dict covering every metric this module exposes, keyed by
        tool/agent name where applicable -- the direct data source for
        the Observability Dashboard (see
        ``docs/OBSERVABILITY-DASHBOARD-SPEC.md``)."""
        tool_names = sorted({c.tool_name for c in self._tool_calls})
        agent_names = sorted({r.agent_name for r in self._agent_runs})
        return {
            "success_rate_overall": self.success_rate(),
            "total_cost_usd": self.total_cost_usd(),
            "token_usage_total": self.token_usage_total(),
            "most_expensive_agent": self.most_expensive_agent(),
            "least_reliable_tool": self.least_reliable_tool(),
            "tools": {
                name: {
                    "latency": self.latency_stats(name, kind=SpanKind.TOOL).__dict__,
                    "failure_rate": self.tool_failure_rate(name),
                    "retry_count": self.retry_count(name),
                }
                for name in tool_names
            },
            "agents": {
                name: {
                    "latency": self.latency_stats(name, kind=SpanKind.AGENT).__dict__,
                    "failure_rate": self.agent_failure_rate(name),
                    "cost_usd": self.total_cost_usd(agent_name=name),
                    "token_usage": self.token_usage_total(agent_name=name),
                }
                for name in agent_names
            },
        }


__all__ = [
    "DEFAULT_PRICE_PER_1K_TOKENS_USD",
    "estimate_cost_usd",
    "ToolCallRecord",
    "LLMCallRecord",
    "AgentRunRecord",
    "LatencyStats",
    "MetricsRegistry",
]
