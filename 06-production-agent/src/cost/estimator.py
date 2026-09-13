"""Cost model for the Multi-Agent Research Agent.

Two complementary directions:

* **Retrospective** (:func:`estimate_request_cost`): given one completed
  request's execution trace (a ``src.observability.tracing.Span`` tree --
  the same tree ``src.observability.metrics.MetricsRegistry`` is fed from),
  compute exactly what that one request actually cost and how long it
  actually took, broken down into every quantity the requirement lists:
  LLM calls, input/output tokens, tool calls, subagent calls, retry calls,
  and parallel workers.
* **Prospective** (:func:`predict_cost`): given only an
  :class:`~src.cost.policy.ExecutionPolicy` (i.e. *before* a request has
  run at all), predict roughly what a request under that policy will cost
  and how long it will take, using simple, clearly-labeled-as-illustrative
  per-call averages. This is what a caller uses to decide, upfront, "can
  we even afford to try Quality mode for this request" before spending
  anything.

Both reuse ``src.observability.metrics.estimate_cost_usd`` for the actual
dollar math rather than a second, divergent pricing table -- one pricing
model, two different questions asked of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.cost.policy import ExecutionPolicy
from src.observability.metrics import estimate_cost_usd
from src.observability.tracing import Span, SpanKind, walk


@dataclass(frozen=True)
class RequestCostReport:
    """Everything the requirement asks to be counted/computed for one
    already-completed request."""

    llm_calls: int
    input_tokens: int
    output_tokens: int
    tool_calls: int
    subagent_calls: int
    retry_calls: int
    parallel_workers: int
    cost_usd: float
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_request_cost(root: Span, *, worker_span_name: str = "research_worker") -> RequestCostReport:
    """Walks a completed execution tree (``root`` is the request's top-level
    WORKFLOW span from ``src.observability.tracing.Tracer``) and computes
    the retrospective cost/latency report.

    ``parallel_workers`` is computed as the maximum number of sibling
    spans named ``worker_span_name`` under any single parent in the tree --
    i.e. the widest fan-out this request actually triggered in one
    Planner iteration (matches ``src/multi_agent_research/research_worker.py``'s
    fan-out, but is computed generically off the span *name*, so it works
    for any graph that names its parallel worker spans consistently).
    """
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0
    tool_calls = 0
    subagent_calls = 0
    retry_calls = 0
    cost_usd = 0.0
    max_parallel_workers = 0

    for span in walk(root):
        retry_calls += int(span.attributes.get("retry_count", 0))

        if span.kind is SpanKind.LLM:
            llm_calls += 1
            prompt_tokens = int(span.attributes.get("prompt_tokens", 0))
            completion_tokens = int(span.attributes.get("completion_tokens", 0))
            input_tokens += prompt_tokens
            output_tokens += completion_tokens
            call_cost = span.attributes.get("cost_usd")
            cost_usd += float(call_cost) if call_cost is not None else estimate_cost_usd(prompt_tokens, completion_tokens)
        elif span.kind is SpanKind.TOOL:
            tool_calls += 1
        elif span.kind is SpanKind.AGENT:
            subagent_calls += 1

        worker_children = [child for child in span.children if child.kind is SpanKind.AGENT and child.name == worker_span_name]
        max_parallel_workers = max(max_parallel_workers, len(worker_children))

    return RequestCostReport(
        llm_calls=llm_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        subagent_calls=subagent_calls,
        retry_calls=retry_calls,
        parallel_workers=max_parallel_workers,
        cost_usd=cost_usd,
        latency_ms=root.duration_ms or 0.0,
    )


# ---------------------------------------------------------------------------
# Prospective estimate -- illustrative averages, not a promise.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostPrediction:
    """A rough, pre-execution estimate for one :class:`ExecutionPolicy`.
    Deliberately labeled "illustrative": the average-tokens-per-call and
    average-seconds-per-call inputs are simple defaults a caller should
    override with real historical averages (e.g. pulled from
    ``src.observability.metrics.MetricsRegistry`` once enough requests have
    actually run) rather than trusting the built-in defaults as authoritative.
    """

    predicted_llm_calls: int
    predicted_tokens: int
    predicted_cost_usd: float
    predicted_latency_ms: float


# Illustrative-only defaults -- see the module and class docstrings above.
DEFAULT_AVG_PROMPT_TOKENS_PER_CALL = 800
DEFAULT_AVG_COMPLETION_TOKENS_PER_CALL = 400
DEFAULT_AVG_SECONDS_PER_LLM_CALL = 2.5


def predict_cost(
    policy: ExecutionPolicy,
    *,
    avg_prompt_tokens_per_call: int = DEFAULT_AVG_PROMPT_TOKENS_PER_CALL,
    avg_completion_tokens_per_call: int = DEFAULT_AVG_COMPLETION_TOKENS_PER_CALL,
    avg_seconds_per_llm_call: float = DEFAULT_AVG_SECONDS_PER_LLM_CALL,
) -> CostPrediction:
    """Predicts cost/latency for one request under ``policy``, *before*
    running it, using the Planner<->Reviewer<->Worker call shape this
    project's Multi-Agent Research graph has: each iteration is
    (1 planner call) + (up to ``max_workers`` worker calls, in parallel --
    parallel fan-out adds to *cost* but not to wall-clock *latency*) +
    (1 synthesizer call) + (1 reviewer call), repeated up to
    ``max_agent_iterations`` times.
    """
    calls_per_iteration = 1 + policy.max_workers + 1 + 1  # planner + workers + synthesizer + reviewer
    predicted_llm_calls = calls_per_iteration * policy.max_agent_iterations

    per_call_tokens = avg_prompt_tokens_per_call + avg_completion_tokens_per_call
    predicted_tokens = min(predicted_llm_calls * per_call_tokens, policy.budget.max_tokens)

    per_call_cost = estimate_cost_usd(avg_prompt_tokens_per_call, avg_completion_tokens_per_call)
    predicted_cost_usd = min(predicted_llm_calls * per_call_cost, policy.budget.max_cost_usd)

    # Latency model: sequential calls within one iteration (planner ->
    # workers-in-parallel -> synthesizer -> reviewer) count the *parallel*
    # worker fan-out as a single wall-clock step, not max_workers steps --
    # that is the entire point of running workers in parallel.
    sequential_calls_per_iteration = 1 + 1 + 1 + 1  # planner, one parallel worker-step, synthesizer, reviewer
    predicted_latency_ms = min(
        sequential_calls_per_iteration * policy.max_agent_iterations * avg_seconds_per_llm_call * 1000.0,
        policy.budget.timeout_seconds * 1000.0,
    )

    return CostPrediction(
        predicted_llm_calls=predicted_llm_calls,
        predicted_tokens=predicted_tokens,
        predicted_cost_usd=predicted_cost_usd,
        predicted_latency_ms=predicted_latency_ms,
    )


__all__ = [
    "RequestCostReport",
    "estimate_request_cost",
    "CostPrediction",
    "DEFAULT_AVG_PROMPT_TOKENS_PER_CALL",
    "DEFAULT_AVG_COMPLETION_TOKENS_PER_CALL",
    "DEFAULT_AVG_SECONDS_PER_LLM_CALL",
    "predict_cost",
]
