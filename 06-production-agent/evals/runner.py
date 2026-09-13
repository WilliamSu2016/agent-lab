"""The System Under Test contract, and the Offline Evaluation runner.

Everything here is deliberately agnostic to *how* the agent under test is
implemented (this project has several: ``multi_agent_research``,
``durable``, plain ``specialists``) -- an evaluation harness that only
worked against one specific graph implementation would not be reusable
across experiments. The one thing every System Under Test must do is
produce a structured :class:`AgentRunResult` -- never just a final answer
string -- so ``evals.metrics`` has enough structured detail to check tool
selection, tool arguments, routing, termination, retries, safety,
groundedness, citations, cost, and latency independently, instead of
collapsing all eleven dimensions into one final-string comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from evals.dataset.tasks import EvalTask, TASKS
from evals.metrics import DimensionScore, TaskEvalReport, evaluate_task


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the System Under Test made while handling a task."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Whether this call was actually gated through an approval step before
    # executing -- required to be True for any tool in
    # ``ExpectedBehavior.requires_approval_tools`` (Dimension 7: Safety).
    approved: bool = True


@dataclass(frozen=True)
class AgentRunResult:
    """What a System Under Test must return for one :class:`EvalTask`.

    This is the single structured artifact every one of the 11 evaluation
    dimensions is computed from -- deliberately never just ``final_answer``.
    """

    final_answer: str
    route: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    terminated: bool = True
    termination_reason: str = "completed"
    retry_count: int = 0
    # True if a guardrail intentionally blocked/refused the request (e.g. a
    # prompt-injection or authorization attack was correctly rejected).
    safety_blocked: bool = False
    # Raw evidence/snippets the agent actually gathered (e.g. tool outputs)
    # that the final answer is supposed to be grounded in -- distinct from
    # ``citations`` (the source identifiers/URLs presented to the user).
    evidence: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw_trace: tuple[str, ...] = ()


# A System Under Test takes one EvalTask and produces one AgentRunResult.
# It may be a thin adapter wrapping a real LangGraph app (translating the
# app's own state/output into this shape), or -- for fast, deterministic
# CI runs -- a fake/rule-based stand-in. Either way, ``evals.metrics``
# only ever sees this one shape.
SystemUnderTest = Callable[[EvalTask], AgentRunResult]


@dataclass(frozen=True)
class EvaluationRun:
    """The full result of one Offline Evaluation pass over a set of tasks."""

    run_id: str
    reports: tuple[TaskEvalReport, ...]

    @property
    def pass_rate(self) -> float:
        if not self.reports:
            return 0.0
        return sum(1 for r in self.reports if r.passed) / len(self.reports)

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.pass_rate

    def _dimension_scores(self, dimension: str) -> list[float]:
        return [r.dimension_score(dimension) for r in self.reports if r.has_dimension(dimension)]

    @property
    def tool_selection_accuracy(self) -> float:
        scores = self._dimension_scores("tool_selection")
        return sum(scores) / len(scores) if scores else 1.0

    @property
    def routing_accuracy(self) -> float:
        scores = self._dimension_scores("routing")
        return sum(scores) / len(scores) if scores else 1.0

    @property
    def safety_failure_rate(self) -> float:
        scores = self._dimension_scores("safety")
        if not scores:
            return 0.0
        return sum(1 for s in scores if s < 1.0) / len(scores)

    @property
    def average_latency_ms(self) -> float:
        values = [r.result.latency_ms for r in self.reports]
        return sum(values) / len(values) if values else 0.0

    @property
    def average_cost_usd(self) -> float:
        values = [r.result.cost_usd for r in self.reports]
        return sum(values) / len(values) if values else 0.0

    def get(self, task_id: str) -> Optional[TaskEvalReport]:
        for report in self.reports:
            if report.task_id == task_id:
                return report
        return None


def run_offline_evaluation(
    system_under_test: SystemUnderTest,
    *,
    tasks: Sequence[EvalTask] = TASKS,
    run_id: Optional[str] = None,
) -> EvaluationRun:
    """Runs every task through ``system_under_test`` and scores each one
    across all 11 dimensions (``evals.metrics.evaluate_task``). This *is*
    the Offline Evaluation: a full, deterministic pass over the fixed
    dataset, independent of any specific prior run -- the starting point
    both for accepting a first baseline and, later, for Regression
    Evaluation (``evals.regression.detect_regressions``) against that
    baseline.
    """
    reports: list[TaskEvalReport] = []
    for task in tasks:
        start = time.perf_counter()
        result = system_under_test(task)
        # If the System Under Test did not report its own latency, fall
        # back to wall-clock time measured here -- never silently leave
        # latency at 0, which would make the Cost/Latency dimensions
        # meaningless.
        if result.latency_ms <= 0:
            result = _with_latency(result, (time.perf_counter() - start) * 1000.0)
        reports.append(evaluate_task(task, result))

    return EvaluationRun(run_id=run_id or f"run-{int(time.time())}", reports=tuple(reports))


def _with_latency(result: AgentRunResult, latency_ms: float) -> AgentRunResult:
    return AgentRunResult(
        final_answer=result.final_answer,
        route=result.route,
        tool_calls=result.tool_calls,
        terminated=result.terminated,
        termination_reason=result.termination_reason,
        retry_count=result.retry_count,
        safety_blocked=result.safety_blocked,
        evidence=result.evidence,
        citations=result.citations,
        cost_usd=result.cost_usd,
        latency_ms=latency_ms,
        raw_trace=result.raw_trace,
    )


# ---------------------------------------------------------------------------
# Evaluation Report rendering -- the final required output.
# ---------------------------------------------------------------------------


def render_report(run: EvaluationRun) -> str:
    """Renders the required Evaluation Report: pass rate, failure rate,
    tool accuracy, routing accuracy, safety failure rate, average latency,
    average cost -- plus a per-task breakdown and per-dimension detail for
    every failing task, so a failure is immediately actionable (which
    dimension, which task, what was expected vs observed) rather than a
    bare "20/20 tasks, 1 failed"."""
    lines = [
        f"# Evaluation Report — {run.run_id}",
        "",
        f"- Tasks evaluated: {len(run.reports)}",
        f"- Pass rate: {run.pass_rate:.1%}",
        f"- Failure rate: {run.failure_rate:.1%}",
        f"- Tool selection accuracy: {run.tool_selection_accuracy:.1%}",
        f"- Routing accuracy: {run.routing_accuracy:.1%}",
        f"- Safety failure rate: {run.safety_failure_rate:.1%}",
        f"- Average latency: {run.average_latency_ms:.1f} ms",
        f"- Average cost: ${run.average_cost_usd:.4f}",
        "",
        "## Per-task results",
        "",
        "| Task | Risk | Passed | Overall Score | Failing Dimensions |",
        "|---|---|---|---|---|",
    ]
    for report in run.reports:
        failing = ", ".join(d.name for d in report.dimension_scores if not d.passed) or "—"
        lines.append(
            f"| {report.task_id} | {report.risk_level.value} | "
            f"{'✅' if report.passed else '❌'} | {report.overall_score:.2f} | {failing} |"
        )

    failing_reports = [r for r in run.reports if not r.passed]
    if failing_reports:
        lines += ["", "## Failure details", ""]
        for report in failing_reports:
            lines.append(f"### {report.task_id}")
            for dim in report.dimension_scores:
                if not dim.passed:
                    lines.append(f"- **{dim.name}** (score={dim.score:.2f}): {dim.details}")
            lines.append("")

    return "\n".join(lines)


def save_report(run: EvaluationRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run), encoding="utf-8")


__all__ = [
    "ToolCallRecord",
    "AgentRunResult",
    "SystemUnderTest",
    "EvaluationRun",
    "run_offline_evaluation",
    "render_report",
    "save_report",
]
