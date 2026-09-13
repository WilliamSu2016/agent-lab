"""Regression Evaluation: compare a new :class:`~evals.runner.EvaluationRun`
against a saved :class:`~evals.baseline.BaselineRun`, per-task, per-dimension.

This is the piece that fulfils "每次代码或 Prompt 修改: run eval -> compare
baseline -> detect regression" and "禁止只比较最终字符串": every comparison
below operates on structured dimension scores (0.0-1.0 floats and
pass/fail booleans) computed by ``evals.metrics``, never on the raw
``final_answer`` text of either run.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.baseline import BaselineRun
from evals.runner import EvaluationRun

DEFAULT_SCORE_DROP_THRESHOLD = 0.1


@dataclass(frozen=True)
class RegressionEntry:
    """One dimension, on one task, that got measurably worse."""

    task_id: str
    dimension: str
    baseline_score: float
    current_score: float

    @property
    def delta(self) -> float:
        return self.current_score - self.baseline_score

    def __str__(self) -> str:  # human-readable one-liner for CI logs
        return (
            f"{self.task_id}::{self.dimension}: {self.baseline_score:.2f} -> "
            f"{self.current_score:.2f} (Δ={self.delta:+.2f})"
        )


@dataclass(frozen=True)
class RegressionReport:
    baseline_run_id: str
    current_run_id: str
    entries: tuple[RegressionEntry, ...]
    newly_failing_tasks: tuple[str, ...]
    new_tasks: tuple[str, ...]
    removed_tasks: tuple[str, ...]

    @property
    def has_regressions(self) -> bool:
        return bool(self.entries) or bool(self.newly_failing_tasks)

    def summary(self) -> str:
        if not self.has_regressions:
            return f"No regressions detected ({self.baseline_run_id} -> {self.current_run_id})."
        lines = [f"{len(self.entries)} dimension regression(s), {len(self.newly_failing_tasks)} newly-failing task(s):"]
        lines += [f"  - {entry}" for entry in self.entries]
        lines += [f"  - {task_id}: passed on baseline, now FAILING" for task_id in self.newly_failing_tasks]
        return "\n".join(lines)


def detect_regressions(
    baseline: BaselineRun,
    current: EvaluationRun,
    *,
    score_drop_threshold: float = DEFAULT_SCORE_DROP_THRESHOLD,
) -> RegressionReport:
    """Compares ``current`` against ``baseline``.

    A regression entry is recorded whenever a task exists in both runs and
    some dimension's score dropped by more than ``score_drop_threshold``
    (default 0.1) -- a small amount of run-to-run noise (e.g. an LLM-judge
    based quality score fluctuating by a couple hundredths) is tolerated so
    the pipeline is not flaky; anything past the threshold is flagged.

    A task is additionally flagged in ``newly_failing_tasks`` if it passed
    on the baseline but fails now, *even if* no single dimension crossed
    ``score_drop_threshold`` on its own -- several small dimension drops can
    combine to flip the overall pass/fail verdict without any individual
    drop looking alarming in isolation.

    Tasks present in one run's dataset but not the other are reported
    separately (``new_tasks``/``removed_tasks``) rather than silently
    ignored -- a shrinking dataset is itself worth flagging to a reviewer.
    """
    baseline_ids = {r.task_id for r in baseline.reports}
    current_ids = {r.task_id for r in current.reports}

    entries: list[RegressionEntry] = []
    newly_failing: list[str] = []

    for current_report in current.reports:
        baseline_report = baseline.get(current_report.task_id)
        if baseline_report is None:
            continue  # a brand-new task -- reported via new_tasks instead

        for dim in current_report.dimension_scores:
            baseline_score = baseline_report.dimension_score(dim.name)
            if baseline_score is None:
                continue  # a brand-new dimension -- nothing to compare against
            if baseline_score - dim.score > score_drop_threshold:
                entries.append(
                    RegressionEntry(
                        task_id=current_report.task_id,
                        dimension=dim.name,
                        baseline_score=baseline_score,
                        current_score=dim.score,
                    )
                )

        if baseline_report.passed and not current_report.passed:
            newly_failing.append(current_report.task_id)

    return RegressionReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        entries=tuple(entries),
        newly_failing_tasks=tuple(newly_failing),
        new_tasks=tuple(sorted(current_ids - baseline_ids)),
        removed_tasks=tuple(sorted(baseline_ids - current_ids)),
    )


__all__ = [
    "DEFAULT_SCORE_DROP_THRESHOLD",
    "RegressionEntry",
    "RegressionReport",
    "detect_regressions",
]
