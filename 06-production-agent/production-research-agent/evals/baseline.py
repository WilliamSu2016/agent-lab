"""Persist and load an :class:`~evals.runner.EvaluationRun` as a baseline.

A baseline is a structured, per-task, per-dimension snapshot of scores --
not a snapshot of final-answer strings -- so a later comparison
(``evals.regression.detect_regressions``) can pinpoint exactly which
dimension of which task got worse, rather than only noticing "the output
text changed" (which happens on almost every run, for entirely benign
reasons -- different phrasing, different but equally valid tool-call
ordering, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evals.runner import EvaluationRun


@dataclass(frozen=True)
class BaselineDimensionScore:
    name: str
    score: float
    passed: bool


@dataclass(frozen=True)
class BaselineTaskReport:
    task_id: str
    risk_level: str
    overall_score: float
    passed: bool
    dimension_scores: tuple[BaselineDimensionScore, ...]
    cost_usd: float
    latency_ms: float

    def dimension_score(self, name: str) -> Optional[float]:
        for dim in self.dimension_scores:
            if dim.name == name:
                return dim.score
        return None


@dataclass(frozen=True)
class BaselineRun:
    run_id: str
    reports: tuple[BaselineTaskReport, ...]

    def get(self, task_id: str) -> Optional[BaselineTaskReport]:
        for report in self.reports:
            if report.task_id == task_id:
                return report
        return None


def _to_baseline_run(run: EvaluationRun) -> BaselineRun:
    reports = tuple(
        BaselineTaskReport(
            task_id=report.task_id,
            risk_level=report.risk_level.value,
            overall_score=report.overall_score,
            passed=report.passed,
            dimension_scores=tuple(
                BaselineDimensionScore(name=d.name, score=d.score, passed=d.passed) for d in report.dimension_scores
            ),
            cost_usd=report.result.cost_usd,
            latency_ms=report.result.latency_ms,
        )
        for report in run.reports
    )
    return BaselineRun(run_id=run.run_id, reports=reports)


def save_baseline(run: EvaluationRun, path: Path) -> None:
    """Serializes ``run`` to JSON at ``path``. Only structured scores are
    persisted (never raw final-answer text), which is what keeps the
    baseline comparison in ``regression.py`` from ever degenerating into a
    string diff."""
    baseline = _to_baseline_run(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": baseline.run_id,
        "reports": [
            {
                "task_id": r.task_id,
                "risk_level": r.risk_level,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "dimension_scores": [{"name": d.name, "score": d.score, "passed": d.passed} for d in r.dimension_scores],
            }
            for r in baseline.reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline(path: Path) -> BaselineRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = tuple(
        BaselineTaskReport(
            task_id=r["task_id"],
            risk_level=r["risk_level"],
            overall_score=r["overall_score"],
            passed=r["passed"],
            cost_usd=r["cost_usd"],
            latency_ms=r["latency_ms"],
            dimension_scores=tuple(
                BaselineDimensionScore(name=d["name"], score=d["score"], passed=d["passed"])
                for d in r["dimension_scores"]
            ),
        )
        for r in payload["reports"]
    )
    return BaselineRun(run_id=payload["run_id"], reports=reports)


def baseline_from_run(run: EvaluationRun) -> BaselineRun:
    """In-memory equivalent of ``save_baseline`` + ``load_baseline`` --
    useful in tests that want a baseline snapshot without touching disk."""
    return _to_baseline_run(run)


__all__ = [
    "BaselineDimensionScore",
    "BaselineTaskReport",
    "BaselineRun",
    "save_baseline",
    "load_baseline",
    "baseline_from_run",
]
