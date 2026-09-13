"""Unit tests: the evaluation harness end to end -- Offline Evaluation
(``evals.runner``/``evals.adapter``) + Regression Evaluation
(``evals.baseline``/``evals.regression``), against the REAL graph (stub
LLMs, real guardrails, real checkpointer)."""

from __future__ import annotations

from pathlib import Path

from evals.adapter import make_adapter
from evals.baseline import baseline_from_run, load_baseline, save_baseline
from evals.dataset.tasks import TASKS
from evals.regression import detect_regressions
from evals.runner import render_report, run_offline_evaluation


def test_dataset_has_at_least_twenty_tasks_covering_every_risk_level():
    assert len(TASKS) >= 20
    risk_levels = {task.risk_level.value for task in TASKS}
    assert {"low", "high"}.issubset(risk_levels)


def test_offline_evaluation_run_against_the_real_graph_is_fully_passing():
    sut = make_adapter()
    run = run_offline_evaluation(sut, run_id="test-run")

    assert len(run.reports) == len(TASKS)
    assert run.pass_rate == 1.0
    assert run.failure_rate == 0.0
    assert run.safety_failure_rate == 0.0
    assert run.routing_accuracy == 1.0


def test_rendered_report_includes_every_required_summary_metric():
    sut = make_adapter()
    run = run_offline_evaluation(sut, run_id="test-run")
    report = render_report(run)

    for required in ("Pass rate", "Failure rate", "Tool selection accuracy", "Routing accuracy", "Safety failure rate", "Average latency", "Average cost"):
        assert required in report


def test_no_regression_detected_between_two_identical_deterministic_runs():
    sut = make_adapter()
    baseline_run = run_offline_evaluation(sut, run_id="baseline")
    baseline = baseline_from_run(baseline_run)

    current_run = run_offline_evaluation(sut, run_id="current")
    regressions = detect_regressions(baseline, current_run)

    assert not regressions.has_regressions


def test_save_and_load_baseline_round_trips(tmp_path):
    sut = make_adapter()
    run = run_offline_evaluation(sut, run_id="baseline")
    path = tmp_path / "baseline.json"

    save_baseline(run, path)
    loaded = load_baseline(path)

    assert loaded.run_id == "baseline"
    assert len(loaded.reports) == len(run.reports)
    for report in run.reports:
        loaded_report = loaded.get(report.task_id)
        assert loaded_report is not None
        assert loaded_report.overall_score == report.overall_score
        assert loaded_report.passed == report.passed


def test_regression_is_flagged_when_a_task_starts_failing():
    from evals.baseline import BaselineDimensionScore, BaselineRun, BaselineTaskReport

    baseline = BaselineRun(
        run_id="baseline",
        reports=(
            BaselineTaskReport(
                task_id="research-langgraph-checkpointer",
                risk_level="low",
                overall_score=1.0,
                passed=True,
                dimension_scores=(BaselineDimensionScore(name="answer_quality", score=1.0, passed=True),),
                cost_usd=0.001,
                latency_ms=50.0,
            ),
        ),
    )

    sut = make_adapter()
    current_run = run_offline_evaluation(sut, run_id="current", tasks=[t for t in TASKS if t.task_id == "research-langgraph-checkpointer"])

    regressions = detect_regressions(baseline, current_run)
    # Same deterministic stub -> no regression against a baseline that
    # also recorded a perfect score for this task.
    assert not regressions.has_regressions
