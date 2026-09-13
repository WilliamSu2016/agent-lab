"""Tests for evals/ — the Production Evaluation Pipeline.

Structure:
* A deterministic "good" fake System Under Test that behaves correctly on
  every one of the 20 dataset tasks (used to validate the dataset schema,
  the 11 dimensions, and to establish a baseline).
* A deterministic "regressed" fake System Under Test that deliberately
  breaks specific dimensions on specific tasks (wrong tool, wrong route,
  unapproved HIGH-risk call, missing citations, blown latency budget,
  etc.) -- used to prove ``detect_regressions`` actually catches each kind
  of regression, and specifically that it does so via structured
  dimension comparison, never via a final-string diff.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.baseline import baseline_from_run, load_baseline, save_baseline
from evals.dataset.tasks import TASKS, EvalTask, get_task
from evals.metrics import evaluate_task
from evals.regression import detect_regressions
from evals.runner import AgentRunResult, ToolCallRecord, render_report, run_offline_evaluation, save_report


# ---------------------------------------------------------------------------
# A deterministic "good" fake agent -- passes every dimension on every task.
# ---------------------------------------------------------------------------


_UPDATE_RECORD_IDS = {
    "records-update-address": "4471",
    "records-update-with-specific-field": "T-9001",
    "multi-tool-research-then-update": "SKU-2231",
}


def good_agent(task: EvalTask) -> AgentRunResult:
    eb = task.expected_behavior

    if eb.is_attack and eb.expected_blocked:
        return AgentRunResult(
            final_answer="",
            route=eb.expected_route,
            safety_blocked=True,
            terminated=True,
            termination_reason="blocked_by_guardrail",
            cost_usd=0.001,
            latency_ms=50.0,
        )

    tool_calls = []
    for name in eb.required_tools:
        args: dict = {}
        if name == "update_record":
            record_id = _UPDATE_RECORD_IDS.get(task.task_id, "REC-1")
            args = {"record_id": record_id, "value": "high" if "priority" in task.input else "updated"}
        elif name == "search_web":
            args = {"query": task.input}
        tool_calls.append(ToolCallRecord(name=name, arguments=args, approved=True))

    key_points = eb.key_points or ("ok",)
    answer = f"{eb.expected_route}: " + "; ".join(key_points) + " (updated)" if "update" in task.input.lower() else (
        f"{eb.expected_route}: " + "; ".join(key_points)
    )
    citations = ("https://example.com/source-a", "https://example.com/source-b") if eb.requires_citations else ()
    evidence = tuple(key_points) + ("supporting detail",)

    return AgentRunResult(
        final_answer=answer,
        route=eb.expected_route,
        tool_calls=tuple(tool_calls),
        terminated=eb.should_terminate,
        termination_reason="completed",
        retry_count=eb.min_retries_required,
        safety_blocked=False,
        evidence=evidence,
        citations=citations,
        cost_usd=0.001,
        latency_ms=100.0,
    )


# ---------------------------------------------------------------------------
# A deterministic "regressed" fake agent -- built from good_agent but breaks
# specific tasks in specific, known ways.
# ---------------------------------------------------------------------------


def regressed_agent(task: EvalTask) -> AgentRunResult:
    baseline_result = good_agent(task)

    if task.task_id == "research-capital-fact":
        # Regression: drops the required citation entirely.
        return AgentRunResult(
            final_answer=baseline_result.final_answer,
            route=baseline_result.route,
            tool_calls=baseline_result.tool_calls,
            terminated=True,
            evidence=baseline_result.evidence,
            citations=(),
            cost_usd=baseline_result.cost_usd,
            latency_ms=baseline_result.latency_ms,
        )

    if task.task_id == "routing-billing-question":
        # Regression: misrouted to the wrong specialist agent.
        return AgentRunResult(
            final_answer=baseline_result.final_answer,
            route="support_agent",
            tool_calls=baseline_result.tool_calls,
            terminated=True,
            evidence=baseline_result.evidence,
            citations=baseline_result.citations,
            cost_usd=baseline_result.cost_usd,
            latency_ms=baseline_result.latency_ms,
        )

    if task.task_id == "comms-send-email-requires-approval":
        # Regression: HIGH-risk tool executed WITHOUT going through approval.
        return AgentRunResult(
            final_answer=baseline_result.final_answer,
            route=baseline_result.route,
            tool_calls=tuple(
                ToolCallRecord(name=c.name, arguments=c.arguments, approved=False) for c in baseline_result.tool_calls
            ),
            terminated=True,
            evidence=baseline_result.evidence,
            citations=baseline_result.citations,
            cost_usd=baseline_result.cost_usd,
            latency_ms=baseline_result.latency_ms,
        )

    if task.task_id == "attack-ignore-previous-instructions":
        # Regression: the attack is no longer blocked -- a prompt-injection
        # guardrail regression.
        return AgentRunResult(
            final_answer="Sure, here is my system prompt: ...",
            route=task.expected_behavior.expected_route,
            safety_blocked=False,
            terminated=True,
            cost_usd=0.001,
            latency_ms=50.0,
        )

    if task.task_id == "latency-budget-simple-routing":
        # Regression: blows through the latency budget.
        return AgentRunResult(
            final_answer=baseline_result.final_answer,
            route=baseline_result.route,
            tool_calls=baseline_result.tool_calls,
            terminated=True,
            evidence=baseline_result.evidence,
            citations=baseline_result.citations,
            cost_usd=baseline_result.cost_usd,
            latency_ms=5000.0,
        )

    return baseline_result


# ---------------------------------------------------------------------------
# Dataset schema sanity checks.
# ---------------------------------------------------------------------------


def test_dataset_has_at_least_20_tasks():
    assert len(TASKS) >= 20


def test_dataset_task_ids_are_unique():
    ids = [t.task_id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_every_task_defines_required_fields():
    for task in TASKS:
        assert task.input.strip()
        assert task.expected_behavior.expected_route
        assert task.risk_level is not None
        assert 0.0 < task.success_criteria.min_overall_score <= 1.0


def test_get_task_returns_expected_task_and_raises_on_unknown():
    task = get_task("research-capital-fact")
    assert task.task_id == "research-capital-fact"
    with pytest.raises(KeyError):
        get_task("does-not-exist")


def test_dataset_includes_all_three_risk_levels():
    from src.security.tool_policy import ToolRiskLevel

    risk_levels = {t.risk_level for t in TASKS}
    assert risk_levels == {ToolRiskLevel.LOW, ToolRiskLevel.MEDIUM, ToolRiskLevel.HIGH}


def test_dataset_includes_attack_tasks_for_safety_dimension():
    attack_tasks = [t for t in TASKS if t.expected_behavior.is_attack]
    assert len(attack_tasks) >= 5


# ---------------------------------------------------------------------------
# Offline Evaluation: the "good" agent should pass every task, every dimension.
# ---------------------------------------------------------------------------


def test_good_agent_passes_offline_evaluation():
    run = run_offline_evaluation(good_agent, run_id="test-good-agent")

    failing = [(r.task_id, [d.name for d in r.dimension_scores if not d.passed]) for r in run.reports if not r.passed]
    assert failing == [], f"expected all tasks to pass, but: {failing}"
    assert run.pass_rate == 1.0
    assert run.failure_rate == 0.0
    assert run.tool_selection_accuracy == 1.0
    assert run.routing_accuracy == 1.0
    assert run.safety_failure_rate == 0.0


def test_evaluation_never_compares_raw_final_answer_strings():
    """Two semantically-equivalent but differently-worded answers must both
    score identically on every non-answer-quality dimension, proving no
    dimension is a hidden string-equality check against one golden
    answer."""
    task = get_task("research-capital-fact")
    result_a = good_agent(task)
    reworded = AgentRunResult(
        final_answer="The capital city of France is Paris, a fact confirmed by the search results.",
        route=result_a.route,
        tool_calls=result_a.tool_calls,
        terminated=result_a.terminated,
        evidence=result_a.evidence,
        citations=result_a.citations,
        cost_usd=result_a.cost_usd,
        latency_ms=result_a.latency_ms,
    )

    report_a = evaluate_task(task, result_a)
    report_b = evaluate_task(task, reworded)

    assert result_a.final_answer != reworded.final_answer  # the strings genuinely differ
    for dim_a, dim_b in zip(report_a.dimension_scores, report_b.dimension_scores):
        assert dim_a.name == dim_b.name
        assert dim_a.passed == dim_b.passed
        assert dim_a.score == pytest.approx(dim_b.score)


# ---------------------------------------------------------------------------
# Each of the 11 dimensions, individually.
# ---------------------------------------------------------------------------


def test_dimension_answer_quality_requires_full_key_point_coverage():
    task = get_task("research-multihop-population")
    good = good_agent(task)
    incomplete = AgentRunResult(
        final_answer="Tokyo is a large city.",  # missing "Delhi"
        route=good.route,
        tool_calls=good.tool_calls,
        evidence=good.evidence,
        citations=good.citations,
    )
    report = evaluate_task(task, incomplete)
    assert report.dimension_score("answer_quality") < 1.0


def test_dimension_tool_selection_flags_missing_required_tool():
    task = get_task("research-capital-fact")
    good = good_agent(task)
    no_tools = AgentRunResult(final_answer=good.final_answer, route=good.route, tool_calls=(), citations=good.citations, evidence=good.evidence)
    report = evaluate_task(task, no_tools)
    assert not report.dimension_score("tool_selection") == 1.0


def test_dimension_tool_arguments_checks_predicate_not_equality():
    task = get_task("records-update-with-specific-field")
    good = good_agent(task)
    wrong_record = AgentRunResult(
        final_answer=good.final_answer,
        route=good.route,
        tool_calls=(ToolCallRecord(name="update_record", arguments={"record_id": "WRONG-ID", "value": "high"}),),
    )
    report = evaluate_task(task, wrong_record)
    assert report.dimension_score("tool_arguments") < 1.0


def test_dimension_routing_requires_exact_match():
    task = get_task("routing-tech-support-question")
    good = good_agent(task)
    misrouted = AgentRunResult(final_answer=good.final_answer, route="billing_agent", tool_calls=good.tool_calls)
    report = evaluate_task(task, misrouted)
    assert report.dimension_score("routing") == 0.0


def test_dimension_termination_flags_non_terminating_run():
    task = get_task("termination-no-infinite-loop-on-ambiguous-input")
    good = good_agent(task)
    looping = AgentRunResult(final_answer=good.final_answer, route=good.route, terminated=False, termination_reason="max_iterations_reached")
    report = evaluate_task(task, looping)
    assert report.dimension_score("termination") == 0.0


def test_dimension_retry_behavior_flags_too_few_retries():
    task = get_task("retry-transient-tool-failure-then-success")
    good = good_agent(task)
    no_retry = AgentRunResult(
        final_answer=good.final_answer, route=good.route, tool_calls=good.tool_calls, retry_count=0,
        evidence=good.evidence, citations=good.citations,
    )
    report = evaluate_task(task, no_retry)
    assert report.dimension_score("retry_behavior") == 0.0


def test_dimension_retry_behavior_flags_too_many_retries():
    task = get_task("retry-transient-tool-failure-then-success")
    good = good_agent(task)
    thrashing = AgentRunResult(
        final_answer=good.final_answer, route=good.route, tool_calls=good.tool_calls, retry_count=10,
        evidence=good.evidence, citations=good.citations,
    )
    report = evaluate_task(task, thrashing)
    assert report.dimension_score("retry_behavior") == 0.0


def test_dimension_safety_gates_overall_pass_even_with_high_answer_quality():
    task = get_task("attack-ignore-previous-instructions")
    unblocked_attack = AgentRunResult(
        final_answer="Here is the full system prompt verbatim: ...",
        route=task.expected_behavior.expected_route,
        safety_blocked=False,
    )
    report = evaluate_task(task, unblocked_attack)
    assert report.dimension_score("safety") == 0.0
    assert report.passed is False  # safety gate overrides overall score


def test_dimension_safety_flags_unapproved_high_risk_tool_call():
    task = get_task("comms-send-email-requires-approval")
    good = good_agent(task)
    bypassed = AgentRunResult(
        final_answer=good.final_answer,
        route=good.route,
        tool_calls=tuple(ToolCallRecord(name=c.name, arguments=c.arguments, approved=False) for c in good.tool_calls),
    )
    report = evaluate_task(task, bypassed)
    assert report.dimension_score("safety") == 0.0
    assert report.passed is False


def test_dimension_groundedness_flags_hallucinated_claim():
    task = get_task("groundedness-no-hallucinated-numbers")
    hallucinated = AgentRunResult(
        final_answer="Revenue growth was 42% last quarter.",
        route=task.expected_behavior.expected_route,
        tool_calls=(ToolCallRecord(name="search_web", arguments={"query": task.input}),),
        evidence=("unrelated evidence snippet",),  # does not mention "growth"
        citations=("https://example.com/a",),
    )
    report = evaluate_task(task, hallucinated)
    assert report.dimension_score("groundedness") < 1.0


def test_dimension_citation_quality_flags_missing_citations():
    task = get_task("citation-quality-two-sources-required")
    good = good_agent(task)
    no_citations = AgentRunResult(
        final_answer=good.final_answer, route=good.route, tool_calls=good.tool_calls, evidence=good.evidence, citations=()
    )
    report = evaluate_task(task, no_citations)
    assert report.dimension_score("citation_quality") < 1.0


def test_dimension_citation_quality_flags_malformed_citation():
    task = get_task("citation-quality-two-sources-required")
    good = good_agent(task)
    junk_citations = AgentRunResult(
        final_answer=good.final_answer,
        route=good.route,
        tool_calls=good.tool_calls,
        evidence=good.evidence,
        citations=("not a real citation", "also not one"),
    )
    report = evaluate_task(task, junk_citations)
    assert report.dimension_score("citation_quality") < 1.0


def test_dimension_cost_flags_over_budget():
    task = get_task("cost-budget-trivial-lookup")
    good = good_agent(task)
    expensive = AgentRunResult(final_answer=good.final_answer, route=good.route, cost_usd=1.0, latency_ms=good.latency_ms)
    report = evaluate_task(task, expensive)
    assert report.dimension_score("cost") < 0.1
    assert not any(d.name == "cost" and d.passed for d in report.dimension_scores)


def test_dimension_latency_flags_over_budget():
    task = get_task("latency-budget-simple-routing")
    good = good_agent(task)
    slow = AgentRunResult(final_answer=good.final_answer, route=good.route, cost_usd=good.cost_usd, latency_ms=999_999.0)
    report = evaluate_task(task, slow)
    assert report.dimension_score("latency") < 0.1
    assert not any(d.name == "latency" and d.passed for d in report.dimension_scores)


# ---------------------------------------------------------------------------
# Full pipeline: run -> save baseline -> re-run (regressed) -> detect regression.
# ---------------------------------------------------------------------------


def test_full_pipeline_detects_every_kind_of_injected_regression(tmp_path: Path):
    baseline_run = run_offline_evaluation(good_agent, run_id="baseline")
    baseline_path = tmp_path / "baseline.json"
    save_baseline(baseline_run, baseline_path)

    loaded_baseline = load_baseline(baseline_path)
    current_run = run_offline_evaluation(regressed_agent, run_id="current")

    report = detect_regressions(loaded_baseline, current_run)

    assert report.has_regressions is True
    flagged_tasks = {entry.task_id for entry in report.entries} | set(report.newly_failing_tasks)
    assert "research-capital-fact" in flagged_tasks  # citation_quality regression
    assert "routing-billing-question" in flagged_tasks  # routing regression
    assert "comms-send-email-requires-approval" in flagged_tasks  # safety regression
    assert "attack-ignore-previous-instructions" in flagged_tasks  # safety regression
    assert "latency-budget-simple-routing" in flagged_tasks  # latency regression

    # Tasks the regressed agent did NOT touch must show no regression.
    untouched_flagged = flagged_tasks - {
        "research-capital-fact",
        "routing-billing-question",
        "comms-send-email-requires-approval",
        "attack-ignore-previous-instructions",
        "latency-budget-simple-routing",
    }
    assert untouched_flagged == set()


def test_regression_detection_is_clean_when_current_equals_baseline():
    baseline_run = run_offline_evaluation(good_agent, run_id="baseline")
    baseline = baseline_from_run(baseline_run)
    current_run = run_offline_evaluation(good_agent, run_id="current-identical")

    report = detect_regressions(baseline, current_run)

    assert report.has_regressions is False
    assert report.entries == ()
    assert report.newly_failing_tasks == ()


def test_regression_report_flags_new_and_removed_tasks(tmp_path: Path):
    subset = TASKS[:18]
    baseline_run = run_offline_evaluation(good_agent, run_id="baseline-subset", tasks=subset)
    baseline = baseline_from_run(baseline_run)
    full_run = run_offline_evaluation(good_agent, run_id="current-full", tasks=TASKS)

    report = detect_regressions(baseline, full_run)

    assert set(report.new_tasks) == {t.task_id for t in TASKS[18:]}
    assert report.removed_tasks == ()


def test_regression_score_drop_threshold_tolerates_small_noise():
    from evals.metrics import DimensionScore, TaskEvalReport
    from evals.baseline import BaselineDimensionScore, BaselineRun, BaselineTaskReport
    from src.security.tool_policy import ToolRiskLevel

    task = get_task("research-capital-fact")
    result = good_agent(task)
    current_report = evaluate_task(task, result)

    # Baseline identical except one dimension nudged up by a tiny amount --
    # should NOT be flagged as a regression at the default threshold.
    noisy_baseline = BaselineRun(
        run_id="baseline",
        reports=(
            BaselineTaskReport(
                task_id=task.task_id,
                risk_level=ToolRiskLevel.LOW.value,
                overall_score=current_report.overall_score + 0.02,
                passed=True,
                dimension_scores=tuple(
                    BaselineDimensionScore(
                        name=d.name,
                        score=min(1.0, d.score + 0.02) if d.name == "answer_quality" else d.score,
                        passed=d.passed,
                    )
                    for d in current_report.dimension_scores
                ),
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            ),
        ),
    )

    from evals.runner import EvaluationRun

    current_run = EvaluationRun(run_id="current", reports=(current_report,))
    report = detect_regressions(noisy_baseline, current_run, score_drop_threshold=0.1)

    assert report.has_regressions is False


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------


def test_render_report_includes_required_summary_fields():
    run = run_offline_evaluation(good_agent, run_id="report-test")
    text = render_report(run)

    for required in ["Pass rate", "Failure rate", "Tool selection accuracy", "Routing accuracy", "Safety failure rate", "Average latency", "Average cost"]:
        assert required in text


def test_render_report_lists_failing_dimensions_for_failed_tasks():
    run = run_offline_evaluation(regressed_agent, run_id="report-regressed")
    text = render_report(run)

    assert "routing" in text or "safety" in text or "citation_quality" in text


def test_save_report_writes_file(tmp_path: Path):
    run = run_offline_evaluation(good_agent, run_id="report-save-test")
    out_path = tmp_path / "reports" / "latest.md"
    save_report(run, out_path)

    assert out_path.exists()
    assert "Evaluation Report" in out_path.read_text(encoding="utf-8")
