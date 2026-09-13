"""Tests for the Evaluator-Optimizer pattern: Generator drafts a report, Evaluator
critiques it, and the loop revises up to a hard iteration cap or until the score
clears a fixed bar.

These use ``agents.testing.ScriptedModel`` to replay deterministic model output --
no network calls and no API key are required. The loop is strictly sequential
(Generator -> Evaluator -> maybe Generator again -> ...), so positional scripted
steps (as in tests/test_prompt_chaining.py and tests/test_routing.py) are
sufficient -- there is no concurrent dispatch here, unlike Parallelization or
Orchestrator-Workers.
"""

import asyncio
import json
import unittest

from agents.testing import ModelStep, ScriptedModel, assistant_message
from agents.testing.model import UnconsumedModelSteps

from src.evaluator_optimizer import (
    MAX_ITERATIONS,
    PASS_SCORE,
    EvaluationResult,
    EvaluatorOptimizerStepError,
    build_evaluator,
    build_generator,
    run_evaluator_optimizer,
)

TOPIC = "Compare the leading AI Agent frameworks for building a production research agent."


def _evaluation_json(passed: bool, score: int, feedback=None, missing_points=None) -> str:
    return json.dumps(
        {
            "pass": passed,
            "score": score,
            "feedback": feedback or [],
            "missing_points": missing_points or [],
        }
    )


class AgentDefinitionTest(unittest.TestCase):
    def test_generator_has_search_web_tool_and_plain_text_output(self):
        agent = build_generator("placeholder-model")
        self.assertEqual(agent.name, "Generator")
        self.assertEqual([tool.name for tool in agent.tools], ["search_web"])
        self.assertIsNone(agent.output_type)

    def test_evaluator_has_no_tools_and_structured_output(self):
        agent = build_evaluator("placeholder-model")
        self.assertEqual(agent.name, "Evaluator")
        self.assertEqual(agent.tools, [])
        self.assertIsNotNone(agent.output_type)

    def test_no_agent_declares_handoffs(self):
        for builder in (build_generator, build_evaluator):
            agent = builder("placeholder-model")
            self.assertEqual(agent.handoffs, [], f"{agent.name} must not declare handoffs")


class EvaluationResultSchemaTest(unittest.TestCase):
    """The wire format uses the literal JSON key "pass" (as requested), exposed in
    Python as ``passed`` since ``pass`` is a reserved keyword."""

    def test_parses_pass_key_from_json(self):
        result = EvaluationResult.model_validate_json(_evaluation_json(True, 9, ["ok"], []))
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 9)

    def test_can_also_be_constructed_with_passed_keyword(self):
        result = EvaluationResult(passed=False, score=4, feedback=[], missing_points=[])
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 4)

    def test_score_out_of_range_is_rejected(self):
        with self.assertRaises(Exception):
            EvaluationResult(passed=True, score=11, feedback=[], missing_points=[])
        with self.assertRaises(Exception):
            EvaluationResult(passed=True, score=-1, feedback=[], missing_points=[])


class StopsWhenPassedThresholdTest(unittest.TestCase):
    def test_stops_after_one_iteration_when_score_meets_the_bar(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1 of the comparison report.")],
                [assistant_message(_evaluation_json(True, 9, [], []))],
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.stopped_reason, "passed_threshold")
        self.assertEqual(result.best_iteration, 1)
        self.assertEqual(result.final_report, "Draft v1 of the comparison report.")
        model.assert_complete()

    def test_score_exactly_at_pass_threshold_is_accepted(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                [assistant_message(_evaluation_json(True, PASS_SCORE, [], []))],
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(result.stopped_reason, "passed_threshold")
        self.assertEqual(len(result.iterations), 1)


class MaxIterationsCapTest(unittest.TestCase):
    def test_runs_exactly_max_iterations_when_score_never_clears_the_bar(self):
        steps = []
        for i in range(MAX_ITERATIONS):
            steps.append([assistant_message(f"Draft v{i + 1}.")])
            steps.append([assistant_message(_evaluation_json(False, 5, [f"issue {i + 1}"], []))])
        model = ScriptedModel(steps)

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(len(result.iterations), MAX_ITERATIONS)
        self.assertEqual(result.stopped_reason, "max_iterations_reached")
        model.assert_complete()

    def test_max_iterations_constant_is_three_by_default(self):
        self.assertEqual(MAX_ITERATIONS, 3)


class BestIterationSelectionTest(unittest.TestCase):
    """Requirement: the final report is the BEST-scoring draft, not necessarily
    the last one -- even if quality regresses or plateaus across iterations."""

    def test_final_report_is_the_highest_scoring_draft_even_if_not_last(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                [assistant_message(_evaluation_json(False, 5, ["fix A"], ["dim A"]))],
                [assistant_message("Draft v2.")],
                [assistant_message(_evaluation_json(False, 7, ["fix B"], ["dim B"]))],
                [assistant_message("Draft v3 (regressed).")],
                [assistant_message(_evaluation_json(False, 6, ["fix C"], ["dim C"]))],
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(len(result.iterations), 3)
        self.assertEqual(result.stopped_reason, "max_iterations_reached")
        # iteration 2 (score 7) beats iterations 1 (5) and 3 (6).
        self.assertEqual(result.best_iteration, 2)
        self.assertEqual(result.final_report, "Draft v2.")
        self.assertEqual([r.evaluation.score for r in result.iterations], [5, 7, 6])


class FeedbackFlowsIntoNextGeneratorCallTest(unittest.TestCase):
    def test_second_generator_call_receives_previous_draft_and_feedback(self):
        def _first_generator(_call):
            return [assistant_message("Draft v1 lacking cost comparison.")]

        def _first_evaluator(_call):
            return [
                assistant_message(
                    _evaluation_json(
                        False, 4, ["Add a cost comparison section"], ["pricing dimension"]
                    )
                )
            ]

        def _second_generator(call):
            input_text = str(call.input)
            assert "Draft v1 lacking cost comparison." in input_text
            assert "Add a cost comparison section" in input_text
            assert "pricing dimension" in input_text
            return [assistant_message("Draft v2 with cost comparison added.")]

        def _second_evaluator(_call):
            return [assistant_message(_evaluation_json(True, 9, [], []))]

        model = ScriptedModel(
            [
                ModelStep.respond(_first_generator),
                ModelStep.respond(_first_evaluator),
                ModelStep.respond(_second_generator),
                ModelStep.respond(_second_evaluator),
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(len(result.iterations), 2)
        self.assertEqual(result.stopped_reason, "passed_threshold")
        self.assertEqual(result.final_report, "Draft v2 with cost comparison added.")
        model.assert_complete()


class ScorePassMismatchTest(unittest.TestCase):
    """Requirement: the code, not the Evaluator's own "passed" flag, decides
    accept/reject. A disagreement is recorded, never silently trusted."""

    def test_low_score_with_passed_true_is_not_accepted(self):
        steps = []
        # Evaluator claims pass=True but gives a low score on every iteration --
        # the code must never accept based on the flag alone.
        for i in range(MAX_ITERATIONS):
            steps.append([assistant_message(f"Draft v{i + 1}.")])
            steps.append([assistant_message(_evaluation_json(True, 5, ["still weak"], []))])
        model = ScriptedModel(steps)

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(result.stopped_reason, "max_iterations_reached")
        for record in result.iterations:
            self.assertFalse(record.accepted)
            self.assertTrue(record.score_pass_mismatch)

    def test_high_score_with_passed_false_is_still_accepted(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                [assistant_message(_evaluation_json(False, 9, ["nitpick"], []))],
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(result.stopped_reason, "passed_threshold")
        self.assertTrue(result.iterations[0].accepted)
        self.assertTrue(result.iterations[0].score_pass_mismatch)

    def test_matching_score_and_pass_flag_has_no_mismatch(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                [assistant_message(_evaluation_json(True, 9, [], []))],
            ]
        )

        result = asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertFalse(result.iterations[0].score_pass_mismatch)


class FailureReportingTest(unittest.TestCase):
    def test_generator_failure_is_reported_with_its_iteration(self):
        # Iteration 1 completes normally with a low score, triggering a revision.
        # No script is provided for iteration 2's Generator call, so it fails with
        # no steps left -- and must be reported as "generator-iteration-2", not
        # blamed on the Evaluator or left unattributed.
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                [assistant_message(_evaluation_json(False, 5, ["issue"], []))],
            ]
        )

        with self.assertRaises(EvaluatorOptimizerStepError) as ctx:
            asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(ctx.exception.step_name, "generator-iteration-2")

    def test_evaluator_failure_is_reported_with_its_iteration(self):
        model = ScriptedModel(
            [
                [assistant_message("Draft v1.")],
                # No evaluator step scripted -> this call fails with no steps left.
            ]
        )

        with self.assertRaises(EvaluatorOptimizerStepError) as ctx:
            asyncio.run(run_evaluator_optimizer(TOPIC, model))

        self.assertEqual(ctx.exception.step_name, "evaluator-iteration-1")

    def test_rejects_empty_topic(self):
        model = ScriptedModel([[assistant_message("unused")]])
        with self.assertRaises(ValueError):
            asyncio.run(run_evaluator_optimizer("   ", model))
        with self.assertRaises(UnconsumedModelSteps):
            model.assert_complete()


if __name__ == "__main__":
    unittest.main()
