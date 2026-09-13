"""Tests for the Orchestrator-Workers pattern: dynamic task decomposition,
parallel Workers, and a Synthesizer.

These use ``agents.testing.ScriptedModel`` to replay deterministic model output --
no network calls and no API key are required. Because Workers run *concurrently*
against one shared ``ScriptedModel`` instance, tests use ``ModelStep.respond``
(instead of positional scripted steps) so each Worker gets the right scripted
answer regardless of the non-deterministic order in which ``asyncio`` actually
resolves the concurrent coroutines -- same technique as
``tests/test_parallelization.py``.
"""

import asyncio
import json
import time
import unittest

from agents.testing import ModelStep, ScriptedModel, assistant_message
from agents.testing.model import UnconsumedModelSteps

from src.orchestrator_workers import (
    MAX_WORKERS,
    OrchestratorWorkersStepError,
    ResearchPlan,
    WorkerTask,
    _assert_within_worker_cap,
    build_orchestrator,
    build_synthesizer,
    build_worker,
    run_orchestrator_workers,
)

QUESTION = "全面分析 2026 年 AI Agent 开发技术栈。"

SYNTHESIZER_TEXT = (
    "Comprehensive 2026 AI agent stack: orchestration frameworks, model providers, and "
    "observability tooling are converging around a few dominant patterns; recommend "
    "adopting an SDK-native agent loop with pluggable model providers and first-class tracing."
)


def _plan_json(tasks: list[tuple[str, str]]) -> str:
    """Build a ResearchPlan JSON string from (task_id, objective) pairs."""
    return json.dumps(
        {"tasks": [{"task_id": tid, "objective": obj, "guidance": ""} for tid, obj in tasks]}
    )


def _findings_json(summary: str, key_findings: list[str] | None = None, sources: list[str] | None = None) -> str:
    return json.dumps(
        {
            "summary": summary,
            "key_findings": key_findings or [],
            "sources": sources or [],
        }
    )


def _worker_responder(objective_keyword: str, other_keywords: list[str], response_json: str):
    """A ``ModelStep.respond`` callback that only answers a call whose input matches
    ``objective_keyword`` and asserts no other task's keyword leaked into this Worker's input
    (proving Workers are isolated from each other's tasks)."""

    def _respond(call):
        input_text = str(call.input)
        assert objective_keyword in input_text, (
            f"expected this worker's input to mention '{objective_keyword}', got: {input_text}"
        )
        for other in other_keywords:
            assert other not in input_text, (
                f"worker for '{objective_keyword}' must not see other task's keyword '{other}'"
            )
        return [assistant_message(response_json)]

    return _respond


def _three_task_script() -> ScriptedModel:
    tasks = [
        ("task-1", "Research dominant AI agent orchestration frameworks in 2026"),
        ("task-2", "Research leading model providers and APIs for AI agents in 2026"),
        ("task-3", "Research observability and evaluation tooling for AI agents in 2026"),
    ]
    keywords = ["orchestration frameworks", "model providers", "observability"]

    steps = [[assistant_message(_plan_json(tasks))]]  # Orchestrator (always runs first, alone)
    for i, (task_id, _objective) in enumerate(tasks):
        this_keyword = keywords[i]
        other_keywords = [k for j, k in enumerate(keywords) if j != i]
        steps.append(
            ModelStep.respond(
                _worker_responder(
                    this_keyword,
                    other_keywords,
                    _findings_json(f"Findings for {task_id}", [f"{task_id} finding 1"], []),
                )
            )
        )
    steps.append([assistant_message(SYNTHESIZER_TEXT)])  # Synthesizer (always runs last, alone)
    return ScriptedModel(steps)


class AgentDefinitionTest(unittest.TestCase):
    def test_orchestrator_has_no_tools_and_structured_output(self):
        agent = build_orchestrator("placeholder-model")
        self.assertEqual(agent.name, "Orchestrator")
        self.assertEqual(agent.tools, [])
        self.assertIsNotNone(agent.output_type)

    def test_worker_is_generic_with_search_web_tool_and_structured_output(self):
        agent = build_worker("placeholder-model")
        self.assertEqual(agent.name, "Worker")
        self.assertEqual([tool.name for tool in agent.tools], ["search_web"])
        self.assertIsNotNone(agent.output_type)

    def test_synthesizer_has_no_tools_and_plain_text_output(self):
        agent = build_synthesizer("placeholder-model")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.output_type)

    def test_no_agent_declares_handoffs(self):
        for builder in (build_orchestrator, build_worker, build_synthesizer):
            agent = builder("placeholder-model")
            self.assertEqual(agent.handoffs, [], f"{agent.name} must not declare handoffs")


class WorkerCapEnforcementTest(unittest.TestCase):
    """Requirement: a hard, code-enforced ceiling that cannot be bypassed by the model."""

    def test_max_workers_is_a_small_positive_constant(self):
        self.assertEqual(MAX_WORKERS, 5)

    def test_assert_within_cap_accepts_a_plan_at_the_limit(self):
        tasks = [WorkerTask(task_id=f"t{i}", objective=f"objective {i}") for i in range(MAX_WORKERS)]
        _assert_within_worker_cap(tasks)  # must not raise

    def test_assert_within_cap_rejects_a_plan_over_the_limit(self):
        tasks = [WorkerTask(task_id=f"t{i}", objective=f"objective {i}") for i in range(MAX_WORKERS + 1)]
        with self.assertRaises(ValueError):
            _assert_within_worker_cap(tasks)

    def test_assert_within_cap_rejects_an_empty_plan(self):
        with self.assertRaises(ValueError):
            _assert_within_worker_cap([])

    def test_pipeline_refuses_to_dispatch_workers_when_orchestrator_exceeds_the_cap(self):
        """Even though the Orchestrator's JSON is well-formed, this module's own code
        must refuse to proceed once the task count exceeds MAX_WORKERS."""
        too_many_tasks = [(f"task-{i}", f"objective {i}") for i in range(MAX_WORKERS + 1)]
        model = ScriptedModel([[assistant_message(_plan_json(too_many_tasks))]])

        with self.assertRaises(OrchestratorWorkersStepError) as ctx:
            asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "orchestrator")
        # No worker or synthesizer steps should have been scripted/consumed beyond
        # the single orchestrator step -- the script had only one step, and the
        # pipeline must have failed before trying to consume any more.
        model.assert_complete()


class DynamicTaskCountTest(unittest.TestCase):
    """Requirement: task count and content are NOT hardcoded; they come from the plan."""

    def test_pipeline_runs_exactly_as_many_workers_as_the_plan_specifies_three_tasks(self):
        model = _three_task_script()

        result = asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(len(result.plan.tasks), 3)
        self.assertEqual(len(result.worker_results), 3)
        self.assertEqual(
            {r.task_id for r in result.worker_results}, {"task-1", "task-2", "task-3"}
        )
        self.assertEqual(result.final_answer, SYNTHESIZER_TEXT)
        model.assert_complete()

    def test_pipeline_runs_a_different_number_of_workers_for_a_different_plan_two_tasks(self):
        """Proves the worker count is not hardcoded to any fixed number (e.g. always 3)."""
        tasks = [
            ("task-a", "Research topic A"),
            ("task-b", "Research topic B"),
        ]
        keywords = ["topic A", "topic B"]
        steps = [[assistant_message(_plan_json(tasks))]]
        for i, (task_id, _obj) in enumerate(tasks):
            other_keywords = [k for j, k in enumerate(keywords) if j != i]
            steps.append(
                ModelStep.respond(
                    _worker_responder(
                        keywords[i], other_keywords, _findings_json(f"Findings for {task_id}")
                    )
                )
            )
        steps.append([assistant_message("Two-topic synthesis.")])
        model = ScriptedModel(steps)

        result = asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(len(result.plan.tasks), 2)
        self.assertEqual(len(result.worker_results), 2)
        model.assert_complete()

    def test_pipeline_runs_a_single_worker_when_the_plan_has_only_one_task(self):
        tasks = [("only-task", "Research the only necessary angle")]
        steps = [
            [assistant_message(_plan_json(tasks))],
            ModelStep.respond(
                _worker_responder("only necessary angle", [], _findings_json("Single finding"))
            ),
            [assistant_message("Single-task synthesis.")],
        ]
        model = ScriptedModel(steps)

        result = asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(len(result.plan.tasks), 1)
        self.assertEqual(len(result.worker_results), 1)
        model.assert_complete()


class WorkerIsolationTest(unittest.TestCase):
    def test_workers_do_not_see_each_others_task_content(self):
        """The custom responders in _three_task_script already assert this per-call;
        running the full pipeline successfully proves no cross-contamination occurred."""
        model = _three_task_script()
        result = asyncio.run(run_orchestrator_workers(QUESTION, model))
        self.assertEqual(len(result.worker_results), 3)
        model.assert_complete()


class ConcurrentDispatchTest(unittest.TestCase):
    """Workers are dispatched via asyncio.gather and genuinely overlap in wall-clock time."""

    def test_workers_run_concurrently_not_sequentially(self):
        delay_seconds = 0.15
        tasks = [(f"task-{i}", f"Research angle {i}") for i in range(3)]
        keywords = [f"angle {i}" for i in range(3)]

        async def make_slow_responder(keyword: str):
            async def _respond(call):
                await asyncio.sleep(delay_seconds)
                return [assistant_message(_findings_json(f"Findings for {keyword}"))]

            return _respond

        async def scenario():
            steps = [[assistant_message(_plan_json(tasks))]]
            for keyword in keywords:
                responder = await make_slow_responder(keyword)
                steps.append(ModelStep.respond(responder))
            steps.append([assistant_message(SYNTHESIZER_TEXT)])
            model = ScriptedModel(steps)

            started = time.perf_counter()
            result = await run_orchestrator_workers(QUESTION, model)
            elapsed = time.perf_counter() - started
            return result, elapsed

        result, elapsed = asyncio.run(scenario())

        # If workers ran sequentially, dispatching 3 would take >= 3 * delay_seconds
        # on its own. Concurrent dispatch keeps total time well under that bound.
        self.assertLess(elapsed, delay_seconds * 3)
        self.assertEqual(len(result.worker_results), 3)


class FailureReportingTest(unittest.TestCase):
    def test_orchestrator_failure_is_reported_as_orchestrator_step(self):
        model = ScriptedModel([[assistant_message("not valid json")]])

        with self.assertRaises(OrchestratorWorkersStepError) as ctx:
            asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "orchestrator")

    def test_worker_failure_is_reported_with_its_task_id(self):
        tasks = [("task-1", "Research topic A"), ("task-2", "Research topic B")]

        def make_responder(task_id: str, keyword: str, valid: bool):
            def _respond(call):
                if not valid:
                    return [assistant_message("not valid json")]
                return [assistant_message(_findings_json(f"Findings for {task_id}"))]

            return _respond

        steps = [[assistant_message(_plan_json(tasks))]]
        steps.append(ModelStep.respond(make_responder("task-1", "topic A", valid=True)))
        steps.append(ModelStep.respond(make_responder("task-2", "topic B", valid=False)))
        model = ScriptedModel(steps)

        with self.assertRaises(OrchestratorWorkersStepError) as ctx:
            asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "worker-task-2")

    def test_synthesizer_failure_is_reported_as_synthesizer_step(self):
        tasks = [("task-1", "Research topic A")]
        steps = [
            [assistant_message(_plan_json(tasks))],
            ModelStep.respond(_worker_responder("topic A", [], _findings_json("Findings"))),
            # No synthesizer step scripted -> its call will fail with no steps remaining.
        ]
        model = ScriptedModel(steps)

        with self.assertRaises(OrchestratorWorkersStepError) as ctx:
            asyncio.run(run_orchestrator_workers(QUESTION, model))

        self.assertEqual(ctx.exception.step_name, "synthesizer")

    def test_rejects_empty_question(self):
        model = ScriptedModel([[assistant_message(_plan_json([("t1", "x")]))]])
        with self.assertRaises(ValueError):
            asyncio.run(run_orchestrator_workers("   ", model))
        with self.assertRaises(UnconsumedModelSteps):
            model.assert_complete()


if __name__ == "__main__":
    unittest.main()
