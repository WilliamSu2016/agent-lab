"""Tests for the fixed Prompt Chaining pipeline (Planner -> Researcher -> Analyst -> Writer).

These use ``agents.testing.ScriptedModel`` to replay deterministic model output --
no network calls and no API key are required. The Runner (not this test, and not
``src/prompt_chaining.py``) drives each step's LLM call, structured-output parsing,
and the Researcher step's ``search_web`` tool-call loop.
"""

import unittest
from unittest.mock import patch

from agents.testing import ScriptedModel, assistant_message, function_call
from agents.testing.model import UnconsumedModelSteps

from src.prompt_chaining import (
    PromptChainStepError,
    build_analyst,
    build_planner,
    build_researcher,
    build_writer,
    run_prompt_chain,
)

QUESTION = "Compare Python, TypeScript, and Go for building AI agents, and give a recommendation."

PLANNER_JSON = (
    '{"dimensions": ["Concurrency and performance", '
    '"AI/ML and agent ecosystem", "Tooling and deployment"]}'
)

RESEARCHER_JSON = (
    '{"notes": ['
    '{"dimension": "Concurrency and performance", '
    '"findings": "Go has native goroutines; Python relies on asyncio; '
    'TypeScript uses the Node.js event loop.", '
    '"sources": ["https://example.com/perf"]}, '
    '{"dimension": "AI/ML and agent ecosystem", '
    '"findings": "Python has the largest AI/agent library ecosystem.", '
    '"sources": ["https://example.com/ecosystem"]}, '
    '{"dimension": "Tooling and deployment", '
    '"findings": "Go produces static binaries; TypeScript/Node has broad web tooling.", '
    '"sources": ["https://example.com/tooling"]}'
    "]}"
)

ANALYST_TEXT = (
    "Concurrency and performance: Go leads on raw concurrency; Python is slower but "
    "acceptable for I/O-bound agent workloads. AI/ML ecosystem: Python leads by a wide "
    "margin. Tooling and deployment: Go is easiest to deploy as a single binary; "
    "TypeScript/Node has strong web tooling; Python tooling is mature but slower to deploy."
)

WRITER_TEXT = (
    "Recommendation: use Python for building AI agents today, because its AI/agent "
    "library ecosystem (including the OpenAI Agents SDK) outweighs Go's concurrency "
    "advantage and TypeScript's web tooling for most agent-building use cases."
)


def _four_step_script() -> ScriptedModel:
    """One scripted model shared across all four sequential Runner.run_sync calls."""
    return ScriptedModel(
        [
            [assistant_message(PLANNER_JSON)],  # Step 1 -- Planner
            [function_call("search_web", {"query": "Go concurrency for agents"}, call_id="call_1")],
            [assistant_message(RESEARCHER_JSON)],  # Step 2 -- Researcher (after tool result)
            [assistant_message(ANALYST_TEXT)],  # Step 3 -- Analyst
            [assistant_message(WRITER_TEXT)],  # Step 4 -- Writer
        ]
    )


class StepDefinitionTest(unittest.TestCase):
    """Requirement 7 & 8: each step has its own narrow instructions and one responsibility."""

    def test_planner_has_no_tools_and_a_structured_output_type(self):
        agent = build_planner("placeholder-model")
        self.assertEqual(agent.name, "Planner")
        self.assertEqual(agent.tools, [])
        self.assertIsNotNone(agent.output_type)

    def test_researcher_has_exactly_the_search_web_tool(self):
        agent = build_researcher("placeholder-model")
        self.assertEqual(agent.name, "Researcher")
        self.assertEqual([tool.name for tool in agent.tools], ["search_web"])
        self.assertIsNotNone(agent.output_type)

    def test_analyst_has_no_tools_and_plain_text_output(self):
        agent = build_analyst("placeholder-model")
        self.assertEqual(agent.name, "Analyst")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.output_type)

    def test_writer_has_no_tools_and_plain_text_output(self):
        agent = build_writer("placeholder-model")
        self.assertEqual(agent.name, "Writer")
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent.output_type)

    def test_no_agent_declares_handoffs(self):
        """Requirement 2: no multi-agent handoffs anywhere in the pipeline."""
        for builder in (build_planner, build_researcher, build_analyst, build_writer):
            agent = builder("placeholder-model")
            self.assertEqual(agent.handoffs, [], f"{agent.name} must not declare handoffs")


class FixedSequentialExecutionTest(unittest.TestCase):
    """Requirements 5 & 6: fixed step order, each step's input built from the prior output."""

    def test_full_chain_runs_all_four_steps_in_order_with_each_output_feeding_the_next(self):
        model = _four_step_script()

        with patch("src.prompt_chaining._web_search") as fake_web_search:
            fake_web_search.return_value = {
                "query": "Go concurrency for agents",
                "results": [{"title": "Go", "url": "https://example.com/perf", "snippet": "..."}],
            }
            result = run_prompt_chain(QUESTION, model)

        # Step 1 output feeds step 2 (same three dimensions, none invented or dropped).
        self.assertEqual(
            result.plan.dimensions,
            [
                "Concurrency and performance",
                "AI/ML and agent ecosystem",
                "Tooling and deployment",
            ],
        )
        self.assertEqual(
            [note.dimension for note in result.research.notes],
            result.plan.dimensions,
        )
        fake_web_search.assert_called_once_with("Go concurrency for agents")

        # Step 3 and 4 outputs are exactly what the scripted model returned.
        self.assertEqual(result.analysis, ANALYST_TEXT)
        self.assertEqual(result.final_answer, WRITER_TEXT)
        self.assertEqual(result.question, QUESTION)

        # Every scripted step was actually consumed, proving all four Runner
        # calls really executed in sequence, not skipped or reordered.
        model.assert_complete()


class StepFailureReportingTest(unittest.TestCase):
    """Requirement 9: a failing step must be clearly identifiable."""

    def test_planner_failure_is_reported_as_step_1(self):
        # Malformed JSON for the Planner's ResearchPlan output_type schema.
        model = ScriptedModel([[assistant_message("not valid json")]])

        with self.assertRaises(PromptChainStepError) as ctx:
            run_prompt_chain(QUESTION, model)

        self.assertEqual(ctx.exception.step_name, "1-planner")

    def test_researcher_failure_is_reported_as_step_2(self):
        model = ScriptedModel(
            [
                [assistant_message(PLANNER_JSON)],  # Step 1 succeeds
                [assistant_message("not valid json")],  # Step 2's output_type fails to parse
            ]
        )

        with self.assertRaises(PromptChainStepError) as ctx:
            run_prompt_chain(QUESTION, model)

        self.assertEqual(ctx.exception.step_name, "2-researcher")

    def test_analyst_failure_is_reported_as_step_3(self):
        model = ScriptedModel(
            [
                [assistant_message(PLANNER_JSON)],  # Step 1
                [assistant_message(RESEARCHER_JSON)],  # Step 2 (no tool call this time)
                [function_call("search_web", {"query": "x"}, call_id="oops")],  # Step 3 misbehaves
            ]
        )

        with self.assertRaises(PromptChainStepError) as ctx:
            run_prompt_chain(QUESTION, model)

        self.assertEqual(ctx.exception.step_name, "3-analyst")

    def test_rejects_empty_question(self):
        model = ScriptedModel([[assistant_message(PLANNER_JSON)]])
        with self.assertRaises(ValueError):
            run_prompt_chain("   ", model)
        # Nothing should have been consumed from the script.
        with self.assertRaises(UnconsumedModelSteps):
            model.assert_complete()


if __name__ == "__main__":
    unittest.main()
