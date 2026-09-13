"""Tests for Experiment 3 (Supervisor / Agents-as-Tools): src/03_supervisor.py.

Uses the SDK's own public testing helper, ``agents.testing.ScriptedModel``, to
replay deterministic model output per agent -- no network calls and no API key
are required. ``Agent.as_tool()`` (not this test file) performs the nested
agent run when SupervisorAgent's model calls ``research_expert`` /
``coding_expert`` / ``review_expert``.

The module under test is named ``03_supervisor.py`` (starts with a digit), so
it is loaded dynamically with ``importlib`` -- same pattern as
``tests/test_handoff.py`` for ``02_handoff.py``.
"""

import importlib
import unittest

from agents.items import HandoffCallItem, ToolCallItem, ToolCallOutputItem
from agents.testing import ScriptedModel, assistant_message, function_call

supervisor_mod = importlib.import_module("src.03_supervisor")

SUPERVISOR_AGENT_NAME = supervisor_mod.SUPERVISOR_AGENT_NAME
RESEARCH_AGENT_NAME = supervisor_mod.RESEARCH_AGENT_NAME
CODING_AGENT_NAME = supervisor_mod.CODING_AGENT_NAME
REVIEW_AGENT_NAME = supervisor_mod.REVIEW_AGENT_NAME
RESEARCH_TOOL_NAME = supervisor_mod.RESEARCH_TOOL_NAME
CODING_TOOL_NAME = supervisor_mod.CODING_TOOL_NAME
REVIEW_TOOL_NAME = supervisor_mod.REVIEW_TOOL_NAME
build_research_agent = supervisor_mod.build_research_agent
build_coding_agent = supervisor_mod.build_coding_agent
build_review_agent = supervisor_mod.build_review_agent
build_supervisor_agent = supervisor_mod.build_supervisor_agent
build_agents = supervisor_mod.build_agents
run_supervisor = supervisor_mod.run_supervisor
extract_expert_tool_calls = supervisor_mod.extract_expert_tool_calls
extract_expert_tool_results = supervisor_mod.extract_expert_tool_results

QUESTION = "比较 LangGraph 和 OpenAI Agents SDK，并给出推荐。"


class AgentWiringTest(unittest.TestCase):
    """Structural checks: who has tools, who has handoffs, tool naming."""

    def test_specialists_have_no_tools_or_handoffs_of_their_own(self):
        research_agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))
        coding_agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))
        review_agent = build_review_agent(ScriptedModel([[assistant_message("x")]]))

        for agent in (research_agent, coding_agent, review_agent):
            self.assertEqual(agent.tools, [])
            self.assertEqual(agent.handoffs, [])

    def test_supervisor_has_exactly_the_three_expert_tools_as_function_tools(self):
        research_agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))
        coding_agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))
        review_agent = build_review_agent(ScriptedModel([[assistant_message("x")]]))
        supervisor = build_supervisor_agent(
            ScriptedModel([[assistant_message("x")]]),
            research_agent=research_agent,
            coding_agent=coding_agent,
            review_agent=review_agent,
        )

        self.assertEqual(supervisor.name, SUPERVISOR_AGENT_NAME)
        tool_names = {tool.name for tool in supervisor.tools}
        self.assertEqual(tool_names, {RESEARCH_TOOL_NAME, CODING_TOOL_NAME, REVIEW_TOOL_NAME})

    def test_supervisor_has_no_handoffs_requirement_7(self):
        # Requirement 7: this experiment must not use Handoff at all.
        research_agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))
        coding_agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))
        review_agent = build_review_agent(ScriptedModel([[assistant_message("x")]]))
        supervisor = build_supervisor_agent(
            ScriptedModel([[assistant_message("x")]]),
            research_agent=research_agent,
            coding_agent=coding_agent,
            review_agent=review_agent,
        )

        self.assertEqual(supervisor.handoffs, [])


class SupervisorStaysInControlTest(unittest.TestCase):
    """Requirement 1 & 5: Supervisor always keeps final control and writes the answer."""

    def test_supervisor_is_always_the_last_agent(self):
        supervisor_model = ScriptedModel(
            [
                [function_call(RESEARCH_TOOL_NAME, {"input": "compare LangGraph and OpenAI Agents SDK"}, call_id="call_1")],
                [assistant_message("Final answer written by Supervisor.")],
            ]
        )
        research_model = ScriptedModel([[assistant_message("LangGraph vs Agents SDK research notes.")]])
        coding_model = ScriptedModel([])
        review_model = ScriptedModel([])

        agents = build_agents(
            supervisor_model, research_model=research_model, coding_model=coding_model, review_model=review_model
        )
        result = run_supervisor(QUESTION, agents.supervisor)

        self.assertEqual(result.last_agent.name, SUPERVISOR_AGENT_NAME)
        self.assertEqual(result.final_output, "Final answer written by Supervisor.")

        supervisor_model.assert_complete()
        research_model.assert_complete()
        coding_model.assert_complete()
        review_model.assert_complete()

    def test_no_handoff_items_are_ever_produced(self):
        # Requirement 7: verify the run mechanically never uses Handoff items.
        supervisor_model = ScriptedModel(
            [
                [function_call(RESEARCH_TOOL_NAME, {"input": "x"}, call_id="call_1")],
                [assistant_message("done")],
            ]
        )
        research_model = ScriptedModel([[assistant_message("research notes")]])
        agents = build_agents(
            supervisor_model,
            research_model=research_model,
            coding_model=ScriptedModel([]),
            review_model=ScriptedModel([]),
        )

        result = run_supervisor(QUESTION, agents.supervisor)

        self.assertFalse(any(isinstance(item, HandoffCallItem) for item in result.new_items))


class SpecialistsNeverAnswerUserDirectlyTest(unittest.TestCase):
    """Requirement 2 & 3: specialist output only ever reaches Supervisor as a tool result."""

    def test_research_expert_output_is_fed_back_as_tool_result_not_final_output(self):
        supervisor_model = ScriptedModel(
            [
                [function_call(RESEARCH_TOOL_NAME, {"input": "x"}, call_id="call_1")],
                [assistant_message("Supervisor's integrated final answer.")],
            ]
        )
        research_model = ScriptedModel([[assistant_message("RAW-RESEARCH-NOTES")]])
        agents = build_agents(
            supervisor_model,
            research_model=research_model,
            coding_model=ScriptedModel([]),
            review_model=ScriptedModel([]),
        )

        result = run_supervisor(QUESTION, agents.supervisor)

        # The specialist's raw text shows up as a tool output item...
        tool_results = extract_expert_tool_results(result)
        self.assertIn("RAW-RESEARCH-NOTES", "\n".join(tool_results))
        # ...but never as the user-facing final_output, which is Supervisor's own text.
        self.assertEqual(result.final_output, "Supervisor's integrated final answer.")
        self.assertNotEqual(result.final_output, "RAW-RESEARCH-NOTES")


class SupervisorCanCallMultipleExpertsTest(unittest.TestCase):
    """Requirement 4: Supervisor may call one or more specialists.

    This is the required test scenario: "比较 LangGraph 和 OpenAI Agents SDK，
    并给出推荐。" must result in at least ResearchAgent and ReviewAgent both
    being called.
    """

    def test_required_scenario_calls_research_then_review_experts(self):
        supervisor_model = ScriptedModel(
            [
                [
                    function_call(
                        RESEARCH_TOOL_NAME,
                        {"input": "compare LangGraph and OpenAI Agents SDK"},
                        call_id="call_research",
                    )
                ],
                [function_call(REVIEW_TOOL_NAME, {"input": "please review the research notes"}, call_id="call_review")],
                [assistant_message("综合调研与审查结果，推荐 X。")],
            ]
        )
        research_model = ScriptedModel([[assistant_message("LangGraph 和 OpenAI Agents SDK 的比较结论...")]])
        review_model = ScriptedModel([[assistant_message("结论：通过。研究内容基本可靠。")]])
        coding_model = ScriptedModel([])  # must never be called for this question

        agents = build_agents(
            supervisor_model, research_model=research_model, coding_model=coding_model, review_model=review_model
        )

        result = run_supervisor(QUESTION, agents.supervisor)

        called_tools = extract_expert_tool_calls(result)
        self.assertIn(RESEARCH_TOOL_NAME, called_tools)
        self.assertIn(REVIEW_TOOL_NAME, called_tools)
        self.assertNotIn(CODING_TOOL_NAME, called_tools)

        self.assertEqual(result.last_agent.name, SUPERVISOR_AGENT_NAME)
        self.assertEqual(result.final_output, "综合调研与审查结果，推荐 X。")

        supervisor_model.assert_complete()
        research_model.assert_complete()
        review_model.assert_complete()
        coding_model.assert_complete()

    def test_supervisor_can_call_the_same_expert_more_than_once(self):
        supervisor_model = ScriptedModel(
            [
                [function_call(RESEARCH_TOOL_NAME, {"input": "topic A"}, call_id="call_1")],
                [function_call(RESEARCH_TOOL_NAME, {"input": "topic B"}, call_id="call_2")],
                [assistant_message("combined answer")],
            ]
        )
        research_model = ScriptedModel(
            [
                [assistant_message("notes on topic A")],
                [assistant_message("notes on topic B")],
            ]
        )
        agents = build_agents(
            supervisor_model,
            research_model=research_model,
            coding_model=ScriptedModel([]),
            review_model=ScriptedModel([]),
        )

        result = run_supervisor(QUESTION, agents.supervisor)

        self.assertEqual(extract_expert_tool_calls(result), [RESEARCH_TOOL_NAME, RESEARCH_TOOL_NAME])
        self.assertEqual(result.final_output, "combined answer")

    def test_supervisor_can_call_zero_experts(self):
        supervisor_model = ScriptedModel([[assistant_message("Answered directly without any expert.")]])
        agents = build_agents(
            supervisor_model,
            research_model=ScriptedModel([]),
            coding_model=ScriptedModel([]),
            review_model=ScriptedModel([]),
        )

        result = run_supervisor("What is 2 + 2?", agents.supervisor)

        self.assertEqual(extract_expert_tool_calls(result), [])
        self.assertEqual(result.final_output, "Answered directly without any expert.")


class ExpertToolCallsAreOrdinaryToolCallsTest(unittest.TestCase):
    """Sanity check that Agents-as-Tools produces ordinary ToolCallItem/ToolCallOutputItem."""

    def test_run_items_contain_ordinary_tool_call_and_output_items(self):
        supervisor_model = ScriptedModel(
            [
                [function_call(CODING_TOOL_NAME, {"input": "write a function"}, call_id="call_1")],
                [assistant_message("final")],
            ]
        )
        coding_model = ScriptedModel([[assistant_message("def f(): pass")]])
        agents = build_agents(
            supervisor_model,
            research_model=ScriptedModel([]),
            coding_model=coding_model,
            review_model=ScriptedModel([]),
        )

        result = run_supervisor(QUESTION, agents.supervisor)

        item_types = [type(item) for item in result.new_items]
        self.assertIn(ToolCallItem, item_types)
        self.assertIn(ToolCallOutputItem, item_types)


class RunSupervisorValidationTest(unittest.TestCase):
    def test_run_supervisor_rejects_empty_question(self):
        agents = build_agents(
            ScriptedModel([]),
            research_model=ScriptedModel([]),
            coding_model=ScriptedModel([]),
            review_model=ScriptedModel([]),
        )
        with self.assertRaises(ValueError):
            run_supervisor("   ", agents.supervisor)


if __name__ == "__main__":
    unittest.main()
