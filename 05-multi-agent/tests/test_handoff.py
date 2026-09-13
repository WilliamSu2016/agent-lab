"""Tests for Experiment 2 (Handoff): src/02_handoff.py.

Uses the SDK's own public testing helper, ``agents.testing.ScriptedModel``, to
replay deterministic model output per agent -- no network calls and no API key
are required. The Runner (not this test file) performs the actual control
transfer: dispatching the ``transfer_to_research``/``transfer_to_coding`` tool
call, switching the active agent, and feeding that agent's model the
conversation so far.

The module under test is named ``02_handoff.py`` (starts with a digit), so it
is loaded dynamically with ``importlib`` -- see ``tests/test_prompt_chaining_graph.py``-
style modules in ``04-langgraph`` for the same pattern.
"""

import importlib
import unittest

from agents import Runner
from agents.items import HandoffCallItem, HandoffOutputItem, ToolCallItem
from agents.testing import ScriptedModel, assistant_message, function_call

handoff_mod = importlib.import_module("src.02_handoff")

TRIAGE_AGENT_NAME = handoff_mod.TRIAGE_AGENT_NAME
RESEARCH_AGENT_NAME = handoff_mod.RESEARCH_AGENT_NAME
CODING_AGENT_NAME = handoff_mod.CODING_AGENT_NAME
RESEARCH_HANDOFF_DESCRIPTION = handoff_mod.RESEARCH_HANDOFF_DESCRIPTION
CODING_HANDOFF_DESCRIPTION = handoff_mod.CODING_HANDOFF_DESCRIPTION
build_research_agent = handoff_mod.build_research_agent
build_coding_agent = handoff_mod.build_coding_agent
build_triage_agent = handoff_mod.build_triage_agent
build_agents = handoff_mod.build_agents
run_handoff = handoff_mod.run_handoff
extract_handoff_trace = handoff_mod.extract_handoff_trace
HandoffEvent = handoff_mod.HandoffEvent

RESEARCH_QUESTION = "比较 LangGraph 和 OpenAI Agents SDK。"
CODING_QUESTION = "帮我写一个 LangGraph Agent。"


class AgentWiringTest(unittest.TestCase):
    """Structural checks: names, handoff_description, and who has handoffs."""

    def test_research_agent_has_handoff_description_and_no_handoffs_of_its_own(self):
        agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))

        self.assertEqual(agent.name, RESEARCH_AGENT_NAME)
        self.assertEqual(agent.handoff_description, RESEARCH_HANDOFF_DESCRIPTION)
        self.assertEqual(agent.handoffs, [])

    def test_coding_agent_has_handoff_description_and_no_handoffs_of_its_own(self):
        agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))

        self.assertEqual(agent.name, CODING_AGENT_NAME)
        self.assertEqual(agent.handoff_description, CODING_HANDOFF_DESCRIPTION)
        self.assertEqual(agent.handoffs, [])

    def test_triage_agent_has_exactly_the_two_expected_handoff_tools(self):
        research_agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))
        coding_agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))
        triage_agent = build_triage_agent(
            ScriptedModel([[assistant_message("x")]]), research_agent=research_agent, coding_agent=coding_agent
        )

        self.assertEqual(triage_agent.name, TRIAGE_AGENT_NAME)
        tool_names = {h.tool_name for h in triage_agent.handoffs}
        self.assertEqual(tool_names, {"transfer_to_research", "transfer_to_coding"})

    def test_triage_handoffs_point_to_the_correct_target_agents(self):
        research_agent = build_research_agent(ScriptedModel([[assistant_message("x")]]))
        coding_agent = build_coding_agent(ScriptedModel([[assistant_message("x")]]))
        triage_agent = build_triage_agent(
            ScriptedModel([[assistant_message("x")]]), research_agent=research_agent, coding_agent=coding_agent
        )

        by_tool_name = {h.tool_name: h for h in triage_agent.handoffs}
        self.assertEqual(by_tool_name["transfer_to_research"].agent_name, research_agent.name)
        self.assertEqual(by_tool_name["transfer_to_coding"].agent_name, coding_agent.name)


class ResearchHandoffTest(unittest.TestCase):
    """"比较 LangGraph 和 OpenAI Agents SDK。" must be handed off to ResearchAgent."""

    def test_triage_hands_off_to_research_agent_which_answers(self):
        triage_model = ScriptedModel([[function_call("transfer_to_research", {}, call_id="call_1")]])
        research_model = ScriptedModel(
            [[assistant_message("LangGraph 是图驱动的编排框架，OpenAI Agents SDK 是内置 Agent Runtime 的 SDK。")]]
        )
        coding_model = ScriptedModel([])  # must never be called

        agents = build_agents(triage_model, research_model=research_model, coding_model=coding_model)

        result = run_handoff(RESEARCH_QUESTION, agents.triage)

        # Requirement 5 & 6: TriageAgent itself never produces the final
        # answer; the specialist it handed off to does.
        self.assertEqual(result.last_agent.name, RESEARCH_AGENT_NAME)
        self.assertIn("LangGraph", result.final_output)

        triage_model.assert_complete()
        research_model.assert_complete()
        coding_model.assert_complete()  # empty script => never invoked, by construction

    def test_trace_shows_triage_transfer_to_research_researchagent(self):
        triage_model = ScriptedModel([[function_call("transfer_to_research", {}, call_id="call_1")]])
        research_model = ScriptedModel([[assistant_message("research answer")]])
        coding_model = ScriptedModel([])

        agents = build_agents(triage_model, research_model=research_model, coding_model=coding_model)
        result = run_handoff(RESEARCH_QUESTION, agents.triage)

        events = extract_handoff_trace(result)

        self.assertEqual(
            events,
            [HandoffEvent(source_agent=TRIAGE_AGENT_NAME, tool_name="transfer_to_research", target_agent=RESEARCH_AGENT_NAME)],
        )

    def test_no_coding_handoff_tool_call_is_made(self):
        triage_model = ScriptedModel([[function_call("transfer_to_research", {}, call_id="call_1")]])
        research_model = ScriptedModel([[assistant_message("research answer")]])
        coding_model = ScriptedModel([])

        agents = build_agents(triage_model, research_model=research_model, coding_model=coding_model)
        result = run_handoff(RESEARCH_QUESTION, agents.triage)

        tool_call_names = [
            item.raw_item.name for item in result.new_items if isinstance(item, (ToolCallItem, HandoffCallItem))
        ]
        self.assertEqual(tool_call_names, ["transfer_to_research"])


class CodingHandoffTest(unittest.TestCase):
    """"帮我写一个 LangGraph Agent。" must be handed off to CodingAgent."""

    def test_triage_hands_off_to_coding_agent_which_answers(self):
        triage_model = ScriptedModel([[function_call("transfer_to_coding", {}, call_id="call_1")]])
        research_model = ScriptedModel([])  # must never be called
        coding_model = ScriptedModel(
            [[assistant_message("from agents import Agent\n\nagent = Agent(name='X', instructions='...')")]]
        )

        agents = build_agents(triage_model, research_model=research_model, coding_model=coding_model)

        result = run_handoff(CODING_QUESTION, agents.triage)

        self.assertEqual(result.last_agent.name, CODING_AGENT_NAME)
        self.assertIn("Agent(", result.final_output)

        triage_model.assert_complete()
        research_model.assert_complete()
        coding_model.assert_complete()

    def test_trace_shows_triage_transfer_to_coding_codingagent(self):
        triage_model = ScriptedModel([[function_call("transfer_to_coding", {}, call_id="call_1")]])
        research_model = ScriptedModel([])
        coding_model = ScriptedModel([[assistant_message("code answer")]])

        agents = build_agents(triage_model, research_model=research_model, coding_model=coding_model)
        result = run_handoff(CODING_QUESTION, agents.triage)

        events = extract_handoff_trace(result)

        self.assertEqual(
            events,
            [HandoffEvent(source_agent=TRIAGE_AGENT_NAME, tool_name="transfer_to_coding", target_agent=CODING_AGENT_NAME)],
        )


class HandoffOutputItemPresentTest(unittest.TestCase):
    def test_run_items_contain_both_handoff_call_and_output_items(self):
        triage_model = ScriptedModel([[function_call("transfer_to_coding", {}, call_id="call_1")]])
        coding_model = ScriptedModel([[assistant_message("code answer")]])
        agents = build_agents(triage_model, research_model=ScriptedModel([]), coding_model=coding_model)

        result = run_handoff(CODING_QUESTION, agents.triage)

        item_types = [type(item) for item in result.new_items]
        self.assertIn(HandoffCallItem, item_types)
        self.assertIn(HandoffOutputItem, item_types)


class RunHandoffValidationTest(unittest.TestCase):
    def test_run_handoff_rejects_empty_question(self):
        agents = build_agents(ScriptedModel([]), research_model=ScriptedModel([]), coding_model=ScriptedModel([]))
        with self.assertRaises(ValueError):
            run_handoff("   ", agents.triage)


class RunnerIsUsedDirectlyTest(unittest.TestCase):
    """Sanity check: the handoff/Runner mechanics are the SDK's, not hand-rolled."""

    def test_run_handoff_delegates_to_runner_run_sync(self):
        triage_model = ScriptedModel([[function_call("transfer_to_research", {}, call_id="call_1")]])
        research_model = ScriptedModel([[assistant_message("answer")]])
        agents = build_agents(triage_model, research_model=research_model, coding_model=ScriptedModel([]))

        direct_result = Runner.run_sync(agents.triage, RESEARCH_QUESTION, max_turns=10)

        self.assertEqual(direct_result.last_agent.name, RESEARCH_AGENT_NAME)
        self.assertEqual(direct_result.final_output, "answer")


if __name__ == "__main__":
    unittest.main()
