"""Tests for the OpenAI Agents SDK-based Agent (User -> Agent -> Runner -> Final Answer).

These tests use the SDK's own public testing helper, ``agents.testing.ScriptedModel``,
to replay deterministic model output -- no network calls and no API key are
required to run these tests. Tool dispatch and the agent loop itself are never
implemented by hand here: the Runner and the ``@function_tool``-wrapped
``web_search`` (see ``src/agent.py``) do all of that.
"""

import unittest
from unittest.mock import patch

from agents import Runner
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.testing import ScriptedModel, assistant_message, function_call

from src.agent import AGENT_NAME, build_agent
from src.prompts import SYSTEM_PROMPT


class SdkAgentDefinitionTest(unittest.TestCase):
    def test_agent_is_configured_with_migrated_instructions_and_web_search_tool(self):
        model = ScriptedModel([[assistant_message("placeholder")]])
        agent = build_agent(model)

        self.assertEqual(agent.name, AGENT_NAME)
        self.assertEqual(agent.instructions, SYSTEM_PROMPT)
        self.assertEqual([tool.name for tool in agent.tools], ["web_search"])

    def test_runner_executes_agent_and_returns_final_text_answer(self):
        model = ScriptedModel([[assistant_message("The capital of France is Paris.")]])
        agent = build_agent(model)

        result = Runner.run_sync(agent, "What is the capital of France?", max_turns=5)

        self.assertEqual(result.final_output, "The capital of France is Paris.")
        # Confirms the scripted model call was actually consumed by the Runner,
        # i.e. the User -> Agent -> Runner -> Final Answer path really executed.
        model.assert_complete()


class WebSearchToolCallDecisionTest(unittest.TestCase):
    """Case 1 & 2: whether ``web_search`` is called is decided by the LLM, not by us."""

    def test_case1_agent_does_not_call_web_search_when_not_needed(self):
        # The scripted model answers directly, with no function_call output item at
        # all -- simulating an LLM that decided the question needs no search.
        model = ScriptedModel([[assistant_message("2 + 2 equals 4.")]])
        agent = build_agent(model)

        with patch("src.agent._web_search") as fake_web_search:
            result = Runner.run_sync(agent, "What is 2 + 2?", max_turns=5)

        self.assertEqual(result.final_output, "2 + 2 equals 4.")
        fake_web_search.assert_not_called()
        self.assertFalse(
            any(isinstance(item, ToolCallItem) for item in result.new_items),
            "no tool call should have been made",
        )
        model.assert_complete()

    def test_case2_agent_can_call_web_search_when_needed(self):
        # The scripted model first emits a function_call for web_search, then (once
        # the Runner has fed the tool result back in) emits the final answer. We
        # never dispatch the tool call or drive the second turn ourselves.
        model = ScriptedModel(
            [
                [
                    function_call(
                        "web_search",
                        {"query": "current population of France"},
                        call_id="call_1",
                    )
                ],
                [assistant_message("France's population is about 68 million, per official sources.")],
            ]
        )
        agent = build_agent(model)

        with patch(
            "src.agent._web_search",
            return_value={
                "query": "current population of France",
                "results": [{"title": "INSEE", "url": "https://insee.fr", "snippet": "68 million"}],
            },
        ) as fake_web_search:
            result = Runner.run_sync(
                agent, "Research the current population of France.", max_turns=5
            )

        fake_web_search.assert_called_once_with("current population of France")
        self.assertEqual(
            result.final_output,
            "France's population is about 68 million, per official sources.",
        )
        model.assert_complete()


class WebSearchToolResultReEntersLoopTest(unittest.TestCase):
    def test_case3_tool_result_automatically_flows_back_into_the_agent_loop(self):
        """Case 3: the tool result re-enters the Agent loop without us wiring it up."""
        model = ScriptedModel(
            [
                [function_call("web_search", {"query": "capital of Japan"}, call_id="call_1")],
                [assistant_message("The capital of Japan is Tokyo.")],
            ]
        )
        agent = build_agent(model)

        with patch(
            "src.agent._web_search",
            return_value={
                "query": "capital of Japan",
                "results": [{"title": "Tokyo", "url": "https://example.com", "snippet": "Tokyo"}],
            },
        ):
            result = Runner.run_sync(agent, "What is the capital of Japan?", max_turns=5)

        # The Runner recorded both the tool call and its output as run items --
        # proof that the Runner (not our code) dispatched the call and fed the
        # result back in for the model's next turn, which produced final_output.
        item_types = [type(item) for item in result.new_items]
        self.assertIn(ToolCallItem, item_types)
        self.assertIn(ToolCallOutputItem, item_types)
        self.assertEqual(result.final_output, "The capital of Japan is Tokyo.")
        model.assert_complete()


if __name__ == "__main__":
    unittest.main()
