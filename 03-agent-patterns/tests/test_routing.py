"""Tests for the Routing pattern: Router classifies, exactly one Specialist answers.

These use ``agents.testing.ScriptedModel`` to replay deterministic model output --
no network calls and no API key are required. The Runner (not this test, and not
``src/routing.py``) drives each agent's LLM call, structured-output parsing, and
any Specialist's ``search_web`` tool-call loop.
"""

import unittest
from unittest.mock import patch

from agents.testing import ScriptedModel, assistant_message, function_call
from agents.testing.model import UnconsumedModelSteps

from src.routing import (
    RoutingStepError,
    build_business_researcher,
    build_general_researcher,
    build_router,
    build_technical_researcher,
    run_routing,
)

TECHNICAL_QUESTION = "LangGraph 和 OpenAI Agents SDK 有什么区别？"
BUSINESS_QUESTION = "AI Agent SaaS 市场有哪些机会？"
GENERAL_QUESTION = "什么是 RAG？"

TECHNICAL_ANSWER = (
    "LangGraph is a graph-based orchestration library built around explicit state machines, "
    "while the OpenAI Agents SDK centers on a Runner-driven agent loop with tools, handoffs, "
    "and structured outputs. Key architectural differences: LangGraph exposes nodes/edges you "
    "wire yourself; the Agents SDK owns the loop and exposes Agent/Runner primitives."
)

BUSINESS_ANSWER = (
    "The AI Agent SaaS market shows opportunities in: vertical-specific agents (support, "
    "sales, coding), usage-based pricing models, and mid-market tooling for agent observability "
    "and evaluation, where competition is currently thinner than in general-purpose chat SaaS."
)

GENERAL_ANSWER = (
    "RAG (Retrieval-Augmented Generation) is a technique where a system retrieves relevant "
    "documents or passages and provides them to a language model as context before it "
    "generates an answer, improving factual grounding compared to relying on the model alone."
)


def _router_json(category: str, reason: str) -> str:
    return f'{{"category": "{category}", "reason": "{reason}"}}'


class RouterAndSpecialistDefinitionTest(unittest.TestCase):
    """Requirement 1 & 2: Router only classifies; each Specialist has distinct instructions."""

    def test_router_has_no_tools_and_structured_output(self):
        agent = build_router("placeholder-model")
        self.assertEqual(agent.name, "Router")
        self.assertEqual(agent.tools, [])
        self.assertIsNotNone(agent.output_type)

    def test_specialists_have_search_web_tool_and_no_output_type(self):
        for builder in (build_technical_researcher, build_business_researcher, build_general_researcher):
            agent = builder("placeholder-model")
            self.assertEqual([tool.name for tool in agent.tools], ["search_web"])
            self.assertIsNone(agent.output_type)

    def test_specialists_have_distinct_names_and_instructions(self):
        technical = build_technical_researcher("placeholder-model")
        business = build_business_researcher("placeholder-model")
        general = build_general_researcher("placeholder-model")

        names = {technical.name, business.name, general.name}
        self.assertEqual(len(names), 3, "each specialist must have a distinct name")

        instructions = {technical.instructions, business.instructions, general.instructions}
        self.assertEqual(len(instructions), 3, "each specialist must have distinct instructions")

    def test_no_agent_declares_handoffs(self):
        """Requirement: no multi-agent handoffs anywhere in the routing system."""
        builders = (
            build_router,
            build_technical_researcher,
            build_business_researcher,
            build_general_researcher,
        )
        for builder in builders:
            agent = builder("placeholder-model")
            self.assertEqual(agent.handoffs, [], f"{agent.name} must not declare handoffs")


class RoutingDispatchTest(unittest.TestCase):
    """Requirements 6 & 7: Router decides the path; only one Specialist ever runs per question."""

    def test_technical_question_routes_to_technical_researcher_only(self):
        model = ScriptedModel(
            [
                [assistant_message(_router_json("technical", "Compares two SDKs/frameworks."))],
                [assistant_message(TECHNICAL_ANSWER)],
            ]
        )

        result = run_routing(TECHNICAL_QUESTION, model)

        self.assertEqual(result.decision.category, "technical")
        self.assertEqual(result.final_answer, TECHNICAL_ANSWER)
        model.assert_complete()

    def test_business_question_routes_to_business_researcher_only(self):
        model = ScriptedModel(
            [
                [assistant_message(_router_json("business", "Asks about market opportunities."))],
                [assistant_message(BUSINESS_ANSWER)],
            ]
        )

        result = run_routing(BUSINESS_QUESTION, model)

        self.assertEqual(result.decision.category, "business")
        self.assertEqual(result.final_answer, BUSINESS_ANSWER)
        model.assert_complete()

    def test_general_question_routes_to_general_researcher_only(self):
        model = ScriptedModel(
            [
                [assistant_message(_router_json("general", "Asks for a definition of a concept."))],
                [assistant_message(GENERAL_ANSWER)],
            ]
        )

        result = run_routing(GENERAL_QUESTION, model)

        self.assertEqual(result.decision.category, "general")
        self.assertEqual(result.final_answer, GENERAL_ANSWER)
        model.assert_complete()

    def test_technical_specialist_can_call_search_web(self):
        model = ScriptedModel(
            [
                [assistant_message(_router_json("technical", "Compares two frameworks."))],
                [function_call("search_web", {"query": "LangGraph vs OpenAI Agents SDK"}, call_id="call_1")],
                [assistant_message(TECHNICAL_ANSWER)],
            ]
        )

        with patch("src.routing._web_search") as fake_web_search:
            fake_web_search.return_value = {
                "query": "LangGraph vs OpenAI Agents SDK",
                "results": [{"title": "docs", "url": "https://example.com", "snippet": "..."}],
            }
            result = run_routing(TECHNICAL_QUESTION, model)

        fake_web_search.assert_called_once_with("LangGraph vs OpenAI Agents SDK")
        self.assertEqual(result.final_answer, TECHNICAL_ANSWER)
        model.assert_complete()


class RoutingFailureReportingTest(unittest.TestCase):
    def test_router_failure_is_reported_as_router_step(self):
        model = ScriptedModel([[assistant_message("not valid json")]])

        with self.assertRaises(RoutingStepError) as ctx:
            run_routing(TECHNICAL_QUESTION, model)

        self.assertEqual(ctx.exception.step_name, "router")

    def test_specialist_failure_is_reported_with_its_category_name(self):
        model = ScriptedModel(
            [
                [assistant_message(_router_json("business", "Market question."))],
                [function_call("search_web", {"query": "x"}, call_id="call_1")],
                [function_call("search_web", {"query": "y"}, call_id="call_2")],
            ]
        )
        # Specialist keeps calling tools and never returns plain text within max_turns.
        with patch("src.routing._web_search", return_value={"query": "x", "results": []}):
            with self.assertRaises(RoutingStepError) as ctx:
                run_routing(BUSINESS_QUESTION, model, max_turns=1)

        self.assertEqual(ctx.exception.step_name, "business")

    def test_rejects_empty_question(self):
        model = ScriptedModel([[assistant_message(_router_json("general", "n/a"))]])
        with self.assertRaises(ValueError):
            run_routing("   ", model)
        with self.assertRaises(UnconsumedModelSteps):
            model.assert_complete()


if __name__ == "__main__":
    unittest.main()
