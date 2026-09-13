"""Tests for ResearchAgent (src/specialists/research_agent.py).

Fully offline: a fake ``TextLLMCall`` is injected, so no network access and no
API key are required.
"""

import unittest

from src.specialists import research_agent

initial_state = research_agent.initial_state
make_research_node = research_agent.make_research_node
build_graph = research_agent.build_graph
run_research_agent = research_agent.run_research_agent
INSTRUCTIONS = research_agent.INSTRUCTIONS

TOPIC = "LangGraph 的 StateGraph 与普通函数式流水线相比有什么优势？"


def _fake_text_llm_call(calls: list[tuple[str, str]], reply: str = "FAKE-RESEARCH-REPORT"):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        return reply

    return call


class InitialStateTest(unittest.TestCase):
    def test_initial_state_has_empty_report(self):
        state = initial_state(TOPIC)
        self.assertEqual(state["topic"], TOPIC)
        self.assertEqual(state["research_report"], "")

    def test_initial_state_rejects_empty_topic(self):
        with self.assertRaises(ValueError):
            initial_state("   ")


class ResearchNodeTest(unittest.TestCase):
    def test_research_node_uses_agent_instructions_and_topic(self):
        calls: list[tuple[str, str]] = []
        node = make_research_node(_fake_text_llm_call(calls))

        update = node(initial_state(TOPIC))

        self.assertEqual(set(update.keys()), {"research_report"})
        self.assertEqual(update["research_report"], "FAKE-RESEARCH-REPORT")
        self.assertEqual(len(calls), 1)
        system_prompt, user_prompt = calls[0]
        self.assertEqual(system_prompt, INSTRUCTIONS)
        self.assertEqual(user_prompt, TOPIC)

    def test_instructions_do_not_reference_other_agents(self):
        # ResearchAgent must not depend on, or be aware of, other agents.
        for forbidden in ("CodingAgent", "ReviewAgent", "handoff", "supervisor"):
            self.assertNotIn(forbidden, INSTRUCTIONS)


class BuildGraphTest(unittest.TestCase):
    def test_graph_runs_topic_to_research_report(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))

        result = graph.invoke(initial_state(TOPIC))

        self.assertEqual(result["topic"], TOPIC)
        self.assertEqual(result["research_report"], "FAKE-RESEARCH-REPORT")

    def test_graph_has_exactly_one_agent_node(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))
        mermaid = graph.get_graph().draw_mermaid()
        self.assertIn("research", mermaid)


class RunResearchAgentTest(unittest.TestCase):
    def test_run_research_agent_returns_report_only(self):
        calls: list[tuple[str, str]] = []
        report = run_research_agent(TOPIC, _fake_text_llm_call(calls))

        self.assertEqual(report, "FAKE-RESEARCH-REPORT")
        self.assertEqual(len(calls), 1)

    def test_run_research_agent_rejects_empty_topic(self):
        with self.assertRaises(ValueError):
            run_research_agent("", _fake_text_llm_call([]))


if __name__ == "__main__":
    unittest.main()
