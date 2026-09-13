"""Tests for CodingAgent (src/specialists/coding_agent.py).

Fully offline: a fake ``TextLLMCall`` is injected, so no network access and no
API key are required.
"""

import unittest

from src.specialists import coding_agent

initial_state = coding_agent.initial_state
make_design_node = coding_agent.make_design_node
build_graph = coding_agent.build_graph
run_coding_agent = coding_agent.run_coding_agent
INSTRUCTIONS = coding_agent.INSTRUCTIONS

REQUIREMENT = "设计一个函数，将一批文本文档去重并按相似度聚类。"


def _fake_text_llm_call(calls: list[tuple[str, str]], reply: str = "FAKE-DESIGN-PROPOSAL"):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        return reply

    return call


class InitialStateTest(unittest.TestCase):
    def test_initial_state_has_empty_proposal(self):
        state = initial_state(REQUIREMENT)
        self.assertEqual(state["requirement"], REQUIREMENT)
        self.assertEqual(state["design_proposal"], "")

    def test_initial_state_rejects_empty_requirement(self):
        with self.assertRaises(ValueError):
            initial_state("")


class DesignNodeTest(unittest.TestCase):
    def test_design_node_uses_agent_instructions_and_requirement(self):
        calls: list[tuple[str, str]] = []
        node = make_design_node(_fake_text_llm_call(calls))

        update = node(initial_state(REQUIREMENT))

        self.assertEqual(set(update.keys()), {"design_proposal"})
        self.assertEqual(update["design_proposal"], "FAKE-DESIGN-PROPOSAL")
        self.assertEqual(len(calls), 1)
        system_prompt, user_prompt = calls[0]
        self.assertEqual(system_prompt, INSTRUCTIONS)
        self.assertEqual(user_prompt, REQUIREMENT)

    def test_instructions_do_not_reference_other_agents(self):
        for forbidden in ("ResearchAgent", "ReviewAgent", "handoff", "supervisor"):
            self.assertNotIn(forbidden, INSTRUCTIONS)


class BuildGraphTest(unittest.TestCase):
    def test_graph_runs_requirement_to_design_proposal(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))

        result = graph.invoke(initial_state(REQUIREMENT))

        self.assertEqual(result["requirement"], REQUIREMENT)
        self.assertEqual(result["design_proposal"], "FAKE-DESIGN-PROPOSAL")

    def test_graph_has_exactly_one_agent_node(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))
        mermaid = graph.get_graph().draw_mermaid()
        self.assertIn("design", mermaid)


class RunCodingAgentTest(unittest.TestCase):
    def test_run_coding_agent_returns_proposal_only(self):
        calls: list[tuple[str, str]] = []
        proposal = run_coding_agent(REQUIREMENT, _fake_text_llm_call(calls))

        self.assertEqual(proposal, "FAKE-DESIGN-PROPOSAL")
        self.assertEqual(len(calls), 1)

    def test_run_coding_agent_rejects_empty_requirement(self):
        with self.assertRaises(ValueError):
            run_coding_agent("   ", _fake_text_llm_call([]))


if __name__ == "__main__":
    unittest.main()
