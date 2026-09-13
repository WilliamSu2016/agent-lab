"""Tests for ReviewAgent (src/specialists/review_agent.py).

Fully offline: a fake ``TextLLMCall`` is injected, so no network access and no
API key are required.
"""

import unittest

from src.specialists import review_agent

initial_state = review_agent.initial_state
make_review_node = review_agent.make_review_node
build_graph = review_agent.build_graph
run_review_agent = review_agent.run_review_agent
INSTRUCTIONS = review_agent.INSTRUCTIONS

RESEARCH_CONTENT = "摘要：LangGraph 提供显式状态图。\n关键要点：...\n风险与不确定项：无"
CODE_DESIGN_CONTENT = "方案概述：用装饰器缓存函数结果。\n模块/文件结构：cache.py\n关键实现：...\n权衡与风险：无"


def _fake_text_llm_call(calls: list[tuple[str, str]], reply: str = "FAKE-REVIEW-RESULT"):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        return reply

    return call


class InitialStateTest(unittest.TestCase):
    def test_initial_state_accepts_research_report(self):
        state = initial_state("research_report", RESEARCH_CONTENT)
        self.assertEqual(state["artifact_type"], "research_report")
        self.assertEqual(state["content"], RESEARCH_CONTENT)
        self.assertEqual(state["review_result"], "")

    def test_initial_state_accepts_code_design(self):
        state = initial_state("code_design", CODE_DESIGN_CONTENT)
        self.assertEqual(state["artifact_type"], "code_design")

    def test_initial_state_rejects_unknown_artifact_type(self):
        with self.assertRaises(ValueError):
            initial_state("something_else", CODE_DESIGN_CONTENT)  # type: ignore[arg-type]

    def test_initial_state_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            initial_state("code_design", "  ")


class ReviewNodeTest(unittest.TestCase):
    def test_review_node_includes_artifact_type_and_content_in_prompt(self):
        calls: list[tuple[str, str]] = []
        node = make_review_node(_fake_text_llm_call(calls))

        update = node(initial_state("code_design", CODE_DESIGN_CONTENT))

        self.assertEqual(set(update.keys()), {"review_result"})
        self.assertEqual(update["review_result"], "FAKE-REVIEW-RESULT")
        self.assertEqual(len(calls), 1)
        system_prompt, user_prompt = calls[0]
        self.assertEqual(system_prompt, INSTRUCTIONS)
        self.assertIn("code_design", user_prompt)
        self.assertIn(CODE_DESIGN_CONTENT, user_prompt)

    def test_review_node_distinguishes_research_report_artifact(self):
        calls: list[tuple[str, str]] = []
        node = make_review_node(_fake_text_llm_call(calls))

        node(initial_state("research_report", RESEARCH_CONTENT))

        _, user_prompt = calls[0]
        self.assertIn("research_report", user_prompt)
        self.assertIn(RESEARCH_CONTENT, user_prompt)

    def test_instructions_do_not_reference_other_agents(self):
        for forbidden in ("ResearchAgent", "CodingAgent", "handoff", "supervisor"):
            self.assertNotIn(forbidden, INSTRUCTIONS)


class BuildGraphTest(unittest.TestCase):
    def test_graph_runs_content_to_review_result(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))

        result = graph.invoke(initial_state("code_design", CODE_DESIGN_CONTENT))

        self.assertEqual(result["content"], CODE_DESIGN_CONTENT)
        self.assertEqual(result["review_result"], "FAKE-REVIEW-RESULT")

    def test_graph_has_exactly_one_agent_node(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_fake_text_llm_call(calls))
        mermaid = graph.get_graph().draw_mermaid()
        self.assertIn("review", mermaid)


class RunReviewAgentTest(unittest.TestCase):
    def test_run_review_agent_returns_result_only(self):
        calls: list[tuple[str, str]] = []
        result = run_review_agent("research_report", RESEARCH_CONTENT, _fake_text_llm_call(calls))

        self.assertEqual(result, "FAKE-REVIEW-RESULT")
        self.assertEqual(len(calls), 1)

    def test_run_review_agent_rejects_bad_artifact_type(self):
        with self.assertRaises(ValueError):
            run_review_agent("nonsense", CODE_DESIGN_CONTENT, _fake_text_llm_call([]))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
