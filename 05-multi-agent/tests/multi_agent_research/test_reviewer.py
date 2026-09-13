"""Tests for the Reviewer (src/multi_agent_research/reviewer.py).

Fully offline: fake ``TextLLMCall``s are injected, no network/API key
required.
"""

import json
import unittest

from src.multi_agent_research.reviewer import _parse_verdict, make_reviewer_node, route_after_review
from src.multi_agent_research.state import initial_state

QUESTION = "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"


def _verdict_llm_call(verdict: dict):
    def call(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(verdict, ensure_ascii=False)

    return call


class ParseVerdictTest(unittest.TestCase):
    def test_parses_full_verdict(self):
        raw = json.dumps(
            {
                "approved": True,
                "completeness": "good",
                "factual_consistency": "ok",
                "evidence_quality": "solid",
                "logical_consistency": "coherent",
                "missing_aspects": [],
                "feedback": "",
            }
        )
        verdict = _parse_verdict(raw)
        self.assertTrue(verdict["approved"])
        self.assertEqual(verdict["missing_aspects"], [])

    def test_parses_json_in_code_fence(self):
        raw = '```json\n{"approved": false, "feedback": "bad", "missing_aspects": ["x"]}\n```'
        verdict = _parse_verdict(raw)
        self.assertFalse(verdict["approved"])
        self.assertEqual(verdict["missing_aspects"], ["x"])
        self.assertEqual(verdict["feedback"], "bad")

    def test_falls_back_to_rejecting_on_unparseable_output(self):
        verdict = _parse_verdict("this is not json at all")
        self.assertFalse(verdict["approved"])
        self.assertTrue(verdict["feedback"])

    def test_coerces_non_list_missing_aspects_to_a_list(self):
        raw = json.dumps({"approved": False, "missing_aspects": "a single string"})
        verdict = _parse_verdict(raw)
        self.assertEqual(verdict["missing_aspects"], ["a single string"])


class ReviewerNodeTest(unittest.TestCase):
    def test_only_returns_review_and_trace(self):
        node = make_reviewer_node(
            _verdict_llm_call({"approved": True, "feedback": "", "missing_aspects": []})
        )
        state = initial_state(QUESTION)
        state["synthesis"] = "a synthesis"
        state["iteration"] = 1
        update = node(state)
        self.assertEqual(set(update.keys()), {"review", "trace"})

    def test_checks_all_five_dimensions_are_present_in_the_prompt_contract(self):
        # The instructions ask the model for five distinct dimensions; make
        # sure a well-formed response round-trips all five through parsing.
        raw = json.dumps(
            {
                "approved": False,
                "completeness": "missing X",
                "factual_consistency": "contradiction found",
                "evidence_quality": "unsupported claim",
                "logical_consistency": "non sequitur",
                "missing_aspects": ["X"],
                "feedback": "add X and fix the contradiction",
            }
        )
        node = make_reviewer_node(lambda s, u: raw)
        state = initial_state(QUESTION)
        state["synthesis"] = "flawed synthesis"
        state["iteration"] = 1
        update = node(state)
        review = update["review"]
        for key in (
            "completeness",
            "factual_consistency",
            "evidence_quality",
            "logical_consistency",
            "missing_aspects",
            "feedback",
        ):
            self.assertTrue(review[key])

    def test_reviewer_reads_synthesis_but_cannot_write_it(self):
        seen_prompts = []

        def call(system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append(user_prompt)
            return json.dumps({"approved": True, "feedback": ""})

        node = make_reviewer_node(call)
        state = initial_state(QUESTION)
        state["synthesis"] = "the synthesis text to review"
        state["iteration"] = 1
        update = node(state)

        self.assertIn("the synthesis text to review", seen_prompts[0])
        self.assertNotIn("synthesis", update)


class RouteAfterReviewTest(unittest.TestCase):
    def test_routes_to_finalizer_when_approved(self):
        state = initial_state(QUESTION)
        state["review"] = {"approved": True, "completeness": "", "factual_consistency": "", "evidence_quality": "", "logical_consistency": "", "missing_aspects": [], "feedback": ""}
        state["iteration"] = 1
        self.assertEqual(route_after_review(state), "finalizer")

    def test_routes_back_to_planner_when_rejected_and_budget_remains(self):
        state = initial_state(QUESTION, max_iterations=3)
        state["review"] = {"approved": False, "completeness": "", "factual_consistency": "", "evidence_quality": "", "logical_consistency": "", "missing_aspects": ["gap"], "feedback": "fix it"}
        state["iteration"] = 1
        self.assertEqual(route_after_review(state), "planner")

    def test_routes_to_finalizer_once_max_iterations_reached_even_if_rejected(self):
        state = initial_state(QUESTION, max_iterations=3)
        state["review"] = {"approved": False, "completeness": "", "factual_consistency": "", "evidence_quality": "", "logical_consistency": "", "missing_aspects": ["gap"], "feedback": "still bad"}
        state["iteration"] = 3
        self.assertEqual(route_after_review(state), "finalizer")


if __name__ == "__main__":
    unittest.main()
