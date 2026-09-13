"""Tests for src/patterns/evaluator_optimizer.py -- fully offline."""

import json
import unittest

from src.patterns import evaluator_optimizer as eo


def _fake_generator(drafts: list[str]):
    calls: list[str] = []

    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append(user_prompt)
        return drafts[len(calls) - 1]

    call.calls = calls
    return call


def _fake_evaluator(scores: list[int]):
    calls: list[str] = []

    def call(system_prompt: str, user_prompt: str) -> dict:
        calls.append(user_prompt)
        score = scores[len(calls) - 1]
        return json.dumps({"score": score, "feedback": [] if score >= eo.PASS_SCORE else ["improve X"]})

    call.calls = calls
    return call


class GeneratorNodeTest(unittest.TestCase):
    def test_generator_writes_draft_and_increments_iteration(self):
        generator = eo.make_generator(lambda s, u: "draft text")
        update = generator(eo.initial_state("Q"))
        self.assertEqual(set(update.keys()), {"draft", "iteration"})
        self.assertEqual(update["draft"], "draft text")
        self.assertEqual(update["iteration"], 1)

    def test_generator_prompt_includes_feedback_on_revision(self):
        captured = {}

        def llm_call(system_prompt: str, user_prompt: str) -> str:
            captured["prompt"] = user_prompt
            return "revised draft"

        generator = eo.make_generator(llm_call)
        state = eo.initial_state("Q")
        state["draft"] = "old draft"
        state["feedback"] = ["fix the intro"]
        state["iteration"] = 1
        generator(state)
        self.assertIn("old draft", captured["prompt"])
        self.assertIn("fix the intro", captured["prompt"])


class EvaluatorNodeTest(unittest.TestCase):
    def test_evaluator_never_writes_draft(self):
        evaluator = eo.make_evaluator(lambda s, u: json.dumps({"score": 5, "feedback": ["x"]}))
        state = eo.initial_state("Q")
        state["draft"] = "some draft"
        update = evaluator(state)
        self.assertNotIn("draft", update)
        self.assertEqual(update["score"], 5)
        self.assertEqual(update["feedback"], ["x"])

    def test_evaluator_tracks_best_draft_across_iterations(self):
        evaluator = eo.make_evaluator(lambda s, u: json.dumps({"score": 7, "feedback": []}))
        state = eo.initial_state("Q")
        state["draft"] = "draft v2"
        state["best_score"] = 3
        state["best_draft"] = "draft v1"
        update = evaluator(state)
        self.assertEqual(update["best_score"], 7)
        self.assertEqual(update["best_draft"], "draft v2")

    def test_evaluator_keeps_previous_best_if_new_score_is_lower(self):
        evaluator = eo.make_evaluator(lambda s, u: json.dumps({"score": 2, "feedback": []}))
        state = eo.initial_state("Q")
        state["draft"] = "worse draft"
        state["best_score"] = 6
        state["best_draft"] = "better draft"
        update = evaluator(state)
        self.assertNotIn("best_draft", update)
        self.assertNotIn("best_score", update)


class ShouldRevideTest(unittest.TestCase):
    def test_routes_to_revise_when_score_below_threshold_and_iterations_remain(self):
        state = eo.initial_state("Q")
        state["score"] = eo.PASS_SCORE - 1
        state["iteration"] = 1
        self.assertEqual(eo.should_revise(state), "revise")

    def test_routes_to_accept_when_score_meets_threshold(self):
        state = eo.initial_state("Q")
        state["score"] = eo.PASS_SCORE
        state["iteration"] = 1
        self.assertEqual(eo.should_revise(state), "accept")

    def test_routes_to_accept_when_max_iterations_reached_even_if_score_is_low(self):
        state = eo.initial_state("Q")
        state["score"] = 0
        state["iteration"] = eo.MAX_ITERATIONS
        self.assertEqual(eo.should_revise(state), "accept")


class FullGraphTest(unittest.TestCase):
    def test_accepts_immediately_when_first_draft_passes(self):
        generator = _fake_generator(["good draft"])
        evaluator = _fake_evaluator([eo.PASS_SCORE])
        graph = eo.build_graph(generator, evaluator)

        result = graph.invoke(eo.initial_state("Q"))

        self.assertEqual(result["final_answer"], "good draft")
        self.assertEqual(result["iteration"], 1)
        self.assertEqual(len(generator.calls), 1)

    def test_loops_until_score_passes(self):
        generator = _fake_generator(["draft v1", "draft v2", "draft v3"])
        evaluator = _fake_evaluator([3, 5, eo.PASS_SCORE])
        graph = eo.build_graph(generator, evaluator)

        result = graph.invoke(eo.initial_state("Q"), config={"recursion_limit": 50})

        self.assertEqual(result["final_answer"], "draft v3")
        self.assertEqual(result["iteration"], 3)
        self.assertEqual(len(generator.calls), 3)

    def test_max_iterations_cap_stops_a_never_passing_draft(self):
        generator = _fake_generator(["v1", "v2", "v3", "v4", "v5"])
        evaluator = _fake_evaluator([1, 2, 3, 4, 5])
        graph = eo.build_graph(generator, evaluator)

        result = graph.invoke(eo.initial_state("Q"), config={"recursion_limit": 50})

        # Stops at MAX_ITERATIONS regardless of never reaching PASS_SCORE.
        self.assertEqual(result["iteration"], eo.MAX_ITERATIONS)
        # Final answer is the BEST-scoring draft seen, which is the last one (score 3).
        self.assertEqual(result["final_answer"], "v3")

    def test_mermaid_diagram_contains_all_three_nodes(self):
        graph = eo.build_graph(_fake_generator(["x"]), _fake_evaluator([eo.PASS_SCORE]))
        mermaid = eo.get_mermaid(graph)
        for name in ("generator", "evaluator", "finalize"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
