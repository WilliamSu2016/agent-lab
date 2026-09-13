"""Tests for the State-focused LangGraph experiment (src/02_state.py).

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

state_module = importlib.import_module("src.02_state")

ResearchState = state_module.ResearchState
research = state_module.research
analysis = state_module.analysis
finalize = state_module.finalize
build_graph = state_module.build_graph


def _initial_state(question: str) -> ResearchState:
    return {
        "question": question,
        "research_notes": [],
        "analysis": "",
        "final_answer": "",
    }


class NodeOwnershipTest(unittest.TestCase):
    """Requirement 1: each Node returns ONLY the field(s) it owns."""

    def test_research_returns_only_research_notes(self):
        update = research(_initial_state("What is LangGraph?"))
        self.assertEqual(set(update.keys()), {"research_notes"})
        self.assertEqual(len(update["research_notes"]), 2)

    def test_analysis_returns_only_analysis(self):
        state = _initial_state("What is LangGraph?")
        state["research_notes"] = ["Note 1: x", "Note 2: y"]
        update = analysis(state)
        self.assertEqual(set(update.keys()), {"analysis"})
        self.assertIn("Note 1: x", update["analysis"])
        self.assertIn("Note 2: y", update["analysis"])

    def test_finalize_returns_only_final_answer(self):
        state = _initial_state("What is LangGraph?")
        state["analysis"] = "some analysis text"
        update = finalize(state)
        self.assertEqual(set(update.keys()), {"final_answer"})
        self.assertIn("some analysis text", update["final_answer"])


class NodeContentTest(unittest.TestCase):
    """Each Node derives its output only from the State fields it reads."""

    def test_research_notes_are_derived_from_question(self):
        update = research(_initial_state("Hello world"))
        self.assertTrue(any("Hello world" in note for note in update["research_notes"]))
        self.assertTrue(any("2 word(s)" in note for note in update["research_notes"]))

    def test_analysis_is_derived_from_research_notes_count(self):
        state = _initial_state("Q")
        state["research_notes"] = ["a", "b", "c"]
        update = analysis(state)
        self.assertIn("Analyzed 3 research note(s)", update["analysis"])

    def test_finalize_wraps_analysis_text(self):
        state = _initial_state("Q")
        state["analysis"] = "XYZ"
        update = finalize(state)
        self.assertEqual(update["final_answer"], "Final answer -> XYZ")


class StatePropagationTest(unittest.TestCase):
    """Requirement 7: State must flow correctly research -> analysis -> finalize."""

    def test_full_graph_propagates_state_through_all_three_nodes(self):
        graph = build_graph()
        result = graph.invoke(_initial_state("What is State in LangGraph?"))

        # question is untouched all the way through
        self.assertEqual(result["question"], "What is State in LangGraph?")

        # research_notes was written by `research` and survives into the final state
        self.assertEqual(len(result["research_notes"]), 2)
        self.assertTrue(any("What is State in LangGraph?" in n for n in result["research_notes"]))

        # analysis was derived from research_notes
        self.assertIn("Analyzed 2 research note(s)", result["analysis"])
        for note in result["research_notes"]:
            self.assertIn(note, result["analysis"])

        # final_answer was derived from analysis
        self.assertEqual(result["final_answer"], f"Final answer -> {result['analysis']}")

    def test_manual_chaining_matches_graph_invoke(self):
        """Chaining node functions by hand should produce the same final State
        as running the compiled graph -- proving State is passed through
        unchanged aside from each Node's own declared update."""
        state = _initial_state("Manual chain question")

        state = {**state, **research(state)}
        state = {**state, **analysis(state)}
        state = {**state, **finalize(state)}

        graph = build_graph()
        graph_result = graph.invoke(_initial_state("Manual chain question"))

        self.assertEqual(state["research_notes"], graph_result["research_notes"])
        self.assertEqual(state["analysis"], graph_result["analysis"])
        self.assertEqual(state["final_answer"], graph_result["final_answer"])

    def test_different_question_changes_downstream_state(self):
        graph = build_graph()
        result_a = graph.invoke(_initial_state("Short"))
        result_b = graph.invoke(_initial_state("A much longer question here"))

        self.assertNotEqual(result_a["research_notes"], result_b["research_notes"])
        self.assertNotEqual(result_a["analysis"], result_b["analysis"])
        self.assertNotEqual(result_a["final_answer"], result_b["final_answer"])


if __name__ == "__main__":
    unittest.main()
