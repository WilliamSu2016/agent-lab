"""Tests for the LangGraph Prompt Chaining pipeline (src/03_prompt_chaining_graph.py).

Fully offline: a fake ``llm_call`` and a fake ``search_web_call`` are injected,
so no network access and no API key are required.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import json
import unittest

pcg = importlib.import_module("src.03_prompt_chaining_graph")

initial_state = pcg.initial_state
make_planner = pcg.make_planner
make_researcher = pcg.make_researcher
make_analyst = pcg.make_analyst
make_writer = pcg.make_writer
build_graph = pcg.build_graph

QUESTION = "Compare Python, TypeScript, and Go for building AI agents."
DIMENSIONS = ["Concurrency", "Ecosystem", "Tooling"]


def _fake_search_web(query: str) -> dict:
    """Deterministic stand-in for src.tools.web_search -- no network calls."""
    return {
        "query": query,
        "results": [
            {"title": f"Result for {query}", "url": f"https://example.com/{query}", "snippet": "..."}
        ],
    }


def _make_fake_llm_call(calls: list[tuple[str, str]]):
    """A fake LLMCall that records every (system_prompt, user_prompt) it receives
    and returns a scripted, deterministic response depending on which step called it."""

    def fake_llm_call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        # Match on "the X step" (the opening sentence of each step's
        # instructions) so a later step's mention of an earlier step's name
        # (e.g. Writer's instructions mentioning "the Analyst's ... analysis")
        # never causes a false match.
        if "the Planner step" in system_prompt:
            return json.dumps(DIMENSIONS)
        if "the Researcher step" in system_prompt:
            return f"Findings about: {user_prompt.splitlines()[0]}"
        if "the Analyst step" in system_prompt:
            return "ANALYSIS-TEXT"
        if "the Writer step" in system_prompt:
            return "FINAL-ANSWER-TEXT"
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")

    return fake_llm_call


class PlannerNodeTest(unittest.TestCase):
    """Planner Node: question -> dimensions (only)."""

    def test_planner_parses_json_array_into_dimensions(self):
        calls: list[tuple[str, str]] = []
        planner = make_planner(_make_fake_llm_call(calls))

        update = planner(initial_state(QUESTION))

        self.assertEqual(set(update.keys()), {"dimensions"})
        self.assertEqual(update["dimensions"], DIMENSIONS)
        # The planner must send the raw question, unmodified, as the user prompt.
        self.assertEqual(calls[0][1], QUESTION)

    def test_planner_falls_back_to_line_parsing_for_non_json_output(self):
        def fake_llm_call(system_prompt: str, user_prompt: str) -> str:
            return "- Concurrency\n- Ecosystem\n"

        planner = make_planner(fake_llm_call)
        update = planner(initial_state(QUESTION))
        self.assertEqual(update["dimensions"], ["Concurrency", "Ecosystem"])


class ResearcherNodeTest(unittest.TestCase):
    """Researcher Node: dimensions -> research_notes (only). search_web is called
    exactly once per dimension by the fixed pipeline code, not by the LLM."""

    def test_researcher_calls_search_web_once_per_dimension(self):
        search_calls: list[str] = []

        def counting_search_web(query: str) -> dict:
            search_calls.append(query)
            return _fake_search_web(query)

        calls: list[tuple[str, str]] = []
        researcher = make_researcher(_make_fake_llm_call(calls), counting_search_web)

        state = initial_state(QUESTION)
        state["dimensions"] = DIMENSIONS
        update = researcher(state)

        self.assertEqual(set(update.keys()), {"research_notes"})
        self.assertEqual(search_calls, DIMENSIONS)
        self.assertEqual(len(update["research_notes"]), 3)

    def test_researcher_notes_carry_dimension_findings_and_sources(self):
        calls: list[tuple[str, str]] = []
        researcher = make_researcher(_make_fake_llm_call(calls), _fake_search_web)

        state = initial_state(QUESTION)
        state["dimensions"] = ["Concurrency"]
        update = researcher(state)

        note = update["research_notes"][0]
        self.assertEqual(note["dimension"], "Concurrency")
        self.assertIn("Findings about", note["findings"])
        self.assertEqual(note["sources"], ["https://example.com/Concurrency"])


class AnalystNodeTest(unittest.TestCase):
    """Analyst Node: research_notes -> analysis (only)."""

    def test_analyst_returns_only_analysis_field(self):
        calls: list[tuple[str, str]] = []
        analyst = make_analyst(_make_fake_llm_call(calls))

        state = initial_state(QUESTION)
        state["research_notes"] = [
            {"dimension": "Concurrency", "findings": "x", "sources": ["https://a"]}
        ]
        update = analyst(state)

        self.assertEqual(set(update.keys()), {"analysis"})
        self.assertEqual(update["analysis"], "ANALYSIS-TEXT")
        # The analyst's prompt must include the question and the research notes.
        self.assertIn(QUESTION, calls[0][1])
        self.assertIn("Concurrency", calls[0][1])


class WriterNodeTest(unittest.TestCase):
    """Writer Node: analysis -> final_answer (only)."""

    def test_writer_returns_only_final_answer_field(self):
        calls: list[tuple[str, str]] = []
        writer = make_writer(_make_fake_llm_call(calls))

        state = initial_state(QUESTION)
        state["analysis"] = "some analysis"
        update = writer(state)

        self.assertEqual(set(update.keys()), {"final_answer"})
        self.assertEqual(update["final_answer"], "FINAL-ANSWER-TEXT")
        self.assertIn("some analysis", calls[0][1])


class FullGraphTest(unittest.TestCase):
    """The compiled graph must run planner -> researcher -> analyst -> writer,
    in that fixed order, with each step's output feeding the next."""

    def test_invoke_runs_all_four_steps_in_order(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_make_fake_llm_call(calls), _fake_search_web)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["question"], QUESTION)
        self.assertEqual(result["dimensions"], DIMENSIONS)
        self.assertEqual(len(result["research_notes"]), len(DIMENSIONS))
        self.assertEqual(result["analysis"], "ANALYSIS-TEXT")
        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")

        # 1 planner call + 3 researcher calls (one per dimension) + 1 analyst + 1 writer.
        self.assertEqual(len(calls), 6)

    def test_invoke_calls_each_step_in_the_fixed_order(self):
        calls: list[tuple[str, str]] = []
        graph = build_graph(_make_fake_llm_call(calls), _fake_search_web)
        graph.invoke(initial_state(QUESTION))

        step_order = []
        for system_prompt, _ in calls:
            for step in ("Planner", "Researcher", "Analyst", "Writer"):
                if f"the {step} step" in system_prompt:
                    step_order.append(step)
                    break

        self.assertEqual(
            step_order,
            ["Planner", "Researcher", "Researcher", "Researcher", "Analyst", "Writer"],
        )

    def test_rejects_missing_dimensions_gracefully_when_planner_returns_non_list(self):
        def bad_llm_call(system_prompt: str, user_prompt: str) -> str:
            if "Planner" in system_prompt:
                return json.dumps({"not": "a list"})
            return "irrelevant"

        graph = build_graph(bad_llm_call, _fake_search_web)
        with self.assertRaises(ValueError):
            graph.invoke(initial_state(QUESTION))


if __name__ == "__main__":
    unittest.main()
