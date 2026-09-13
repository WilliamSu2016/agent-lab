"""Tests for src/patterns/prompt_chaining.py -- fully offline (fake LLM + fake
search), no network access and no API key required."""

import json
import unittest

from src.patterns import prompt_chaining as pc


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": f"https://example.com/{query}", "snippet": "y"}]}


def _fake_llm_call(calls: list[tuple[str, str]]):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if "You are the Planner" in system_prompt:
            return json.dumps(["Concurrency", "Ecosystem"])
        if "You are the Researcher" in system_prompt:
            return f"Findings for: {user_prompt.splitlines()[0]}"
        if "You are the Writer" in system_prompt:
            return "FINAL"
        if "You are the Analyst" in system_prompt:
            return "ANALYSIS"
        raise AssertionError(system_prompt)

    return call


class NodeTest(unittest.TestCase):
    def test_planner_writes_only_dimensions(self):
        calls: list[tuple[str, str]] = []
        planner = pc.make_planner(_fake_llm_call(calls))
        update = planner(pc.initial_state("Q"))
        self.assertEqual(set(update.keys()), {"dimensions"})
        self.assertEqual(update["dimensions"], ["Concurrency", "Ecosystem"])

    def test_researcher_writes_only_research_notes(self):
        calls: list[tuple[str, str]] = []
        researcher = pc.make_researcher(_fake_llm_call(calls), _fake_search_web)
        state = pc.initial_state("Q")
        state["dimensions"] = ["Concurrency"]
        update = researcher(state)
        self.assertEqual(set(update.keys()), {"research_notes"})
        self.assertEqual(update["research_notes"][0]["dimension"], "Concurrency")

    def test_analyst_writes_only_analysis(self):
        calls: list[tuple[str, str]] = []
        analyst = pc.make_analyst(_fake_llm_call(calls))
        state = pc.initial_state("Q")
        state["research_notes"] = [{"dimension": "X", "findings": "y", "sources": []}]
        update = analyst(state)
        self.assertEqual(set(update.keys()), {"analysis"})
        self.assertEqual(update["analysis"], "ANALYSIS")

    def test_writer_writes_only_final_answer(self):
        calls: list[tuple[str, str]] = []
        writer = pc.make_writer(_fake_llm_call(calls))
        state = pc.initial_state("Q")
        state["analysis"] = "a"
        update = writer(state)
        self.assertEqual(set(update.keys()), {"final_answer"})
        self.assertEqual(update["final_answer"], "FINAL")


class GraphTest(unittest.TestCase):
    def test_full_chain_runs_in_fixed_order(self):
        calls: list[tuple[str, str]] = []
        graph = pc.build_graph(_fake_llm_call(calls), _fake_search_web)
        result = graph.invoke(pc.initial_state("Compare A and B"))

        self.assertEqual(result["dimensions"], ["Concurrency", "Ecosystem"])
        self.assertEqual(len(result["research_notes"]), 2)
        self.assertEqual(result["analysis"], "ANALYSIS")
        self.assertEqual(result["final_answer"], "FINAL")

    def test_mermaid_diagram_contains_all_four_nodes(self):
        graph = pc.build_graph(_fake_llm_call([]), _fake_search_web)
        mermaid = pc.get_mermaid(graph)
        for name in ("planner", "researcher", "analyst", "writer"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
