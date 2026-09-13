"""Tests for src/patterns/parallelization.py -- fully offline, no network/API key.

Also verifies the Reducer requirement: all three researcher Nodes write the
same ``findings`` field concurrently, and none of their results are lost.
"""

import unittest

from src.patterns import parallelization as par


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": f"https://example.com/{query}", "snippet": "y"}]}


def _fake_llm_call(calls: list[tuple[str, str]]):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if "Synthesizer" in system_prompt:
            return "SYNTHESIZED-ANSWER"
        for language in par.LANGUAGES:
            if language in system_prompt:
                return f"{language}-FINDINGS"
        raise AssertionError(system_prompt)

    return call


class ResearcherNodeTest(unittest.TestCase):
    def test_each_researcher_writes_only_its_own_language_finding(self):
        calls: list[tuple[str, str]] = []
        researcher = par.make_researcher("Python", _fake_llm_call(calls), _fake_search_web)
        update = researcher(par.initial_state("Compare languages"))
        self.assertEqual(set(update.keys()), {"findings"})
        self.assertEqual(len(update["findings"]), 1)
        self.assertEqual(update["findings"][0]["language"], "Python")

    def test_researcher_never_mentions_other_languages_in_its_prompt(self):
        calls: list[tuple[str, str]] = []
        researcher = par.make_researcher("Go", _fake_llm_call(calls), _fake_search_web)
        researcher(par.initial_state("Q"))
        system_prompt = calls[0][0]
        self.assertIn("Go", system_prompt)
        self.assertNotIn("Python", system_prompt)
        self.assertNotIn("TypeScript", system_prompt)


class SynthesizerNodeTest(unittest.TestCase):
    def test_synthesizer_writes_only_final_answer(self):
        calls: list[tuple[str, str]] = []
        synthesizer = par.make_synthesizer(_fake_llm_call(calls))
        state = par.initial_state("Q")
        state["findings"] = [{"language": "Python", "findings": "x", "sources": []}]
        update = synthesizer(state)
        self.assertEqual(set(update.keys()), {"final_answer"})
        self.assertEqual(update["final_answer"], "SYNTHESIZED-ANSWER")


class FullGraphTest(unittest.TestCase):
    def test_all_three_researchers_run_and_none_of_their_findings_are_lost(self):
        """This is the Reducer test: without `operator.add` on `findings`, a
        concurrent write from 3 Nodes to the same field would silently drop
        results down to 1 (last-write-wins)."""
        calls: list[tuple[str, str]] = []
        graph = par.build_graph(_fake_llm_call(calls), _fake_search_web)
        result = graph.invoke(par.initial_state("Compare Python, TypeScript, Go"))

        languages_seen = sorted(f["language"] for f in result["findings"])
        self.assertEqual(languages_seen, sorted(par.LANGUAGES))
        self.assertEqual(result["final_answer"], "SYNTHESIZED-ANSWER")

    def test_mermaid_diagram_contains_all_researchers_and_synthesizer(self):
        graph = par.build_graph(_fake_llm_call([]), _fake_search_web)
        mermaid = par.get_mermaid(graph)
        for name in ("research_python", "research_typescript", "research_go", "synthesizer"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
