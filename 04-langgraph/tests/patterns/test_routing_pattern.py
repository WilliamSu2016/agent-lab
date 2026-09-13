"""Tests for src/patterns/routing.py -- fully offline, no network/API key."""

import unittest

from src.patterns import routing


def _fake_llm_call_for(category: str):
    def call(system_prompt: str, user_prompt: str) -> str:
        if "Router" in system_prompt:
            return category
        return f"ANSWER-FROM-{system_prompt.split()[3]}"  # crude marker per specialist

    return call


class RouterNodeTest(unittest.TestCase):
    def test_router_classifies_and_writes_only_category(self):
        router = routing.make_router(_fake_llm_call_for("technical"))
        update = router(routing.initial_state("Compare frameworks"))
        self.assertEqual(set(update.keys()), {"category"})
        self.assertEqual(update["category"], "technical")

    def test_router_rejects_unrecognized_category(self):
        router = routing.make_router(lambda s, u: "unknown-thing")
        with self.assertRaises(ValueError):
            router(routing.initial_state("Q"))


class RoutingFunctionTest(unittest.TestCase):
    def test_route_after_router_returns_the_chosen_category(self):
        state = routing.initial_state("Q")
        state["category"] = "business"
        self.assertEqual(routing.route_after_router(state), "business")


class FullGraphTest(unittest.TestCase):
    def test_only_the_chosen_specialist_runs(self):
        calls: list[str] = []

        def llm_call(system_prompt: str, user_prompt: str) -> str:
            calls.append(system_prompt)
            if "Router" in system_prompt:
                return "technical"
            return "TECH-ANSWER"

        graph = routing.build_graph(llm_call)
        result = graph.invoke(routing.initial_state("Compare frameworks"))

        self.assertEqual(result["category"], "technical")
        self.assertEqual(result["final_answer"], "TECH-ANSWER")
        # Exactly router + one specialist ran -- never all three.
        self.assertEqual(len(calls), 2)
        self.assertIn("Technical Researcher", calls[1])

    def test_different_categories_route_to_different_specialists(self):
        for category, marker in (("business", "Business Researcher"), ("general", "General Researcher")):
            with self.subTest(category=category):
                calls: list[str] = []

                def llm_call(system_prompt: str, user_prompt: str, _category=category) -> str:
                    calls.append(system_prompt)
                    return _category if "Router" in system_prompt else "ANSWER"

                graph = routing.build_graph(llm_call)
                result = graph.invoke(routing.initial_state("Q"))
                self.assertEqual(result["category"], category)
                self.assertIn(marker, calls[1])

    def test_mermaid_diagram_contains_router_and_all_specialists(self):
        graph = routing.build_graph(_fake_llm_call_for("general"))
        mermaid = routing.get_mermaid(graph)
        for name in ("router", "technical", "business", "general"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
