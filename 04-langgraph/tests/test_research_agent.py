"""Tests for the LangGraph Research Agent (src/05_research_agent.py).

Fully offline: fake ``text_llm_call`` / ``agent_llm_call`` / ``search_web_call``
are injected, so no network access and no API key are required.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

research_agent = importlib.import_module("src.05_research_agent")

initial_state = research_agent.initial_state
make_planner_node = research_agent.make_planner_node
make_agent_node = research_agent.make_agent_node
make_tools_node = research_agent.make_tools_node
make_finalize_node = research_agent.make_finalize_node
should_continue = research_agent.should_continue
build_graph = research_agent.build_graph
print_mermaid_diagram = research_agent.print_mermaid_diagram
MAX_STEPS = research_agent.MAX_STEPS

QUESTION = "Compare Python, TypeScript, and Go for building AI agents."


class ScriptedAgentLLM:
    """Fake AgentLLMCall returning a fixed sequence of decisions."""

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        if not self._decisions:
            raise AssertionError("ScriptedAgentLLM ran out of scripted decisions")
        return self._decisions.pop(0)


def _fake_text_llm_call(calls: list[tuple[str, str]]):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if "Planner" in system_prompt:
            return "1. Concurrency\n2. Ecosystem\n3. Tooling"
        if "Finalize" in system_prompt:
            return "FINAL-ANSWER-TEXT"
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")

    return call


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": f"https://example.com/{query}", "snippet": "y"}]}


class PlannerNodeTest(unittest.TestCase):
    def test_planner_writes_plan_and_seeds_messages(self):
        calls: list[tuple[str, str]] = []
        planner = make_planner_node(_fake_text_llm_call(calls))

        update = planner(initial_state(QUESTION))

        self.assertEqual(set(update.keys()), {"plan", "messages"})
        self.assertIn("Concurrency", update["plan"])
        self.assertEqual(update["messages"][-1]["content"], QUESTION)
        self.assertEqual(update["messages"][0]["role"], "system")
        self.assertIn(QUESTION, update["messages"][0]["content"])
        self.assertIn(update["plan"], update["messages"][0]["content"])


class AgentNodeTest(unittest.TestCase):
    def test_agent_requests_a_tool_call(self):
        llm = ScriptedAgentLLM([{"tool_call": {"name": "search_web", "args": {"query": "Go concurrency"}}}])
        agent = make_agent_node(llm)

        state = initial_state(QUESTION)
        state["messages"] = [{"role": "system", "content": "sys"}, {"role": "user", "content": QUESTION}]
        update = agent(state)

        self.assertEqual(update["pending_tool_call"], {"name": "search_web", "args": {"query": "Go concurrency"}})
        self.assertEqual(update["steps"], 1)

    def test_agent_stops_when_llm_has_no_more_tool_calls(self):
        llm = ScriptedAgentLLM([{"reasoning": "enough evidence"}])
        agent = make_agent_node(llm)

        state = initial_state(QUESTION)
        state["messages"] = [{"role": "system", "content": "sys"}]
        update = agent(state)

        self.assertIsNone(update["pending_tool_call"])
        self.assertEqual(update["messages"][-1]["content"], "enough evidence")

    def test_agent_forces_stop_once_max_steps_exceeded(self):
        llm = ScriptedAgentLLM([])  # must not be called
        agent = make_agent_node(llm, max_steps=2)

        state = initial_state(QUESTION)
        state["messages"] = [{"role": "system", "content": "sys"}]
        state["steps"] = 2

        update = agent(state)

        self.assertIsNone(update["pending_tool_call"])
        self.assertEqual(llm.calls, [])
        self.assertIn("maximum", update["messages"][-1]["content"].lower())


class ToolsNodeTest(unittest.TestCase):
    def test_tools_writes_result_into_tool_results_and_messages(self):
        tools = make_tools_node(_fake_search_web)

        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "Go concurrency"}}
        update = tools(state)

        self.assertIsNone(update["pending_tool_call"])
        self.assertEqual(len(update["tool_results"]), 1)
        self.assertEqual(update["tool_results"][0]["query"], "Go concurrency")
        self.assertIn("results", update["tool_results"][0]["result"])
        self.assertEqual(update["messages"][-1]["role"], "tool")

    def test_tools_handles_tool_errors_safely(self):
        def failing_search(query: str) -> dict:
            raise RuntimeError("network down")

        tools = make_tools_node(failing_search)
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "x"}}

        update = tools(state)  # must not raise

        self.assertIn("network down", update["tool_results"][0]["result"]["error"])
        self.assertEqual(update["messages"][-1]["content"]["error"], "network down")

    def test_tools_handles_unknown_tool_name_safely(self):
        tools = make_tools_node(_fake_search_web)
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "unknown", "args": {}}

        update = tools(state)

        self.assertIn("Unknown tool", update["tool_results"][0]["result"]["error"])


class FinalizeNodeTest(unittest.TestCase):
    def test_finalize_returns_only_final_answer_and_uses_tool_results(self):
        calls: list[tuple[str, str]] = []
        finalize = make_finalize_node(_fake_text_llm_call(calls))

        state = initial_state(QUESTION)
        state["plan"] = "1. Concurrency"
        state["tool_results"] = [{"query": "Go concurrency", "result": {"results": []}}]

        update = finalize(state)

        self.assertEqual(set(update.keys()), {"final_answer"})
        self.assertEqual(update["final_answer"], "FINAL-ANSWER-TEXT")
        self.assertIn("Go concurrency", calls[0][1])
        self.assertIn(QUESTION, calls[0][1])


class RoutingFunctionTest(unittest.TestCase):
    def test_routes_to_tools_when_pending_tool_call_set(self):
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "x"}}
        self.assertEqual(should_continue(state), "tools")

    def test_routes_to_finalize_when_no_pending_tool_call(self):
        state = initial_state(QUESTION)
        self.assertEqual(should_continue(state), "finalize")


class FullGraphTest(unittest.TestCase):
    def test_full_pipeline_with_one_search_round(self):
        text_calls: list[tuple[str, str]] = []
        agent_llm = ScriptedAgentLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "Go concurrency"}}},
                {"reasoning": "enough evidence gathered"},
            ]
        )
        graph = build_graph(_fake_text_llm_call(text_calls), agent_llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION))

        self.assertIn("Concurrency", result["plan"])
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(result["tool_results"][0]["query"], "Go concurrency")
        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")
        self.assertIsNone(result["pending_tool_call"])
        self.assertEqual(result["steps"], 2)

    def test_full_pipeline_with_zero_search_rounds(self):
        text_calls: list[tuple[str, str]] = []
        agent_llm = ScriptedAgentLLM([{"reasoning": "no research needed"}])
        graph = build_graph(_fake_text_llm_call(text_calls), agent_llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["tool_results"], [])
        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")

    def test_multiple_search_rounds_all_recorded_in_tool_results(self):
        text_calls: list[tuple[str, str]] = []
        agent_llm = ScriptedAgentLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "q1"}}},
                {"tool_call": {"name": "search_web", "args": {"query": "q2"}}},
                {"reasoning": "done"},
            ]
        )
        graph = build_graph(_fake_text_llm_call(text_calls), agent_llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 50})

        self.assertEqual([r["query"] for r in result["tool_results"]], ["q1", "q2"])
        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")

    def test_max_steps_cap_prevents_infinite_loop(self):
        text_calls: list[tuple[str, str]] = []
        decisions = [{"tool_call": {"name": "search_web", "args": {"query": "q"}}}] * (MAX_STEPS + 5)
        agent_llm = ScriptedAgentLLM(decisions)
        graph = build_graph(_fake_text_llm_call(text_calls), agent_llm, _fake_search_web, max_steps=3)

        result = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 50})

        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")
        self.assertLessEqual(result["steps"], 4)

    def test_a_failing_search_does_not_crash_the_graph(self):
        text_calls: list[tuple[str, str]] = []

        def failing_search(query: str) -> dict:
            raise RuntimeError("boom")

        agent_llm = ScriptedAgentLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "q1"}}},
                {"reasoning": "give up searching, answer with what we have"},
            ]
        )
        graph = build_graph(_fake_text_llm_call(text_calls), agent_llm, failing_search)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["final_answer"], "FINAL-ANSWER-TEXT")
        self.assertIn("boom", result["tool_results"][0]["result"]["error"])


class MermaidDiagramTest(unittest.TestCase):
    def test_mermaid_diagram_contains_all_four_nodes(self):
        graph = build_graph(_fake_text_llm_call([]), ScriptedAgentLLM([{"reasoning": "x"}]), _fake_search_web)
        mermaid = print_mermaid_diagram(graph)

        for node_name in ("planner", "agent", "tools", "finalize"):
            self.assertIn(node_name, mermaid)


if __name__ == "__main__":
    unittest.main()
