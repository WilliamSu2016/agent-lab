"""Tests for the LangGraph Agent Loop (src/04_agent_loop.py).

Fully offline: a fake ``llm_call`` (scripted decisions) and a fake
``search_web_call`` are injected, so no network access and no API key are
required.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

agent_loop = importlib.import_module("src.04_agent_loop")

initial_state = agent_loop.initial_state
make_agent_node = agent_loop.make_agent_node
make_tools_node = agent_loop.make_tools_node
should_continue = agent_loop.should_continue
build_graph = agent_loop.build_graph
MAX_STEPS = agent_loop.MAX_STEPS

QUESTION = "What is the capital of France?"


class ScriptedLLM:
    """A fake LLMCall that returns a fixed sequence of decisions, one per call."""

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        if not self._decisions:
            raise AssertionError("ScriptedLLM ran out of scripted decisions")
        return self._decisions.pop(0)


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": "https://example.com", "snippet": "y"}]}


class RoutingFunctionTest(unittest.TestCase):
    """Requirement 4: should_continue is a plain function of State."""

    def test_routes_to_tools_when_pending_tool_call_is_set(self):
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "x"}}
        self.assertEqual(should_continue(state), "tools")

    def test_routes_to_end_when_no_pending_tool_call(self):
        state = initial_state(QUESTION)
        state["pending_tool_call"] = None
        self.assertEqual(should_continue(state), "end")


class AgentNodeTest(unittest.TestCase):
    def test_agent_sets_pending_tool_call_when_llm_requests_a_tool(self):
        llm = ScriptedLLM([{"tool_call": {"name": "search_web", "args": {"query": "Paris"}}}])
        agent = make_agent_node(llm)

        update = agent(initial_state(QUESTION))

        self.assertEqual(update["pending_tool_call"], {"name": "search_web", "args": {"query": "Paris"}})
        self.assertEqual(update["final_answer"], "")
        self.assertEqual(update["steps"], 1)

    def test_agent_sets_final_answer_when_llm_is_done(self):
        llm = ScriptedLLM([{"final_answer": "Paris is the capital of France."}])
        agent = make_agent_node(llm)

        update = agent(initial_state(QUESTION))

        self.assertIsNone(update["pending_tool_call"])
        self.assertEqual(update["final_answer"], "Paris is the capital of France.")

    def test_agent_forces_a_final_answer_once_max_steps_is_exceeded(self):
        llm = ScriptedLLM([])  # should never be called once the cap kicks in
        agent = make_agent_node(llm, max_steps=2)

        state = initial_state(QUESTION)
        state["steps"] = 2  # already at the cap

        update = agent(state)

        self.assertIsNone(update["pending_tool_call"])
        self.assertIn("maximum", update["final_answer"].lower())
        self.assertEqual(llm.calls, [])


class ToolsNodeTest(unittest.TestCase):
    def test_tools_node_executes_the_pending_tool_call(self):
        tools = make_tools_node(_fake_search_web)
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "Paris"}}

        update = tools(state)

        self.assertIsNone(update["pending_tool_call"])
        last_message = update["messages"][-1]
        self.assertEqual(last_message["role"], "tool")
        self.assertEqual(last_message["content"]["query"], "Paris")

    def test_tools_node_catches_errors_and_reports_them_as_a_safe_message(self):
        def failing_search(query: str) -> dict:
            raise RuntimeError("network is down")

        tools = make_tools_node(failing_search)
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "search_web", "args": {"query": "Paris"}}

        update = tools(state)  # must not raise

        self.assertIsNone(update["pending_tool_call"])
        last_message = update["messages"][-1]
        self.assertEqual(last_message["role"], "tool")
        self.assertIn("network is down", last_message["content"]["error"])

    def test_tools_node_reports_unknown_tool_names_safely(self):
        tools = make_tools_node(_fake_search_web)
        state = initial_state(QUESTION)
        state["pending_tool_call"] = {"name": "unknown_tool", "args": {}}

        update = tools(state)  # must not raise

        last_message = update["messages"][-1]
        self.assertIn("Unknown tool", last_message["content"]["error"])


class FullGraphTest(unittest.TestCase):
    """Requirement 6: tool result goes back to agent. Requirement 7: agent's
    final answer is the only way to reach END."""

    def test_answers_immediately_when_no_tool_call_is_requested(self):
        llm = ScriptedLLM([{"final_answer": "Paris."}])
        graph = build_graph(llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["final_answer"], "Paris.")
        self.assertEqual(result["steps"], 1)
        self.assertEqual(len(llm.calls), 1)

    def test_tool_result_loops_back_to_agent_before_finishing(self):
        llm = ScriptedLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "capital of France"}}},
                {"final_answer": "Paris is the capital of France."},
            ]
        )
        graph = build_graph(llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["final_answer"], "Paris is the capital of France.")
        self.assertIsNone(result["pending_tool_call"])
        self.assertEqual(result["steps"], 2)
        # agent ran twice: once to request the tool, once to answer using its result.
        self.assertEqual(len(llm.calls), 2)
        # The second agent call must have seen the tool's result in its messages.
        second_call_messages = llm.calls[1]
        self.assertTrue(any(m["role"] == "tool" for m in second_call_messages))

    def test_multiple_tool_calls_all_loop_back_before_finishing(self):
        llm = ScriptedLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "q1"}}},
                {"tool_call": {"name": "search_web", "args": {"query": "q2"}}},
                {"final_answer": "done"},
            ]
        )
        graph = build_graph(llm, _fake_search_web)

        result = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 50})

        self.assertEqual(result["final_answer"], "done")
        self.assertEqual(result["steps"], 3)
        tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)

    def test_a_failing_tool_does_not_crash_the_graph_and_agent_can_still_finish(self):
        def failing_search(query: str) -> dict:
            raise RuntimeError("boom")

        llm = ScriptedLLM(
            [
                {"tool_call": {"name": "search_web", "args": {"query": "q1"}}},
                {"final_answer": "recovered"},
            ]
        )
        graph = build_graph(llm, failing_search)

        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["final_answer"], "recovered")
        tool_message = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertIn("boom", tool_message["content"]["error"])

    def test_max_steps_cap_stops_a_never_ending_tool_requester(self):
        # A model that always requests a tool call, forever, must still be
        # stopped by the max-steps cap instead of looping indefinitely.
        decisions = [{"tool_call": {"name": "search_web", "args": {"query": "q"}}}] * (MAX_STEPS + 5)
        llm = ScriptedLLM(decisions)
        graph = build_graph(llm, _fake_search_web, max_steps=3)

        result = graph.invoke(initial_state(QUESTION), config={"recursion_limit": 50})

        self.assertIn("maximum", result["final_answer"].lower())
        self.assertIsNone(result["pending_tool_call"])
        self.assertLessEqual(result["steps"], 4)


if __name__ == "__main__":
    unittest.main()
