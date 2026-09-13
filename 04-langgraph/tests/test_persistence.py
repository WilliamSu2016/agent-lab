"""Tests for LangGraph Persistence (src/07_persistence.py).

Fully offline: scripted fake LLM callables and a fake ``search_web`` are
injected, so no network access and no API key are required.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

persistence = importlib.import_module("src.07_persistence")

initial_state = persistence.initial_state
build_graph = persistence.build_graph
build_memory_checkpointer = persistence.build_memory_checkpointer
thread_config = persistence.thread_config


def _fake_text_llm_call(system_prompt: str, user_prompt: str) -> str:
    if "Planner" in system_prompt:
        return f"Plan for: {user_prompt}"
    if "Finalize" in system_prompt:
        return f"Final answer about: {user_prompt.splitlines()[0]}"
    raise AssertionError(system_prompt)


class ScriptedAgentLLM:
    """Returns one scripted decision per call; raises if it runs out."""

    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        if not self._decisions:
            raise AssertionError("ScriptedAgentLLM ran out of scripted decisions")
        return self._decisions.pop(0)


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": "https://example.com", "snippet": "y"}]}


ONE_SEARCH_THEN_DONE = [
    {"tool_call": {"name": "search_web", "args": {"query": "some query"}}},
    {"reasoning": "Research is sufficient."},
]

IMMEDIATELY_DONE = [{"reasoning": "No search needed."}]


class ThreadIsolationTest(unittest.TestCase):
    """Requirement 1: Thread A and Thread B's State must not leak into
    each other, even though they share the same compiled graph and
    checkpointer."""

    def test_two_threads_keep_independent_final_state(self):
        checkpointer = build_memory_checkpointer()
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE + IMMEDIATELY_DONE),
            _fake_search_web,
            checkpointer=checkpointer,
        )

        config_a = thread_config("thread-a")
        config_b = thread_config("thread-b")

        result_a = graph.invoke(initial_state("Question A"), config=config_a)
        result_b = graph.invoke(initial_state("Question B"), config=config_b)

        self.assertEqual(result_a["question"], "Question A")
        self.assertEqual(result_b["question"], "Question B")
        self.assertNotEqual(result_a["final_answer"], result_b["final_answer"])

    def test_reading_back_thread_a_state_is_unaffected_by_thread_b(self):
        checkpointer = build_memory_checkpointer()
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE + IMMEDIATELY_DONE),
            _fake_search_web,
            checkpointer=checkpointer,
        )

        config_a = thread_config("thread-a")
        config_b = thread_config("thread-b")

        graph.invoke(initial_state("Question A"), config=config_a)
        graph.invoke(initial_state("Question B"), config=config_b)

        state_a = graph.get_state(config_a)
        state_b = graph.get_state(config_b)

        self.assertEqual(state_a.values["question"], "Question A")
        self.assertEqual(state_b.values["question"], "Question B")

    def test_same_thread_id_accumulates_the_same_checkpoint_history(self):
        checkpointer = build_memory_checkpointer()
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE),
            _fake_search_web,
            checkpointer=checkpointer,
        )
        config = thread_config("thread-a")
        graph.invoke(initial_state("Question A"), config=config)

        history = list(graph.get_state_history(config))
        # Every super-step (planner, agent, finalize, ...) produced its own
        # checkpoint for this single thread id.
        self.assertGreater(len(history), 1)
        for snapshot in history:
            self.assertEqual(snapshot.config["configurable"]["thread_id"], "thread-a")


class ReadPreviousStateTest(unittest.TestCase):
    """Requirement 3: it must be possible to read back a thread's saved
    State without re-running the graph."""

    def test_get_state_reflects_the_latest_completed_run(self):
        checkpointer = build_memory_checkpointer()
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(ONE_SEARCH_THEN_DONE),
            _fake_search_web,
            checkpointer=checkpointer,
        )
        config = thread_config("thread-a")
        result = graph.invoke(initial_state("Question A"), config=config)

        state = graph.get_state(config)
        self.assertEqual(state.values["final_answer"], result["final_answer"])
        self.assertEqual(len(state.values["tool_results"]), 1)
        # A completed run has no more pending Nodes.
        self.assertEqual(state.next, ())


class ResumeFromInterruptionTest(unittest.TestCase):
    """Requirement 2 & 4: the Agent must be able to pick up from a saved,
    interrupted State instead of starting over."""

    def test_graph_pauses_before_tools_and_resumes_from_there(self):
        checkpointer = build_memory_checkpointer()
        agent_llm = ScriptedAgentLLM(ONE_SEARCH_THEN_DONE)
        graph = build_graph(
            _fake_text_llm_call,
            agent_llm,
            _fake_search_web,
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )
        config = thread_config("thread-a")

        # First invoke: runs planner -> agent, then pauses right before tools.
        first_result = graph.invoke(initial_state("Question A"), config=config)
        state = graph.get_state(config)

        self.assertEqual(state.next, ("tools",))
        self.assertEqual(first_result["tool_results"], [])
        self.assertIsNotNone(state.values["pending_tool_call"])
        # Only the agent's first decision has been made so far.
        self.assertEqual(len(agent_llm.calls), 1)

        # Resume: passing None replays from the last checkpoint, not from START.
        final_result = graph.invoke(None, config=config)

        self.assertEqual(len(final_result["tool_results"]), 1)
        self.assertNotEqual(final_result["final_answer"], "")
        self.assertEqual(len(agent_llm.calls), 2)  # second decision made only after resuming
        self.assertEqual(graph.get_state(config).next, ())

    def test_resuming_a_thread_that_never_paused_is_a_no_op(self):
        checkpointer = build_memory_checkpointer()
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE),
            _fake_search_web,
            checkpointer=checkpointer,
            interrupt_before=["tools"],
        )
        config = thread_config("thread-a")
        result = graph.invoke(initial_state("Question A"), config=config)
        self.assertEqual(graph.get_state(config).next, ())

        # Nothing left to resume -- invoking again with None is a safe no-op.
        resumed = graph.invoke(None, config=config)
        self.assertEqual(resumed["final_answer"], result["final_answer"])


class NoCheckpointerTest(unittest.TestCase):
    """Without a checkpointer, the graph still works exactly like experiment
    5 -- Persistence is opt-in, not a required dependency of the Agent
    Loop itself."""

    def test_graph_runs_fine_with_checkpointer_none(self):
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE),
            _fake_search_web,
        )
        result = graph.invoke(initial_state("Question A"))
        self.assertNotEqual(result["final_answer"], "")

    def test_get_state_requires_a_checkpointer(self):
        graph = build_graph(
            _fake_text_llm_call,
            ScriptedAgentLLM(IMMEDIATELY_DONE),
            _fake_search_web,
        )
        config = thread_config("thread-a")
        graph.invoke(initial_state("Question A"), config=config)
        with self.assertRaises(ValueError):
            graph.get_state(config)


if __name__ == "__main__":
    unittest.main()
