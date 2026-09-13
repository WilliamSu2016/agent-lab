"""Tests for Human-in-the-loop (src/08_human_in_loop.py).

Fully offline: scripted fake LLM callables, a fake ``search_web``, and a fake
``publish_report`` are injected -- no network access and no API key are
required.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; we load it dynamically with ``importlib``.
"""

import importlib
import unittest

hitl = importlib.import_module("src.08_human_in_loop")

initial_state = hitl.initial_state
build_graph = hitl.build_graph
build_memory_checkpointer = hitl.build_memory_checkpointer
thread_config = hitl.thread_config


def _fake_text_llm_call(system_prompt: str, user_prompt: str) -> str:
    if "Planner" in system_prompt:
        return "Plan: investigate the question."
    if "Generator" in system_prompt:
        if "Human reviewer feedback" in user_prompt:
            return "REVISED REPORT addressing feedback."
        return "DRAFT REPORT v1."
    raise AssertionError(system_prompt)


class ScriptedAgentLLM:
    def __init__(self, decisions: list[dict]) -> None:
        self._decisions = list(decisions)

    def __call__(self, messages: list[dict]) -> dict:
        if not self._decisions:
            raise AssertionError("ScriptedAgentLLM ran out of scripted decisions")
        return self._decisions.pop(0)


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": []}


IMMEDIATELY_DONE = [{"reasoning": "No search needed."}]


class FakePublish:
    """Records every call -- used to prove ``publish`` is only ever reached
    through the ``approve`` branch."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, report: str) -> dict:
        self.calls.append(report)
        return {"status": "published", "length": len(report)}


def _build(checkpointer, publish_call=None):
    return build_graph(
        _fake_text_llm_call,
        ScriptedAgentLLM(IMMEDIATELY_DONE),
        _fake_search_web,
        publish_call or FakePublish(),
        checkpointer=checkpointer,
    )


class PausesForHumanReviewTest(unittest.TestCase):
    """Requirement: after the report is drafted, the graph MUST pause
    before publish_report can ever run."""

    def test_graph_pauses_before_human_review_with_report_visible(self):
        checkpointer = build_memory_checkpointer()
        graph = _build(checkpointer)
        config = thread_config("t1")

        result = graph.invoke(initial_state("Compare X and Y"), config=config)

        state = graph.get_state(config)
        self.assertEqual(state.next, ("human_review",))
        self.assertEqual(result["report"], "DRAFT REPORT v1.")
        self.assertIsNone(result["review_decision"])
        self.assertFalse(result["published"])

    def test_interrupt_payload_carries_the_draft_report(self):
        checkpointer = build_memory_checkpointer()
        graph = _build(checkpointer)
        config = thread_config("t1")
        graph.invoke(initial_state("Compare X and Y"), config=config)

        state = graph.get_state(config)
        interrupts = state.tasks[0].interrupts
        self.assertEqual(len(interrupts), 1)
        self.assertEqual(interrupts[0].value["report"], "DRAFT REPORT v1.")
        self.assertEqual(interrupts[0].value["kind"], "publish_report_approval")


class ApproveTest(unittest.TestCase):
    def test_approve_publishes_and_ends(self):
        from langgraph.types import Command

        checkpointer = build_memory_checkpointer()
        publish_call = FakePublish()
        graph = _build(checkpointer, publish_call)
        config = thread_config("t1")

        graph.invoke(initial_state("Compare X and Y"), config=config)
        result = graph.invoke(Command(resume={"decision": "approve"}), config=config)

        self.assertTrue(result["published"])
        self.assertEqual(result["final_answer"], "DRAFT REPORT v1.")
        self.assertEqual(publish_call.calls, ["DRAFT REPORT v1."])
        self.assertEqual(graph.get_state(config).next, ())


class RejectTest(unittest.TestCase):
    def test_reject_ends_without_publishing(self):
        from langgraph.types import Command

        checkpointer = build_memory_checkpointer()
        publish_call = FakePublish()
        graph = _build(checkpointer, publish_call)
        config = thread_config("t1")

        graph.invoke(initial_state("Compare X and Y"), config=config)
        result = graph.invoke(Command(resume={"decision": "reject"}), config=config)

        self.assertFalse(result["published"])
        self.assertEqual(result["final_answer"], "")
        self.assertEqual(publish_call.calls, [])  # publish_report never called
        self.assertEqual(graph.get_state(config).next, ())


class RequestChangesTest(unittest.TestCase):
    def test_request_changes_loops_back_to_generator_then_can_be_approved(self):
        from langgraph.types import Command

        checkpointer = build_memory_checkpointer()
        publish_call = FakePublish()
        graph = _build(checkpointer, publish_call)
        config = thread_config("t1")

        first = graph.invoke(initial_state("Compare X and Y"), config=config)
        self.assertEqual(first["report"], "DRAFT REPORT v1.")

        # Human asks for changes -- graph must go back to the generator, not to END.
        second = graph.invoke(
            Command(resume={"decision": "request_changes", "comments": "Add more evidence."}),
            config=config,
        )
        self.assertEqual(second["report"], "REVISED REPORT addressing feedback.")
        self.assertEqual(second["review_feedback"], ["Add more evidence."])
        self.assertFalse(second["published"])
        # Paused again for a second round of human review.
        self.assertEqual(graph.get_state(config).next, ("human_review",))
        self.assertEqual(publish_call.calls, [])

        # Now approve the revised report.
        third = graph.invoke(Command(resume={"decision": "approve"}), config=config)
        self.assertTrue(third["published"])
        self.assertEqual(third["final_answer"], "REVISED REPORT addressing feedback.")
        self.assertEqual(publish_call.calls, ["REVISED REPORT addressing feedback."])

    def test_multiple_rounds_of_feedback_accumulate(self):
        from langgraph.types import Command

        checkpointer = build_memory_checkpointer()
        graph = _build(checkpointer)
        config = thread_config("t1")

        graph.invoke(initial_state("Compare X and Y"), config=config)
        graph.invoke(
            Command(resume={"decision": "request_changes", "comments": "first round"}),
            config=config,
        )
        state = graph.get_state(config)
        self.assertEqual(state.values["review_feedback"], ["first round"])

        graph.invoke(
            Command(resume={"decision": "request_changes", "comments": "second round"}),
            config=config,
        )
        state = graph.get_state(config)
        self.assertEqual(state.values["review_feedback"], ["first round", "second round"])


class InvalidDecisionTest(unittest.TestCase):
    def test_unknown_decision_raises(self):
        from langgraph.types import Command

        checkpointer = build_memory_checkpointer()
        graph = _build(checkpointer)
        config = thread_config("t1")
        graph.invoke(initial_state("Compare X and Y"), config=config)

        with self.assertRaises(ValueError):
            graph.invoke(Command(resume={"decision": "maybe"}), config=config)


class PersistenceRequirementTest(unittest.TestCase):
    """Requirement: interrupt + persistence + resume must work together --
    without a checkpointer, the pause cannot be resumed or even observed."""

    def test_without_a_checkpointer_the_pause_cannot_be_inspected(self):
        graph = _build(checkpointer=None)
        config = thread_config("t1")
        result = graph.invoke(initial_state("Compare X and Y"), config=config)

        # The graph silently stops at the interrupt point (no checkpoint to
        # resume from later), and the human's decision never gets applied.
        self.assertIsNone(result["review_decision"])
        self.assertFalse(result["published"])
        with self.assertRaises(ValueError):
            graph.get_state(config)


class MermaidTest(unittest.TestCase):
    def test_mermaid_diagram_contains_all_nodes(self):
        checkpointer = build_memory_checkpointer()
        graph = _build(checkpointer)
        mermaid = graph.get_graph().draw_mermaid()
        for name in ("planner", "agent", "tools", "generator", "human_review", "publish"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
