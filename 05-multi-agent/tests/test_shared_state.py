"""Tests for the LangGraph Shared State pipeline (src/05_shared_state.py).

Fully offline: fake ``TextLLMCall``s are injected for ResearchAgent,
AnalysisAgent, and ReviewAgent, so no network access and no API key are
required. Persistence is exercised through LangGraph's real
``InMemorySaver`` checkpointer -- nothing here re-implements checkpointing.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; it is loaded dynamically with ``importlib``, the same
pattern used by ``tests/test_parallel_multi_agent.py`` for
``04_parallel_multi_agent.py``.
"""

import importlib
import json
import unittest

shared_state_mod = importlib.import_module("src.05_shared_state")

ResearchState = shared_state_mod.ResearchState
initial_state = shared_state_mod.initial_state
make_research_node = shared_state_mod.make_research_node
make_analysis_node = shared_state_mod.make_analysis_node
make_review_node = shared_state_mod.make_review_node
make_finalizer_node = shared_state_mod.make_finalizer_node
route_after_review = shared_state_mod.route_after_review
build_graph = shared_state_mod.build_graph
run_research = shared_state_mod.run_research
DEFAULT_MAX_LOOPS = shared_state_mod.DEFAULT_MAX_LOOPS

QUESTION = "LangGraph 和 OpenAI Agents SDK 应该如何选择？"


def _fake_research_llm_call(reply: str = "research findings"):
    def call(system_prompt: str, user_prompt: str) -> str:
        return reply

    return call


def _fake_analysis_llm_call(replies: list[str]):
    """Return successive replies on each call (one per analysis attempt)."""
    calls = {"n": 0}

    def call(system_prompt: str, user_prompt: str) -> str:
        index = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return replies[index]

    return call


def _fake_review_llm_call(verdicts: list[dict]):
    """Return successive JSON verdicts on each call (one per review attempt)."""
    calls = {"n": 0}

    def call(system_prompt: str, user_prompt: str) -> str:
        index = min(calls["n"], len(verdicts) - 1)
        calls["n"] += 1
        return json.dumps(verdicts[index], ensure_ascii=False)

    return call


def _always_approve_review_llm_call():
    return _fake_review_llm_call([{"approved": True, "feedback": ""}])


def _always_reject_review_llm_call(feedback: str = "not good enough"):
    def call(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"approved": False, "feedback": feedback}, ensure_ascii=False)

    return call


class InitialStateTest(unittest.TestCase):
    def test_initial_state_has_all_required_keys(self):
        state = initial_state(QUESTION)
        for key in (
            "question",
            "research_results",
            "analysis",
            "review",
            "final_answer",
            "loop_count",
            "max_loops",
            "trace",
        ):
            self.assertIn(key, state)

    def test_initial_state_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            initial_state("   ")

    def test_default_max_loops_is_three(self):
        self.assertEqual(DEFAULT_MAX_LOOPS, 3)
        state = initial_state(QUESTION)
        self.assertEqual(state["max_loops"], 3)


class OwnershipTest(unittest.TestCase):
    """Requirement 3: each agent's node only ever returns the field(s) it owns."""

    def test_research_node_only_returns_research_results_and_trace(self):
        node = make_research_node(_fake_research_llm_call("some findings"))
        state = initial_state(QUESTION)
        update = node(state)
        self.assertEqual(set(update.keys()), {"research_results", "trace"})
        self.assertEqual(update["research_results"], "some findings")

    def test_analysis_node_only_returns_analysis_loop_count_and_trace(self):
        node = make_analysis_node(_fake_analysis_llm_call(["first analysis"]))
        state = initial_state(QUESTION)
        state["research_results"] = "some findings"
        update = node(state)
        self.assertEqual(set(update.keys()), {"analysis", "loop_count", "trace"})
        self.assertEqual(update["analysis"], "first analysis")
        self.assertEqual(update["loop_count"], 1)

    def test_review_node_only_returns_review_and_trace(self):
        node = make_review_node(_always_approve_review_llm_call())
        state = initial_state(QUESTION)
        state["research_results"] = "some findings"
        state["analysis"] = "first analysis"
        update = node(state)
        self.assertEqual(set(update.keys()), {"review", "trace"})
        self.assertEqual(update["review"], {"approved": True, "feedback": ""})

    def test_finalizer_node_only_returns_final_answer_and_trace(self):
        node = make_finalizer_node()
        state = initial_state(QUESTION)
        state["analysis"] = "first analysis"
        state["review"] = {"approved": True, "feedback": ""}
        state["loop_count"] = 1
        update = node(state)
        self.assertEqual(set(update.keys()), {"final_answer", "trace"})
        self.assertIn("first analysis", update["final_answer"])

    def test_review_node_never_writes_analysis_even_when_rejecting(self):
        # Requirement 4/5: ReviewAgent can *read* analysis and find fault with
        # it (via feedback), but it must never be able to overwrite the
        # `analysis` field itself -- only AnalysisAgent owns that key.
        node = make_review_node(_always_reject_review_llm_call("logic gap"))
        state = initial_state(QUESTION)
        state["analysis"] = "flawed analysis"
        update = node(state)
        self.assertNotIn("analysis", update)
        self.assertEqual(update["review"]["approved"], False)
        self.assertEqual(update["review"]["feedback"], "logic gap")

    def test_two_independent_states_do_not_leak_into_each_other(self):
        # Requirement 2: no global/module-level variable is used to pass data
        # between agents -- calling the same node factory/function twice with
        # two different states must not let one call's data leak into the
        # other's result.
        node = make_research_node(_fake_research_llm_call("shared-looking reply"))
        state_a = initial_state("question A")
        state_b = initial_state("question B")
        update_a = node(state_a)
        update_b = node(state_b)
        self.assertEqual(state_a["question"], "question A")
        self.assertEqual(state_b["question"], "question B")
        self.assertNotIn("question", update_a)
        self.assertNotIn("question", update_b)


class RouteAfterReviewTest(unittest.TestCase):
    def test_routes_to_finalizer_when_approved(self):
        state = initial_state(QUESTION)
        state["review"] = {"approved": True, "feedback": ""}
        state["loop_count"] = 1
        self.assertEqual(route_after_review(state), "finalizer")

    def test_routes_back_to_analysis_when_rejected_and_budget_remains(self):
        state = initial_state(QUESTION)
        state["review"] = {"approved": False, "feedback": "fix it"}
        state["loop_count"] = 1
        state["max_loops"] = 3
        self.assertEqual(route_after_review(state), "analysis_agent")

    def test_routes_to_finalizer_once_max_loops_reached_even_if_rejected(self):
        # Requirement 6: the loop must not run forever.
        state = initial_state(QUESTION)
        state["review"] = {"approved": False, "feedback": "still not good"}
        state["loop_count"] = 3
        state["max_loops"] = 3
        self.assertEqual(route_after_review(state), "finalizer")


class FullGraphTest(unittest.TestCase):
    def test_happy_path_approved_on_first_review_runs_each_node_once(self):
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["good analysis"]),
            _always_approve_review_llm_call(),
        )
        state = initial_state(QUESTION)
        result = graph.invoke(state)

        self.assertEqual(result["research_results"], "findings")
        self.assertEqual(result["analysis"], "good analysis")
        self.assertEqual(result["review"], {"approved": True, "feedback": ""})
        self.assertEqual(result["loop_count"], 1)
        self.assertIn("good analysis", result["final_answer"])
        self.assertIn("approved", result["final_answer"].lower())

    def test_trace_records_every_agent_in_order_requirement_7(self):
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["good analysis"]),
            _always_approve_review_llm_call(),
        )
        result = graph.invoke(initial_state(QUESTION))
        agent_order = [
            entry.split(":")[0]
            for entry in result["trace"]
            if ":" in entry and not entry.startswith("init")
        ]
        self.assertEqual(
            agent_order,
            ["ResearchAgent", "AnalysisAgent", "ReviewAgent", "Finalizer"],
        )

    def test_review_rejection_loops_back_to_analysis_requirement_5(self):
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["draft one", "revised two"]),
            _fake_review_llm_call(
                [
                    {"approved": False, "feedback": "missing evidence"},
                    {"approved": True, "feedback": ""},
                ]
            ),
        )
        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["loop_count"], 2)
        self.assertEqual(result["analysis"], "revised two")
        self.assertTrue(result["review"]["approved"])
        agent_order = [
            entry.split(":")[0]
            for entry in result["trace"]
            if ":" in entry and not entry.startswith("init")
        ]
        self.assertEqual(
            agent_order,
            ["ResearchAgent", "AnalysisAgent", "ReviewAgent", "AnalysisAgent", "ReviewAgent", "Finalizer"],
        )

    def test_analysis_sees_previous_feedback_when_retrying(self):
        seen_prompts = []

        def analysis_call(system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append(user_prompt)
            return "draft one" if len(seen_prompts) == 1 else "revised two"

        graph = build_graph(
            _fake_research_llm_call("findings"),
            analysis_call,
            _fake_review_llm_call(
                [
                    {"approved": False, "feedback": "missing evidence"},
                    {"approved": True, "feedback": ""},
                ]
            ),
        )
        graph.invoke(initial_state(QUESTION))

        self.assertEqual(len(seen_prompts), 2)
        self.assertNotIn("missing evidence", seen_prompts[0])
        self.assertIn("missing evidence", seen_prompts[1])
        self.assertIn("draft one", seen_prompts[1])

    def test_max_loops_enforced_even_if_review_never_approves_requirement_6(self):
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["v1", "v2", "v3", "v4", "v5"]),
            _always_reject_review_llm_call("still not good enough"),
        )
        state = initial_state(QUESTION, max_loops=3)
        result = graph.invoke(state)

        self.assertEqual(result["loop_count"], 3)
        self.assertFalse(result["review"]["approved"])
        self.assertIn("not approved", result["final_answer"])
        agent_order = [
            entry.split(":")[0]
            for entry in result["trace"]
            if ":" in entry and not entry.startswith("init")
        ]
        self.assertEqual(
            agent_order,
            [
                "ResearchAgent",
                "AnalysisAgent",
                "ReviewAgent",
                "AnalysisAgent",
                "ReviewAgent",
                "AnalysisAgent",
                "ReviewAgent",
                "Finalizer",
            ],
        )

    def test_review_can_flag_a_specific_problem_and_it_reaches_final_answer_requirement_4(self):
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["draft with unsupported claim"]),
            _always_reject_review_llm_call("claim X is not supported by research_results"),
        )
        state = initial_state(QUESTION, max_loops=1)
        result = graph.invoke(state)

        self.assertIn("claim X is not supported by research_results", result["final_answer"])


class ParseReviewTest(unittest.TestCase):
    def test_parses_plain_json(self):
        parse = shared_state_mod._parse_review
        parsed = parse('{"approved": true, "feedback": ""}')
        self.assertEqual(parsed, {"approved": True, "feedback": ""})

    def test_parses_json_in_code_fence(self):
        parse = shared_state_mod._parse_review
        parsed = parse('```json\n{"approved": false, "feedback": "bad"}\n```')
        self.assertEqual(parsed, {"approved": False, "feedback": "bad"})

    def test_falls_back_to_rejecting_on_unparseable_output(self):
        parse = shared_state_mod._parse_review
        parsed = parse("this is not json at all")
        self.assertFalse(parsed["approved"])
        self.assertTrue(parsed["feedback"])


class PersistenceTest(unittest.TestCase):
    """Requirement 8: LangGraph persistence via a checkpointer."""

    def test_state_is_recoverable_after_the_run_via_get_state(self):
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["good analysis"]),
            _always_approve_review_llm_call(),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "thread-1"}}
        graph.invoke(initial_state(QUESTION), config)

        snapshot = graph.get_state(config)
        self.assertEqual(snapshot.values["analysis"], "good analysis")
        self.assertTrue(snapshot.values["review"]["approved"])

    def test_state_history_contains_a_snapshot_per_superstep(self):
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["draft one", "revised two"]),
            _fake_review_llm_call(
                [
                    {"approved": False, "feedback": "missing evidence"},
                    {"approved": True, "feedback": ""},
                ]
            ),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "thread-2"}}
        graph.invoke(initial_state(QUESTION), config)

        history = list(graph.get_state_history(config))
        # One snapshot per superstep executed (research, analysis, review,
        # analysis, review, finalizer) plus the initial input snapshot.
        self.assertGreaterEqual(len(history), 6)

    def test_run_research_returns_a_usable_thread_id_for_later_lookup(self):
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        outcome = run_research(
            QUESTION,
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["good analysis"]),
            _always_approve_review_llm_call(),
            checkpointer=checkpointer,
        )
        graph = build_graph(
            _fake_research_llm_call("findings"),
            _fake_analysis_llm_call(["good analysis"]),
            _always_approve_review_llm_call(),
            checkpointer=checkpointer,
        )
        snapshot = graph.get_state({"configurable": {"thread_id": outcome.thread_id}})
        self.assertEqual(snapshot.values["final_answer"], outcome.final_answer)


class MermaidDiagramTest(unittest.TestCase):
    def test_mermaid_diagram_contains_all_four_nodes(self):
        graph = build_graph(
            _fake_research_llm_call(),
            _fake_analysis_llm_call(["x"]),
            _always_approve_review_llm_call(),
        )
        mermaid = graph.get_graph().draw_mermaid()
        for node_name in ("research", "analysis_agent", "review_agent", "finalizer"):
            self.assertIn(node_name, mermaid)


if __name__ == "__main__":
    unittest.main()
