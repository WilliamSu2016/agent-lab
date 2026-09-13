"""Integration tests for the full Multi-Agent Research System graph
(src/multi_agent_research/graph.py).

Fully offline: fake ``TextLLMCall``s are injected for the Planner, every
Research Worker, the Synthesizer, and the Reviewer, so no network access and
no API key are required. Concurrency, fan-out/fan-in, looping, timeouts, and
persistence are all exercised through the real, compiled LangGraph graph.
"""

import json
import time
import unittest

from src.multi_agent_research.graph import build_graph, run_multi_agent_research
from src.multi_agent_research.state import initial_state

QUESTION = "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"


def _planner_llm_call(aspects_by_call: list[list[str]]):
    """Return successive aspect lists on each Planner call (one per
    iteration): first call -> aspects_by_call[0], second -> [1], etc."""
    calls = {"n": 0}

    def call(system_prompt: str, user_prompt: str) -> str:
        index = min(calls["n"], len(aspects_by_call) - 1)
        calls["n"] += 1
        return json.dumps(aspects_by_call[index], ensure_ascii=False)

    return call


def _worker_llm_call(default: str = "some findings", *, delay: float = 0.0, fail_for: set | None = None):
    fail_for = fail_for or set()

    def call(system_prompt: str, user_prompt: str) -> str:
        if delay:
            time.sleep(delay)
        if user_prompt in fail_for:
            raise RuntimeError(f"simulated failure for {user_prompt!r}")
        return f"{default} about {user_prompt}"

    return call


def _synthesizer_llm_call(reply: str = "SYNTHESIS"):
    return lambda system_prompt, user_prompt: reply


def _reviewer_llm_call(verdicts: list[dict]):
    calls = {"n": 0}

    def call(system_prompt: str, user_prompt: str) -> str:
        index = min(calls["n"], len(verdicts) - 1)
        calls["n"] += 1
        return json.dumps(verdicts[index], ensure_ascii=False)

    return call


def _approve_verdict():
    return {"approved": True, "completeness": "ok", "factual_consistency": "ok", "evidence_quality": "ok", "logical_consistency": "ok", "missing_aspects": [], "feedback": ""}


def _reject_verdict(missing=("gap",), feedback="please cover the gap"):
    return {"approved": False, "completeness": "incomplete", "factual_consistency": "ok", "evidence_quality": "weak", "logical_consistency": "ok", "missing_aspects": list(missing), "feedback": feedback}


class HappyPathTest(unittest.TestCase):
    def test_single_iteration_when_review_approves_immediately(self):
        graph = build_graph(
            _planner_llm_call([["aspect A", "aspect B"]]),
            _worker_llm_call(),
            _synthesizer_llm_call("final synthesis"),
            _reviewer_llm_call([_approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION))

        self.assertEqual(result["iteration"], 1)
        self.assertEqual(len(result["tasks"]), 2)
        self.assertEqual(len(result["worker_results"]), 2)
        self.assertTrue(result["review"]["approved"])
        self.assertIn("final synthesis", result["final_answer"])
        self.assertIn("approved", result["final_answer"])

    def test_trace_records_every_agent_in_order(self):
        graph = build_graph(
            _planner_llm_call([["aspect A"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION))
        prefixes = [
            entry.split(":")[0].split("[")[0]
            for entry in result["trace"]
            if ":" in entry and "received" not in entry
        ]
        self.assertEqual(prefixes, ["Supervisor", "Planner", "ResearchWorker", "Synthesizer", "Reviewer", "Supervisor"])


class LoopBackTest(unittest.TestCase):
    def test_review_rejection_loops_back_through_planner_requirement(self):
        graph = build_graph(
            _planner_llm_call([["aspect A"], ["follow-up aspect"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_reject_verdict(missing=["gap X"]), _approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION, max_iterations=3))

        self.assertEqual(result["iteration"], 2)
        self.assertTrue(result["review"]["approved"])
        # worker_results accumulate across iterations (reducer) -- both the
        # first-round and the follow-up task's results are retained.
        self.assertEqual(len(result["worker_results"]), 2)
        aspects_researched = {r["aspect"] for r in result["worker_results"]}
        self.assertEqual(aspects_researched, {"aspect A", "follow-up aspect"})

    def test_planner_sees_reviewer_feedback_when_replanning(self):
        seen_prompts = []

        def planner_call(system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append(user_prompt)
            if len(seen_prompts) == 1:
                return json.dumps(["aspect A"])
            return json.dumps(["gap-filling aspect"])

        graph = build_graph(
            planner_call,
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_reject_verdict(missing=["monetization"], feedback="cover monetization"), _approve_verdict()]),
        )
        graph.invoke(initial_state(QUESTION, max_iterations=3))

        self.assertEqual(len(seen_prompts), 2)
        self.assertIn("monetization", seen_prompts[1])
        self.assertIn("cover monetization", seen_prompts[1])

    def test_max_iterations_enforced_even_if_review_never_approves(self):
        graph = build_graph(
            _planner_llm_call([["a"], ["b"], ["c"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_reject_verdict()]),  # always rejects
        )
        result = graph.invoke(initial_state(QUESTION, max_iterations=3))

        self.assertEqual(result["iteration"], 3)
        self.assertFalse(result["review"]["approved"])
        self.assertIn("NOT approved", result["final_answer"])


class ParallelExecutionTest(unittest.TestCase):
    def test_workers_run_concurrently_not_sequentially(self):
        graph = build_graph(
            _planner_llm_call([["a", "b", "c", "d"]]),
            _worker_llm_call(delay=1.0),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        start = time.monotonic()
        graph.invoke(initial_state(QUESTION, max_workers=4))
        elapsed = time.monotonic() - start
        # 4 workers each sleeping 1s: sequential would be ~4s, parallel ~1s.
        self.assertLess(elapsed, 2.5)

    def test_max_workers_cap_enforced_end_to_end(self):
        many_aspects = [f"aspect-{i}" for i in range(10)]
        graph = build_graph(
            _planner_llm_call([many_aspects]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION, max_workers=3))
        self.assertEqual(len(result["tasks"]), 3)
        self.assertEqual(len(result["worker_results"]), 3)

    def test_each_worker_only_sees_its_own_task(self):
        graph = build_graph(
            _planner_llm_call([["alpha", "beta"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION))
        findings = {r["aspect"]: r["findings"] for r in result["worker_results"]}
        self.assertIn("about alpha", findings["alpha"])
        self.assertIn("about beta", findings["beta"])
        self.assertNotIn("alpha", findings["beta"])
        self.assertNotIn("beta", findings["alpha"])

    def test_one_failing_worker_does_not_crash_the_whole_run(self):
        graph = build_graph(
            _planner_llm_call([["good aspect", "bad aspect"]]),
            _worker_llm_call(fail_for={"bad aspect"}),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        result = graph.invoke(initial_state(QUESTION))
        statuses = {r["aspect"]: r["status"] for r in result["worker_results"]}
        self.assertEqual(statuses["good aspect"], "completed")
        self.assertEqual(statuses["bad aspect"], "failed")


class PersistenceTest(unittest.TestCase):
    def test_state_is_recoverable_after_the_run_via_get_state(self):
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        graph = build_graph(
            _planner_llm_call([["a"]]),
            _worker_llm_call(),
            _synthesizer_llm_call("persisted synthesis"),
            _reviewer_llm_call([_approve_verdict()]),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "thread-graph-1"}}
        graph.invoke(initial_state(QUESTION), config)

        snapshot = graph.get_state(config)
        self.assertEqual(snapshot.values["synthesis"], "persisted synthesis")
        self.assertTrue(snapshot.values["review"]["approved"])

    def test_state_history_has_a_snapshot_per_iteration_round(self):
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        graph = build_graph(
            _planner_llm_call([["a"], ["b"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_reject_verdict(), _approve_verdict()]),
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "thread-graph-2"}}
        graph.invoke(initial_state(QUESTION, max_iterations=3), config)

        history = list(graph.get_state_history(config))
        # supervisor_entry, planner x2, research_worker x2, synthesizer x2,
        # reviewer x2, finalizer, plus the initial input snapshot.
        self.assertGreaterEqual(len(history), 9)


class RunMultiAgentResearchTest(unittest.TestCase):
    def test_returns_a_usable_outcome(self):
        outcome = run_multi_agent_research(
            QUESTION,
            _planner_llm_call([["a"]]),
            _worker_llm_call(),
            _synthesizer_llm_call("s"),
            _reviewer_llm_call([_approve_verdict()]),
        )
        self.assertEqual(outcome.question, QUESTION)
        self.assertEqual(outcome.iteration, 1)
        self.assertTrue(outcome.review["approved"])
        self.assertTrue(outcome.thread_id)

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            run_multi_agent_research(
                "   ",
                _planner_llm_call([["a"]]),
                _worker_llm_call(),
                _synthesizer_llm_call(),
                _reviewer_llm_call([_approve_verdict()]),
            )


class MermaidDiagramTest(unittest.TestCase):
    def test_contains_all_six_nodes(self):
        graph = build_graph(
            _planner_llm_call([["a"]]),
            _worker_llm_call(),
            _synthesizer_llm_call(),
            _reviewer_llm_call([_approve_verdict()]),
        )
        mermaid = graph.get_graph().draw_mermaid()
        for node_name in (
            "supervisor_entry",
            "planner",
            "research_worker",
            "synthesizer",
            "reviewer",
            "finalizer",
        ):
            self.assertIn(node_name, mermaid)


if __name__ == "__main__":
    unittest.main()
