"""Tests for the LangGraph Parallel Multi-Agent Research pipeline
(src/04_parallel_multi_agent.py).

Fully offline: fake ``TextLLMCall``s are injected for the Planner, every
Worker, and the Synthesizer, so no network access and no API key are
required. All concurrency, fan-out, and fan-in behavior is exercised through
the real, compiled LangGraph graph -- nothing here re-implements LangGraph's
scheduling.

The module name starts with a digit, so it cannot be imported with a plain
``import`` statement; it is loaded dynamically with ``importlib`` -- the same
pattern used by ``tests/test_research_agent.py`` for ``05_research_agent.py``
and by ``tests/test_handoff.py`` for ``02_handoff.py``.
"""

import importlib
import json
import time
import unittest

parallel_mod = importlib.import_module("src.04_parallel_multi_agent")

initial_state = parallel_mod.initial_state
make_planner_node = parallel_mod.make_planner_node
fan_out_to_workers = parallel_mod.fan_out_to_workers
make_worker_node = parallel_mod.make_worker_node
make_synthesizer_node = parallel_mod.make_synthesizer_node
build_graph = parallel_mod.build_graph
run_parallel_research = parallel_mod.run_parallel_research
ParallelResearchTimeoutError = parallel_mod.ParallelResearchTimeoutError
MAX_WORKERS = parallel_mod.MAX_WORKERS

TOPIC = "全面研究 AI Agent Framework。"


def _fake_planner_llm_call(aspects: list[str]):
    def call(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(aspects, ensure_ascii=False)

    return call


def _fake_worker_llm_call(summaries: dict[str, str] | None = None, *, default: str = "OK"):
    """Return findings keyed by aspect; unknown aspects get ``default``."""
    summaries = summaries or {}

    def call(system_prompt: str, user_prompt: str) -> str:
        return summaries.get(user_prompt, default)

    return call


def _slow_worker_llm_call(delay_seconds: float, *, reply: str = "slow but done"):
    def call(system_prompt: str, user_prompt: str) -> str:
        time.sleep(delay_seconds)
        return reply

    return call


def _flaky_worker_llm_call(fail_for_aspects: set[str]):
    def call(system_prompt: str, user_prompt: str) -> str:
        if user_prompt in fail_for_aspects:
            raise RuntimeError(f"simulated failure for {user_prompt!r}")
        return f"findings for {user_prompt}"

    return call


def _fake_synthesizer_llm_call(reply: str = "FINAL-REPORT"):
    def call(system_prompt: str, user_prompt: str) -> str:
        return reply

    return call


class PlannerNodeTest(unittest.TestCase):
    def test_planner_produces_dynamic_tasks_not_a_hardcoded_list(self):
        # Requirement 1: whatever the LLM proposes becomes the task list --
        # nothing here hard-codes "OpenAI Agents SDK"/"LangGraph"/etc.
        aspects = ["某个从未出现在代码里的自定义研究方面 A", "自定义方面 B"]
        planner = make_planner_node(_fake_planner_llm_call(aspects))

        update = planner(initial_state(TOPIC))

        self.assertEqual([t["aspect"] for t in update["tasks"]], aspects)
        for task in update["tasks"]:
            self.assertIn("task_id", task)

    def test_planner_caps_tasks_at_max_workers(self):
        many_aspects = [f"aspect-{i}" for i in range(MAX_WORKERS + 5)]
        planner = make_planner_node(_fake_planner_llm_call(many_aspects))

        state = initial_state(TOPIC, max_workers=3)
        update = planner(state)

        self.assertEqual(len(update["tasks"]), 3)
        self.assertEqual([t["aspect"] for t in update["tasks"]], many_aspects[:3])

    def test_planner_parses_non_json_fallback_format(self):
        def call(system_prompt: str, user_prompt: str) -> str:
            return "1. Aspect one\n2. Aspect two\n- Aspect three"

        planner = make_planner_node(call)
        update = planner(initial_state(TOPIC))

        aspects = [t["aspect"] for t in update["tasks"]]
        self.assertEqual(aspects, ["Aspect one", "Aspect two", "Aspect three"])

    def test_planner_raises_when_no_aspects_produced(self):
        planner = make_planner_node(_fake_planner_llm_call([]))
        with self.assertRaises(ValueError):
            planner(initial_state(TOPIC))

    def test_task_ids_are_unique(self):
        aspects = ["A", "B", "C"]
        planner = make_planner_node(_fake_planner_llm_call(aspects))
        update = planner(initial_state(TOPIC))

        ids = [t["task_id"] for t in update["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))


class FanOutTest(unittest.TestCase):
    def test_fan_out_emits_one_send_per_task_with_only_its_own_task(self):
        from langgraph.types import Send

        state = initial_state(TOPIC)
        state["tasks"] = [
            {"task_id": "t1", "aspect": "aspect one"},
            {"task_id": "t2", "aspect": "aspect two"},
        ]

        sends = fan_out_to_workers(state)

        self.assertEqual(len(sends), 2)
        for send in sends:
            self.assertIsInstance(send, Send)
            self.assertEqual(send.node, "worker")
            # Requirement 4: each Send payload carries only its own task, not
            # the full task list and not any other worker's data.
            self.assertIn("task", send.arg)
            self.assertNotIn("tasks", send.arg)
        self.assertEqual({s.arg["task"]["aspect"] for s in sends}, {"aspect one", "aspect two"})


class WorkerNodeTest(unittest.TestCase):
    def test_worker_returns_completed_result_for_its_own_task_only(self):
        worker = make_worker_node(_fake_worker_llm_call({"my aspect": "some findings"}))

        update = worker(
            {"topic": TOPIC, "task": {"task_id": "t1", "aspect": "my aspect"}, "per_worker_timeout_seconds": 5.0}
        )

        self.assertEqual(len(update["results"]), 1)
        result = update["results"][0]
        self.assertEqual(result["task_id"], "t1")
        self.assertEqual(result["aspect"], "my aspect")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["summary"], "some findings")
        self.assertIsNone(result["error"])

    def test_worker_never_raises_on_llm_failure_requirement_11(self):
        worker = make_worker_node(_flaky_worker_llm_call({"bad aspect"}))

        update = worker(
            {"topic": TOPIC, "task": {"task_id": "t1", "aspect": "bad aspect"}, "per_worker_timeout_seconds": 5.0}
        )  # must not raise

        result = update["results"][0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("simulated failure", result["error"])

    def test_worker_times_out_instead_of_hanging_requirement_10_and_11(self):
        worker = make_worker_node(_slow_worker_llm_call(delay_seconds=2.0))

        started = time.perf_counter()
        update = worker(
            {"topic": TOPIC, "task": {"task_id": "t1", "aspect": "slow aspect"}, "per_worker_timeout_seconds": 0.2}
        )
        elapsed = time.perf_counter() - started

        result = update["results"][0]
        self.assertEqual(result["status"], "timeout")
        self.assertIn("timeout", result["error"].lower())
        # The node must return promptly at the configured timeout, not wait
        # for the full 2-second delayed call to finish.
        self.assertLess(elapsed, 1.5)


class SynthesizerNodeTest(unittest.TestCase):
    def test_synthesizer_combines_all_results_including_failures(self):
        synthesizer = make_synthesizer_node(_fake_synthesizer_llm_call())

        state = initial_state(TOPIC)
        state["results"] = [
            {"task_id": "t1", "aspect": "A", "status": "completed", "summary": "findings A", "error": None},
            {"task_id": "t2", "aspect": "B", "status": "failed", "summary": "", "error": "boom"},
        ]

        update = synthesizer(state)

        self.assertEqual(set(update.keys()), {"final_report"})
        self.assertEqual(update["final_report"], "FINAL-REPORT")

    def test_synthesizer_prompt_mentions_failed_aspects_explicitly(self):
        captured: list[str] = []

        def call(system_prompt: str, user_prompt: str) -> str:
            captured.append(user_prompt)
            return "ok"

        synthesizer = make_synthesizer_node(call)
        state = initial_state(TOPIC)
        state["results"] = [
            {"task_id": "t1", "aspect": "A", "status": "timeout", "summary": "", "error": "Worker exceeded timeout"}
        ]
        synthesizer(state)

        self.assertIn("TIMEOUT", captured[0])
        self.assertIn("Worker exceeded timeout", captured[0])


class FullGraphFanOutFanInTest(unittest.TestCase):
    def test_full_pipeline_dynamic_number_of_tasks(self):
        aspects = ["自定义方面一", "自定义方面二", "自定义方面三"]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _fake_worker_llm_call(default="worker finding"),
            _fake_synthesizer_llm_call(),
        )

        result = graph.invoke(initial_state(TOPIC))

        self.assertEqual(len(result["tasks"]), 3)
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual({r["aspect"] for r in result["results"]}, set(aspects))
        self.assertEqual(result["final_report"], "FINAL-REPORT")

    def test_synthesizer_runs_exactly_once_after_all_workers_fan_in(self):
        synthesizer_call_count = {"count": 0}

        def counting_synthesizer(system_prompt: str, user_prompt: str) -> str:
            synthesizer_call_count["count"] += 1
            return "FINAL-REPORT"

        aspects = ["A", "B", "C", "D"]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _fake_worker_llm_call(default="finding"),
            counting_synthesizer,
        )

        result = graph.invoke(initial_state(TOPIC))

        self.assertEqual(synthesizer_call_count["count"], 1)
        self.assertEqual(len(result["results"]), 4)

    def test_workers_run_concurrently_not_sequentially(self):
        # Each of 4 workers sleeps 0.5s; if they ran sequentially this would
        # take >=2s, but LangGraph's own fan-out scheduler runs them
        # concurrently, so the whole run should take well under that.
        aspects = ["A", "B", "C", "D"]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _slow_worker_llm_call(delay_seconds=0.5),
            _fake_synthesizer_llm_call(),
        )

        started = time.perf_counter()
        result = graph.invoke(initial_state(TOPIC))
        elapsed = time.perf_counter() - started

        self.assertEqual(len(result["results"]), 4)
        self.assertLess(elapsed, 1.5)  # comfortably less than 4 * 0.5s = 2.0s

    def test_one_failing_worker_does_not_crash_the_whole_run_requirement_11(self):
        aspects = ["good aspect one", "bad aspect", "good aspect two"]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _flaky_worker_llm_call({"bad aspect"}),
            _fake_synthesizer_llm_call(),
        )

        result = graph.invoke(initial_state(TOPIC))  # must not raise

        statuses = {r["aspect"]: r["status"] for r in result["results"]}
        self.assertEqual(statuses["good aspect one"], "completed")
        self.assertEqual(statuses["good aspect two"], "completed")
        self.assertEqual(statuses["bad aspect"], "failed")
        self.assertEqual(result["final_report"], "FINAL-REPORT")

    def test_worker_results_are_never_lost_across_concurrent_branches(self):
        # Regression guard for the fan-in reducer: with many concurrent
        # branches, every single result must still show up (no dropped
        # writes/races in the "results" reducer).
        aspects = [f"aspect-{i}" for i in range(6)]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _fake_worker_llm_call(default="ok"),
            _fake_synthesizer_llm_call(),
        )

        result = graph.invoke(initial_state(TOPIC, max_workers=6))

        self.assertEqual(sorted(r["aspect"] for r in result["results"]), sorted(aspects))


class MaxWorkersCapEnforcedEndToEndTest(unittest.TestCase):
    def test_more_aspects_than_max_workers_are_truncated_requirement_9(self):
        aspects = [f"aspect-{i}" for i in range(20)]
        graph = build_graph(
            _fake_planner_llm_call(aspects),
            _fake_worker_llm_call(default="ok"),
            _fake_synthesizer_llm_call(),
        )

        result = graph.invoke(initial_state(TOPIC, max_workers=4))

        self.assertEqual(len(result["tasks"]), 4)
        self.assertEqual(len(result["results"]), 4)


class RunParallelResearchTest(unittest.TestCase):
    def test_run_parallel_research_returns_outcome(self):
        aspects = ["A", "B"]
        outcome = run_parallel_research(
            TOPIC,
            _fake_planner_llm_call(aspects),
            _fake_worker_llm_call(default="ok"),
            _fake_synthesizer_llm_call(),
        )

        self.assertEqual(len(outcome.tasks), 2)
        self.assertEqual(len(outcome.results), 2)
        self.assertEqual(outcome.final_report, "FINAL-REPORT")

    def test_run_parallel_research_enforces_overall_timeout_requirement_10(self):
        # Simulate a pipeline that would hang past the overall budget (e.g. a
        # stuck Synthesizer) using a per-worker timeout long enough to not
        # trip on its own, but an overall budget short enough to trip first.
        def hanging_synthesizer(system_prompt: str, user_prompt: str) -> str:
            time.sleep(5.0)
            return "too late"

        with self.assertRaises(ParallelResearchTimeoutError):
            run_parallel_research(
                TOPIC,
                _fake_planner_llm_call(["A"]),
                _fake_worker_llm_call(default="ok"),
                hanging_synthesizer,
                per_worker_timeout_seconds=10.0,
                max_total_seconds=0.5,
            )

    def test_run_parallel_research_rejects_empty_topic(self):
        with self.assertRaises(ValueError):
            run_parallel_research(
                "   ",
                _fake_planner_llm_call(["A"]),
                _fake_worker_llm_call(),
                _fake_synthesizer_llm_call(),
            )


class MermaidDiagramTest(unittest.TestCase):
    def test_mermaid_diagram_contains_all_three_nodes(self):
        graph = build_graph(
            _fake_planner_llm_call(["A"]),
            _fake_worker_llm_call(),
            _fake_synthesizer_llm_call(),
        )
        mermaid = parallel_mod.print_mermaid_diagram(graph)

        for node_name in ("planner", "worker", "synthesizer"):
            self.assertIn(node_name, mermaid)


if __name__ == "__main__":
    unittest.main()
