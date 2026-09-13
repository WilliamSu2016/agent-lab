"""Tests for src/patterns/orchestrator_workers.py -- fully offline.

Also verifies: the Orchestrator's dynamically-sized task list drives a
dynamic number of Worker invocations (via `Send`), and the Reducer on
`worker_results` keeps every Worker's result even though they all write
concurrently.
"""

import json
import unittest

from src.patterns import orchestrator_workers as ow


def _fake_search_web(query: str) -> dict:
    return {"query": query, "results": [{"title": "x", "url": "https://example.com", "snippet": "y"}]}


def _fake_llm_call(task_count: int, calls: list[tuple[str, str]]):
    def call(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if "You are the Orchestrator" in system_prompt:
            tasks = [{"task_id": f"task-{i}", "objective": f"objective {i}"} for i in range(task_count)]
            return json.dumps(tasks)
        if "You are a Worker" in system_prompt:
            return f"summary for: {user_prompt.splitlines()[0]}"
        if "You are the Synthesizer" in system_prompt:
            return "FINAL-COMBINED-ANSWER"
        raise AssertionError(system_prompt)

    return call


class OrchestratorNodeTest(unittest.TestCase):
    def test_orchestrator_writes_only_tasks(self):
        calls: list[tuple[str, str]] = []
        orchestrator = ow.make_orchestrator(_fake_llm_call(3, calls))
        update = orchestrator(ow.initial_state("Research X"))
        self.assertEqual(set(update.keys()), {"tasks"})
        self.assertEqual(len(update["tasks"]), 3)

    def test_orchestrator_rejects_more_tasks_than_the_hard_cap(self):
        orchestrator = ow.make_orchestrator(_fake_llm_call(ow.MAX_WORKERS + 1, []))
        with self.assertRaises(ValueError):
            orchestrator(ow.initial_state("Q"))

    def test_orchestrator_rejects_non_json_output(self):
        orchestrator = ow.make_orchestrator(lambda s, u: "not json")
        with self.assertRaises(ValueError):
            orchestrator(ow.initial_state("Q"))


class WorkerNodeTest(unittest.TestCase):
    def test_worker_appends_one_result_for_its_own_task(self):
        calls: list[tuple[str, str]] = []
        worker = ow.make_worker(_fake_llm_call(1, calls), _fake_search_web)
        update = worker({"task": {"task_id": "task-0", "objective": "investigate x"}})
        self.assertEqual(set(update.keys()), {"worker_results"})
        self.assertEqual(update["worker_results"][0]["task_id"], "task-0")


class AssignWorkersTest(unittest.TestCase):
    def test_assign_workers_returns_one_send_per_task(self):
        state = ow.initial_state("Q")
        state["tasks"] = [{"task_id": "t1", "objective": "o1"}, {"task_id": "t2", "objective": "o2"}]
        sends = ow.assign_workers(state)
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0].node, "worker")


class FullGraphTest(unittest.TestCase):
    def test_dynamic_task_count_drives_the_same_number_of_worker_results(self):
        for task_count in (1, 3, ow.MAX_WORKERS):
            with self.subTest(task_count=task_count):
                calls: list[tuple[str, str]] = []
                graph = ow.build_graph(_fake_llm_call(task_count, calls), _fake_search_web)
                result = graph.invoke(ow.initial_state("Research something"))

                self.assertEqual(len(result["tasks"]), task_count)
                # Reducer check: every dynamically spawned Worker's result survives.
                self.assertEqual(len(result["worker_results"]), task_count)
                self.assertEqual(
                    sorted(r["task_id"] for r in result["worker_results"]),
                    sorted(t["task_id"] for t in result["tasks"]),
                )
                self.assertEqual(result["final_answer"], "FINAL-COMBINED-ANSWER")

    def test_mermaid_diagram_contains_all_three_nodes(self):
        graph = ow.build_graph(_fake_llm_call(1, []), _fake_search_web)
        mermaid = ow.get_mermaid(graph)
        for name in ("orchestrator", "worker", "synthesizer"):
            self.assertIn(name, mermaid)


if __name__ == "__main__":
    unittest.main()
