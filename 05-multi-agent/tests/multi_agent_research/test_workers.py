"""Tests for the Research Worker (src/multi_agent_research/research_worker.py).

Fully offline: fake ``TextLLMCall``s are injected, no network/API key
required.
"""

import time
import unittest

from src.multi_agent_research.research_worker import make_worker_node

QUESTION = "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"


def _task(task_id: str = "t1", aspect: str = "aspect one", reason: str = "initial decomposition"):
    return {"task_id": task_id, "aspect": aspect, "reason": reason}


def _worker_input(task=None, timeout=5.0):
    return {
        "question": QUESTION,
        "task": task or _task(),
        "per_worker_timeout_seconds": timeout,
    }


class WorkerNodeTest(unittest.TestCase):
    def test_only_returns_worker_results_and_trace(self):
        node = make_worker_node(lambda system_prompt, user_prompt: "some findings")
        update = node(_worker_input())
        self.assertEqual(set(update.keys()), {"worker_results", "trace"})

    def test_success_produces_structured_completed_result(self):
        node = make_worker_node(lambda system_prompt, user_prompt: "detailed findings")
        update = node(_worker_input(task=_task("t9", "my aspect")))
        [result] = update["worker_results"]
        self.assertEqual(result["task_id"], "t9")
        self.assertEqual(result["aspect"], "my aspect")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["findings"], "detailed findings")
        self.assertIsNone(result["error"])

    def test_only_sees_its_own_task_never_a_task_list(self):
        seen = []

        def call(system_prompt: str, user_prompt: str) -> str:
            seen.append(user_prompt)
            return "ok"

        node = make_worker_node(call)
        node(_worker_input(task=_task("t1", "aspect-A")))
        self.assertEqual(seen, ["aspect-A"])

    def test_worker_times_out_instead_of_hanging(self):
        def slow_call(system_prompt: str, user_prompt: str) -> str:
            time.sleep(2.0)
            return "too slow"

        node = make_worker_node(slow_call)
        start = time.monotonic()
        update = node(_worker_input(timeout=0.2))
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 1.5)
        [result] = update["worker_results"]
        self.assertEqual(result["status"], "timeout")
        self.assertIn("timeout", result["error"].lower())

    def test_worker_never_raises_on_llm_failure(self):
        def failing_call(system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("simulated failure")

        node = make_worker_node(failing_call)
        update = node(_worker_input())  # must not raise
        [result] = update["worker_results"]
        self.assertEqual(result["status"], "failed")
        self.assertIn("simulated failure", result["error"])


if __name__ == "__main__":
    unittest.main()
