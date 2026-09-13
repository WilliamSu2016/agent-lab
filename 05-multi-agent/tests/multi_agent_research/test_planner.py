"""Tests for the Planner (src/multi_agent_research/planner.py).

Fully offline: fake ``TextLLMCall``s are injected, no network/API key
required.
"""

import json
import unittest

from src.multi_agent_research.planner import _parse_aspects, fan_out_to_workers, make_planner_node
from src.multi_agent_research.state import empty_review_verdict, initial_state

QUESTION = "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"


def _fake_planner_llm_call(aspects: list[str]):
    def call(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(aspects, ensure_ascii=False)

    return call


class ParseAspectsTest(unittest.TestCase):
    def test_parses_plain_json_array(self):
        self.assertEqual(_parse_aspects('["a", "b"]'), ["a", "b"])

    def test_parses_json_in_code_fence(self):
        self.assertEqual(_parse_aspects('```json\n["a", "b"]\n```'), ["a", "b"])

    def test_falls_back_to_line_splitting(self):
        raw = "1. first aspect\n2. second aspect\n- third aspect"
        self.assertEqual(_parse_aspects(raw), ["first aspect", "second aspect", "third aspect"])


class PlannerNodeTest(unittest.TestCase):
    def test_produces_dynamic_tasks_not_a_hardcoded_list(self):
        # Whatever the LLM proposes becomes the task list -- nothing here
        # hard-codes any specific aspect names.
        aspects = ["Made-up Aspect Alpha", "Totally Different Beta", "Gamma"]
        node = make_planner_node(_fake_planner_llm_call(aspects))
        state = initial_state(QUESTION)
        update = node(state)
        self.assertEqual([t["aspect"] for t in update["tasks"]], aspects)

    def test_only_returns_tasks_iteration_and_trace(self):
        node = make_planner_node(_fake_planner_llm_call(["a", "b"]))
        state = initial_state(QUESTION)
        update = node(state)
        self.assertEqual(set(update.keys()), {"tasks", "iteration", "trace"})

    def test_caps_tasks_at_max_workers(self):
        aspects = [f"aspect-{i}" for i in range(20)]
        node = make_planner_node(_fake_planner_llm_call(aspects))
        state = initial_state(QUESTION, max_workers=4)
        update = node(state)
        self.assertEqual(len(update["tasks"]), 4)

    def test_first_call_sets_iteration_to_one(self):
        node = make_planner_node(_fake_planner_llm_call(["a"]))
        state = initial_state(QUESTION)
        update = node(state)
        self.assertEqual(update["iteration"], 1)

    def test_second_call_bumps_iteration_to_two(self):
        node = make_planner_node(_fake_planner_llm_call(["a"]))
        state = initial_state(QUESTION)
        state["iteration"] = 1
        state["review"] = {**empty_review_verdict(), "missing_aspects": ["gap"], "feedback": "missing gap"}
        update = node(state)
        self.assertEqual(update["iteration"], 2)

    def test_task_ids_are_unique(self):
        node = make_planner_node(_fake_planner_llm_call(["a", "b", "c"]))
        state = initial_state(QUESTION)
        update = node(state)
        ids = [t["task_id"] for t in update["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_raises_when_no_aspects_produced(self):
        node = make_planner_node(_fake_planner_llm_call([]))
        state = initial_state(QUESTION)
        with self.assertRaises(ValueError):
            node(state)

    def test_refine_mode_prompt_includes_missing_aspects_and_feedback(self):
        seen_prompts = []

        def call(system_prompt: str, user_prompt: str) -> str:
            seen_prompts.append((system_prompt, user_prompt))
            return json.dumps(["follow-up aspect"], ensure_ascii=False)

        node = make_planner_node(call)
        state = initial_state(QUESTION)
        state["iteration"] = 1
        state["review"] = {
            **empty_review_verdict(),
            "missing_aspects": ["monetization models"],
            "feedback": "please cover monetization models explicitly",
        }
        update = node(state)

        self.assertEqual(len(seen_prompts), 1)
        system_prompt, user_prompt = seen_prompts[0]
        self.assertIn("monetization models", user_prompt)
        self.assertIn("please cover monetization models explicitly", user_prompt)
        self.assertIn("补充", system_prompt)
        self.assertEqual(update["tasks"][0]["aspect"], "follow-up aspect")
        self.assertIn("gap-filling", update["tasks"][0]["reason"])

    def test_first_iteration_does_not_use_refine_instructions_even_if_review_present(self):
        # An empty/default review (approved False, no missing_aspects) at
        # iteration 0 must still be treated as a fresh decomposition, not a
        # refinement -- otherwise the very first Planner call would
        # incorrectly think there was prior feedback.
        seen_system_prompts = []

        def call(system_prompt: str, user_prompt: str) -> str:
            seen_system_prompts.append(system_prompt)
            return json.dumps(["a"], ensure_ascii=False)

        node = make_planner_node(call)
        state = initial_state(QUESTION)
        node(state)
        self.assertIn("拆解", seen_system_prompts[0])


class FanOutTest(unittest.TestCase):
    def test_emits_one_send_per_task_with_only_its_own_task(self):
        state = initial_state(QUESTION)
        state["tasks"] = [
            {"task_id": "t1", "aspect": "aspect one", "reason": "x"},
            {"task_id": "t2", "aspect": "aspect two", "reason": "y"},
        ]
        sends = fan_out_to_workers(state)
        self.assertEqual(len(sends), 2)
        for send in sends:
            self.assertEqual(send.node, "research_worker")
            self.assertEqual(set(send.arg.keys()), {"question", "task", "per_worker_timeout_seconds"})
        self.assertEqual(sends[0].arg["task"]["task_id"], "t1")
        self.assertEqual(sends[1].arg["task"]["task_id"], "t2")


if __name__ == "__main__":
    unittest.main()
