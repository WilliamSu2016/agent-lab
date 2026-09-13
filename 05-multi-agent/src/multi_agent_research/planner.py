"""Planner: dynamically decides what needs to be researched.

Runs twice as many roles as it looks like:

* **Iteration 1** (fresh decomposition): break the user's question into
  independent research aspects, exactly like experiment 4's Planner.
* **Iteration 2+** (gap-filling re-research, driven by the Reviewer's
  ``missing_aspects``/``feedback``): propose a smaller, *targeted* set of
  follow-up tasks instead of redoing everything from scratch -- this is what
  makes the Reviewer's structured feedback actually useful rather than just
  a pass/fail flag.

Ownership: this node only ever returns ``tasks`` and ``iteration`` (plus its
own ``trace`` entry) -- it never touches ``worker_results``, ``synthesis``,
``review``, or ``final_answer``.
"""

from __future__ import annotations

import json
import uuid

from src.multi_agent_research.state import MultiAgentResearchState, ResearchTask
from src.specialists.llm import TextLLMCall

PLANNER_INSTRUCTIONS = """你是一个研究任务规划器（Planner）。给定一个宽泛的研究问题，
你需要把它拆解成若干个具体、互不重叠的"研究方面"（aspect），每个方面都足够具体，
可以交给一个独立的研究者去单独调查，不需要了解其他方面的内容。

不要假设任何固定的候选清单——具体要拆出哪些方面，完全取决于这个问题本身。

只输出一个 JSON 数组，数组元素是字符串，每个字符串是一个研究方面的简短描述
（不超过 20 个字）。不要输出任何其他文字、解释或 Markdown 代码块标记，只输出
JSON 数组本身，例如：

["方面一", "方面二", "方面三"]
"""

PLANNER_REFINE_INSTRUCTIONS = """你是一个研究任务规划器（Planner），这一次你的任务不是
从头拆解问题，而是根据审阅者（Reviewer）指出的具体缺口，生成一组"补充研究任务"。

你会收到：原始问题、审阅者认为缺失的方面列表（missing_aspects）、以及审阅者的
具体反馈（feedback）。请只针对这些缺口生成新的、具体的研究方面，不要重复已经
覆盖过的内容，也不要生成与缺口无关的新方面。

只输出一个 JSON 数组，数组元素是字符串，格式要求与拆解阶段相同，不要输出任何
其他文字：

["补充方面一", "补充方面二"]
"""


def _parse_aspects(raw: str) -> list[str]:
    """Parse the Planner's proposed aspects, tolerating minor formatting
    deviations instead of crashing the pipeline (same pattern as experiment
    4's Planner)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    lines = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.、） )").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def make_planner_node(text_llm_call: TextLLMCall):
    """Node: Planner. Owns ONLY ``tasks`` and ``iteration``."""

    def planner(state: MultiAgentResearchState) -> MultiAgentResearchState:
        iteration = state.get("iteration", 0) + 1
        review = state.get("review")

        if iteration == 1 or not review or not review.get("missing_aspects"):
            raw = text_llm_call(PLANNER_INSTRUCTIONS, state["question"])
            reason = "initial decomposition"
        else:
            user_prompt = (
                f"原始问题：{state['question']}\n\n"
                f"审阅者认为缺失的方面：{json.dumps(review['missing_aspects'], ensure_ascii=False)}\n\n"
                f"审阅者的反馈：{review['feedback']}"
            )
            raw = text_llm_call(PLANNER_REFINE_INSTRUCTIONS, user_prompt)
            reason = f"gap-filling for iteration {iteration} (Reviewer feedback)"

        aspects = _parse_aspects(raw)
        if not aspects:
            raise ValueError("Planner produced no research aspects; cannot continue.")

        capped = aspects[: state["max_workers"]]
        tasks: list[ResearchTask] = [
            {
                "task_id": f"iter{iteration}-task{index}-{uuid.uuid4().hex[:8]}",
                "aspect": aspect,
                "reason": reason,
            }
            for index, aspect in enumerate(capped)
        ]

        update: MultiAgentResearchState = {  # type: ignore[typeddict-item]
            "tasks": tasks,
            "iteration": iteration,
            "trace": [
                f"Planner: iteration {iteration} produced {len(tasks)} task(s) ({reason})"
            ],
        }
        return update

    return planner


def fan_out_to_workers(state: MultiAgentResearchState):
    """Conditional edge (NOT a node): turn each planned task into one ``Send``.

    Each ``Send`` payload carries only that one task -- never the full task
    list, never another worker's in-flight or completed result (same
    isolation guarantee as experiment 4).
    """
    from langgraph.types import Send

    return [
        Send(
            "research_worker",
            {
                "question": state["question"],
                "task": task,
                "per_worker_timeout_seconds": state["per_worker_timeout_seconds"],
            },
        )
        for task in state["tasks"]
    ]
