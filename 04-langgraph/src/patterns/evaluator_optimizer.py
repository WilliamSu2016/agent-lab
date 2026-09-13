"""Evaluator-Optimizer, expressed as a LangGraph ``StateGraph`` with a
Conditional Edge implementing the accept/revise loop.

Anthropic pattern: a Generator drafts an artifact; an independent Evaluator
critiques it against fixed criteria; if it is not good enough, the
Evaluator's feedback is fed back into the Generator for a revision. This
repeats until the score clears a fixed bar or a hard iteration cap is hit.

Graph:

    START -> generator -> evaluator -> should_revise --"revise"--> generator (loop)
                                                      `--"accept"--> finalize --> END
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from typing_extensions import TypedDict

MAX_ITERATIONS = 3
PASS_SCORE = 8

TextLLMCall = Callable[[str, str], str]
EvaluatorLLMCall = Callable[[str, str], dict[str, Any]]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class EvaluatorOptimizerState(TypedDict):
    question: str
    draft: str
    feedback: list[str]
    score: int
    iteration: int
    best_draft: str
    best_score: int
    final_answer: str


def initial_state(question: str) -> EvaluatorOptimizerState:
    return {
        "question": question,
        "draft": "",
        "feedback": [],
        "score": -1,
        "iteration": 0,
        "best_draft": "",
        "best_score": -1,
        "final_answer": "",
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATOR_INSTRUCTIONS = """You are the Generator. Write (or revise) a draft answer to the
user's question. If you are given previous feedback, address every point explicitly in
your revision rather than starting over from scratch."""

EVALUATOR_INSTRUCTIONS = """You are the Evaluator. You NEVER rewrite the draft yourself --
you only judge it. Score the draft from 0 (unacceptable) to 10 (excellent) based on
whether it answers the question, is factually sound, and is well organized. Return ONLY
a JSON object: {"score": <int 0-10>, "feedback": [<short actionable critique>, ...]}."""


def _build_generator_prompt(state: EvaluatorOptimizerState) -> str:
    if state["iteration"] == 0:
        return state["question"]
    feedback_text = "\n".join(f"- {item}" for item in state["feedback"])
    return (
        f"Original question: {state['question']}\n\n"
        f"Previous draft:\n{state['draft']}\n\n"
        f"Evaluator feedback to address:\n{feedback_text}"
    )


def _parse_evaluation(raw_text: str) -> tuple[int, list[str]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluator did not return valid JSON: {raw_text!r}") from exc
    score = int(parsed["score"])
    feedback = list(parsed.get("feedback", []))
    if not 0 <= score <= 10:
        raise ValueError(f"Evaluator score out of range: {score}")
    return score, feedback


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_generator(text_llm_call: TextLLMCall):
    """Owns ``draft`` and ``iteration``. Never sees its own previous score --
    only the feedback text, so it revises based on critique, not a number."""

    def generator(state: EvaluatorOptimizerState) -> EvaluatorOptimizerState:
        draft = text_llm_call(GENERATOR_INSTRUCTIONS, _build_generator_prompt(state))
        return {"draft": draft, "iteration": state["iteration"] + 1}  # type: ignore[typeddict-item]

    return generator


def make_evaluator(evaluator_llm_call: EvaluatorLLMCall):
    """Owns ``score``, ``feedback``, ``best_draft``, ``best_score``. Never
    touches ``draft`` itself -- it only judges the text it is given (the
    Evaluator NEVER rewrites the draft)."""

    def evaluator(state: EvaluatorOptimizerState) -> EvaluatorOptimizerState:
        score, feedback = _parse_evaluation(
            evaluator_llm_call(EVALUATOR_INSTRUCTIONS, f"Question: {state['question']}\n\nDraft:\n{state['draft']}")
        )
        update: EvaluatorOptimizerState = {"score": score, "feedback": feedback}  # type: ignore[typeddict-item]
        if score > state["best_score"]:
            # Requirement (kept from the original pattern): the final report is
            # the BEST-scoring draft seen so far, not necessarily the last one.
            update["best_draft"] = state["draft"]
            update["best_score"] = score
        return update

    return evaluator


def make_finalize():
    """Owns ``final_answer`` only: the best-scoring draft across all iterations."""

    def finalize(state: EvaluatorOptimizerState) -> EvaluatorOptimizerState:
        return {"final_answer": state["best_draft"]}  # type: ignore[typeddict-item]

    return finalize


# ---------------------------------------------------------------------------
# Routing function (NOT a Node): the accept/revise decision is made by THIS
# CODE (score-based, code-enforced), not trusted from the LLM's own opinion.
# ---------------------------------------------------------------------------


def should_revise(state: EvaluatorOptimizerState) -> Literal["revise", "accept"]:
    if state["score"] >= PASS_SCORE:
        return "accept"
    if state["iteration"] >= MAX_ITERATIONS:
        # Hard, code-enforced cap: stop regardless of what the Evaluator says.
        return "accept"
    return "revise"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(text_llm_call: TextLLMCall, evaluator_llm_call: EvaluatorLLMCall):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(EvaluatorOptimizerState)
    graph_builder.add_node("generator", make_generator(text_llm_call))
    graph_builder.add_node("evaluator", make_evaluator(evaluator_llm_call))
    graph_builder.add_node("finalize", make_finalize())

    graph_builder.add_edge(START, "generator")
    graph_builder.add_edge("generator", "evaluator")
    graph_builder.add_conditional_edges(
        "evaluator",
        should_revise,
        {"revise": "generator", "accept": "finalize"},
    )
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()


def get_mermaid(graph) -> str:
    return graph.get_graph().draw_mermaid()
