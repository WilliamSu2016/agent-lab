"""Experiment 5 (Multi-Agent): Shared State, built on LangGraph.

Three agents -- ResearchAgent, AnalysisAgent, ReviewAgent -- plus a
deterministic Finalizer step, all communicating *only* through one shared,
typed ``ResearchState`` (Requirement 1). There is no module-level/global
mutable variable anywhere in this file (Requirement 2): every node is a
closure created by a ``make_*_node`` factory, and the only state that ever
crosses a node boundary is whatever LangGraph passes in/out of that node
through the graph's state channel.

Graph:

    START
      |
      v
    research    (ResearchAgent node: writes ONLY research_results)
      |
      v
    analysis_agent  (AnalysisAgent node: writes ONLY analysis, bumps loop_count)
      |
      v
    review_agent     (ReviewAgent node: writes ONLY review; can find fault with analysis)
      |
      v (conditional edge)
      +--- review.approved is False AND loop_count < max_loops --> back to analysis_agent
      |
      v (otherwise)
    finalizer   (writes ONLY final_answer)
      |
      v
    END

Design constraints:

* Each node's return value is a *partial* state update containing only the
  key(s) that agent owns (Requirement 3): ``research`` only ever returns
  ``{"research_results": ..., "trace": [...]}"``, ``analysis`` only ever
  returns ``{"analysis": ..., "loop_count": ..., "trace": [...]}"``, etc.
  LangGraph merges partial updates into the shared state for you -- no node
  reads another node's *raw* dict and mutates it in place, and no node ever
  writes a field it does not own (see ``tests/test_shared_state.py``'s
  ``OwnershipTest`` for automated proof of this per node).
* ``review`` is intentionally allowed to *read* ``analysis`` (that is the
  whole point of a review step -- Requirement 4) but can only ever *write*
  its own ``review`` field; it can never edit ``analysis`` itself. If it
  finds a problem it says so in ``review["feedback"]`` and routes control
  back to ``analysis`` (Requirement 5), which re-reads the question, the
  original research, its own previous attempt, and the feedback -- and
  produces a brand new ``analysis`` value itself.
* ``loop_count`` + ``max_loops`` (default 3, Requirement 6) bound the
  analysis<->review cycle: once ``loop_count >= max_loops`` the conditional
  edge forces the run to ``finalizer`` regardless of whether review approved,
  so an obstinate ReviewAgent can never loop forever.
* ``trace`` is an ``Annotated[list[str], operator.add]`` reducer field: every
  node appends its own one-line audit entry (never anyone else's), so the
  full history of state changes is reconstructable from the final state
  alone (Requirement 7) -- and additionally, because the graph is compiled
  with a checkpointer (Requirement 8, ``InMemorySaver``), every intermediate
  snapshot is also recoverable via ``graph.get_state_history(config)``, not
  just the final ``trace`` list.
* No MCP, no supervisor, no handoffs, no agents-as-tools: this is plain
  ``StateGraph`` nodes and edges wired directly, matching every other
  experiment's constraints.
"""

from __future__ import annotations

import json
import operator
import uuid
from dataclasses import dataclass
from typing import Any

from typing_extensions import Annotated, TypedDict

from src.specialists.llm import TextLLMCall

GRAPH_NAME = "Shared State Research Pipeline"

# Requirement 6: hard cap on analysis<->review cycles so a ReviewAgent that
# never approves cannot loop forever.
DEFAULT_MAX_LOOPS = 3


# ---------------------------------------------------------------------------
# Shared state schema (Requirement: ResearchState with at least these keys).
# ---------------------------------------------------------------------------


class ReviewResult(TypedDict):
    """ReviewAgent's verdict on the current ``analysis``."""

    approved: bool
    feedback: str


class ResearchState(TypedDict):
    """The one and only channel every agent communicates through.

    No agent in this module ever reads or writes a global/module-level
    variable to pass information to another agent -- everything flows
    through this dict, merged by LangGraph between node calls.
    """

    question: str
    research_results: str
    analysis: str
    review: ReviewResult
    final_answer: str
    loop_count: int
    max_loops: int
    # Append-only audit log: each node contributes only its own entries via
    # the ``operator.add`` reducer, so every state change is traceable from
    # the final state alone (Requirement 7).
    trace: Annotated[list[str], operator.add]


def initial_state(question: str, max_loops: int = DEFAULT_MAX_LOOPS) -> ResearchState:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    return {
        "question": question,
        "research_results": "",
        "analysis": "",
        "review": {"approved": False, "feedback": ""},
        "final_answer": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "trace": [f"init: question={question!r} max_loops={max_loops}"],
    }


# ---------------------------------------------------------------------------
# Instructions -- one independent instructions block per agent.
# ---------------------------------------------------------------------------

RESEARCH_INSTRUCTIONS = """你是研究者（ResearchAgent），只负责针对用户问题收集相关的事实、
背景资料和关键信息点。不要做任何分析、评价或下结论，只输出你研究到的原始信息要点，
以简洁的要点列表形式呈现。"""

ANALYSIS_INSTRUCTIONS = """你是分析者（AnalysisAgent），会收到一份研究资料（research_results），
你的任务是基于这些资料给出有条理的分析：识别关键结论、权衡利弊、给出有依据的判断。

如果这不是你第一次分析（即上一轮的分析被 Review 打回），你还会收到你自己上一轮的分析
以及 Review 给出的具体反馈（feedback）；你必须针对反馈中指出的问题进行修正，
不能重复上一轮的错误，也不能忽略反馈。"""

REVIEW_INSTRUCTIONS = """你是审阅者（ReviewAgent），负责检查 AnalysisAgent 产出的分析
（analysis）是否有问题：例如结论没有研究资料支撑、逻辑不自洽、遗漏了研究资料中的重要信息、
过于武断或含糊。你不需要，也不应该自己重写分析——你只给出判断和具体、可执行的反馈。

只输出一个 JSON 对象，且只有这一个 JSON 对象，不要有任何其他文字或代码块标记，格式为：
{"approved": true 或 false, "feedback": "具体反馈，approved 为 true 时可以为空字符串"}"""


# ---------------------------------------------------------------------------
# ResearchAgent -- owns ONLY `research_results`.
# ---------------------------------------------------------------------------


def make_research_node(text_llm_call: TextLLMCall):
    """Node: ResearchAgent. Reads only ``question``; writes only ``research_results``."""

    def research(state: ResearchState) -> ResearchState:
        result = text_llm_call(RESEARCH_INSTRUCTIONS, state["question"])
        update: ResearchState = {  # type: ignore[typeddict-item]
            "research_results": result,
            "trace": [f"ResearchAgent: produced research_results ({len(result)} chars)"],
        }
        return update

    return research


# ---------------------------------------------------------------------------
# AnalysisAgent -- owns ONLY `analysis` (and its own loop counter).
# ---------------------------------------------------------------------------


def make_analysis_node(text_llm_call: TextLLMCall):
    """Node: AnalysisAgent. Reads question/research_results/prior analysis+review;
    writes only ``analysis`` and ``loop_count`` (its own retry counter)."""

    def analysis(state: ResearchState) -> ResearchState:
        loop_count = state.get("loop_count", 0) + 1

        parts = [
            f"问题：{state['question']}",
            f"研究资料：\n{state['research_results']}",
        ]
        previous_review = state.get("review") or {"approved": False, "feedback": ""}
        if loop_count > 1 and previous_review.get("feedback"):
            parts.append(f"你上一轮的分析：\n{state.get('analysis', '')}")
            parts.append(f"Review 反馈（必须针对性修正）：\n{previous_review['feedback']}")

        user_prompt = "\n\n".join(parts)
        result = text_llm_call(ANALYSIS_INSTRUCTIONS, user_prompt)

        update: ResearchState = {  # type: ignore[typeddict-item]
            "analysis": result,
            "loop_count": loop_count,
            "trace": [f"AnalysisAgent: produced analysis (attempt {loop_count})"],
        }
        return update

    return analysis


# ---------------------------------------------------------------------------
# ReviewAgent -- owns ONLY `review`. May read `analysis`, must never write it.
# ---------------------------------------------------------------------------


def _parse_review(raw: str) -> ReviewResult:
    """Parse the ReviewAgent's JSON verdict, with a tolerant fallback.

    If the model does not return valid JSON (e.g. adds stray prose), we fall
    back to a conservative heuristic instead of crashing the pipeline: look
    for an explicit approval keyword, otherwise treat the whole reply as
    feedback and reject (safer default than silently approving).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "approved" in parsed:
            return {
                "approved": bool(parsed["approved"]),
                "feedback": str(parsed.get("feedback", "")),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    lowered = raw.strip().lower()
    if lowered.startswith("approved") or lowered == "true":
        return {"approved": True, "feedback": ""}
    return {"approved": False, "feedback": raw.strip() or "Review output was not parseable JSON."}


def make_review_node(text_llm_call: TextLLMCall):
    """Node: ReviewAgent. Reads ``analysis`` (Requirement 4); writes only ``review``."""

    def review(state: ResearchState) -> ResearchState:
        user_prompt = (
            f"问题：{state['question']}\n\n"
            f"研究资料：\n{state['research_results']}\n\n"
            f"待审阅的分析：\n{state['analysis']}"
        )
        raw = text_llm_call(REVIEW_INSTRUCTIONS, user_prompt)
        result = _parse_review(raw)

        update: ResearchState = {  # type: ignore[typeddict-item]
            "review": result,
            "trace": [f"ReviewAgent: approved={result['approved']!r} feedback={result['feedback']!r}"],
        }
        return update

    return review


def route_after_review(state: ResearchState) -> str:
    """Conditional edge (not a node): decide analysis-retry vs finalizer.

    Requirement 5/6: loop back to AnalysisAgent only while review disapproves
    AND the loop budget (``max_loops``) is not exhausted; otherwise stop the
    cycle (either because review approved, or because we must not loop
    forever) and move on to the Finalizer.
    """
    review = state["review"]
    if review["approved"]:
        return "finalizer"
    if state["loop_count"] >= state["max_loops"]:
        return "finalizer"
    return "analysis_agent"


# ---------------------------------------------------------------------------
# Finalizer -- owns ONLY `final_answer`. Deterministic, no LLM call needed.
# ---------------------------------------------------------------------------


def make_finalizer_node():
    """Node: Finalizer. Reads analysis/review/loop_count; writes only ``final_answer``."""

    def finalizer(state: ResearchState) -> ResearchState:
        review = state["review"]
        if review["approved"]:
            final_answer = state["analysis"].strip()
            note = "Review: approved."
        else:
            final_answer = state["analysis"].strip()
            note = (
                f"Review: not approved after {state['loop_count']} attempt(s) "
                f"(max_loops={state['max_loops']}); outstanding feedback: {review['feedback']}"
            )

        update: ResearchState = {  # type: ignore[typeddict-item]
            "final_answer": f"{final_answer}\n\n[{note}]",
            "trace": ["Finalizer: produced final_answer"],
        }
        return update

    return finalizer


# ---------------------------------------------------------------------------
# Graph assembly.
# ---------------------------------------------------------------------------


def build_graph(
    research_llm_call: TextLLMCall,
    analysis_llm_call: TextLLMCall,
    review_llm_call: TextLLMCall,
    checkpointer: Any = None,
):
    """Assemble the StateGraph. Pass a checkpointer to enable persistence
    (Requirement 8); pass ``None`` to compile without one (e.g. in tests that
    don't care about persistence)."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ResearchState)
    builder.add_node("research", make_research_node(research_llm_call))
    builder.add_node("analysis_agent", make_analysis_node(analysis_llm_call))
    builder.add_node("review_agent", make_review_node(review_llm_call))
    builder.add_node("finalizer", make_finalizer_node())

    builder.add_edge(START, "research")
    builder.add_edge("research", "analysis_agent")
    builder.add_edge("analysis_agent", "review_agent")
    builder.add_conditional_edges("review_agent", route_after_review, ["analysis_agent", "finalizer"])
    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)


def print_mermaid_diagram(graph) -> None:
    print(graph.get_graph().draw_mermaid())


# ---------------------------------------------------------------------------
# Public entry point used by main() and by tests that want persistence.
# ---------------------------------------------------------------------------


@dataclass
class SharedStateOutcome:
    question: str
    research_results: str
    analysis: str
    review: ReviewResult
    final_answer: str
    loop_count: int
    trace: list[str]
    thread_id: str


def run_research(
    question: str,
    research_llm_call: TextLLMCall,
    analysis_llm_call: TextLLMCall,
    review_llm_call: TextLLMCall,
    max_loops: int = DEFAULT_MAX_LOOPS,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> SharedStateOutcome:
    """Run the graph once, optionally with a persistent checkpointer.

    When ``checkpointer`` is given, the run's full state history becomes
    recoverable afterwards via ``graph.get_state_history({"configurable":
    {"thread_id": thread_id}})`` -- see ``tests/test_shared_state.py``'s
    ``PersistenceTest`` for a concrete demonstration (Requirement 8).
    """
    graph = build_graph(research_llm_call, analysis_llm_call, review_llm_call, checkpointer=checkpointer)
    state = initial_state(question, max_loops=max_loops)
    thread_id = thread_id or uuid.uuid4().hex

    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, config)

    return SharedStateOutcome(
        question=result["question"],
        research_results=result["research_results"],
        analysis=result["analysis"],
        review=result["review"],
        final_answer=result["final_answer"],
        loop_count=result["loop_count"],
        trace=result["trace"],
        thread_id=thread_id,
    )


def main() -> None:
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    from config import ConfigurationError, load_settings

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    question = " ".join(sys.argv[1:]).strip() or "LangGraph 和 OpenAI Agents SDK 应该如何选择？"

    from langgraph.checkpoint.memory import InMemorySaver

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(settings)
    checkpointer = InMemorySaver()

    graph = build_graph(text_llm_call, text_llm_call, text_llm_call, checkpointer=checkpointer)
    print("=" * 80)
    print("GRAPH (Mermaid)")
    print("=" * 80)
    print_mermaid_diagram(graph)

    outcome = run_research(
        question,
        text_llm_call,
        text_llm_call,
        text_llm_call,
        max_loops=settings.limits.max_iterations,
        checkpointer=checkpointer,
    )

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    print(f"\nRESEARCH_RESULTS:\n{outcome.research_results}")
    print(f"\nANALYSIS:\n{outcome.analysis}")
    print(f"\nREVIEW:\n{outcome.review}")
    print(f"\nFINAL_ANSWER:\n{outcome.final_answer}")
    print(f"\nLOOP_COUNT: {outcome.loop_count}")

    print("\n" + "=" * 80)
    print("TRACE (from final state)")
    print("=" * 80)
    for line in outcome.trace:
        print(f"  - {line}")

    print("\n" + "=" * 80)
    print("STATE HISTORY (from checkpointer persistence)")
    print("=" * 80)
    config = {"configurable": {"thread_id": outcome.thread_id}}
    history = list(graph.get_state_history(config))
    for snapshot in reversed(history):
        node = snapshot.metadata.get("writes") if snapshot.metadata else None
        print(f"  step={snapshot.metadata.get('step') if snapshot.metadata else '?'} writes={list(node.keys()) if node else []}")


if __name__ == "__main__":
    main()
