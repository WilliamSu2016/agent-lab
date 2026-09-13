"""Graph assembly for the Multi-Agent Research System.

    User
      |
      v
    supervisor_entry   (Supervisor: coordinates the run -- deterministic,
      |                 no LLM call, just bookkeeping/trace)
      v
    planner             (dynamic Research Tasks; capped at max_workers)
      |
      v (conditional edge -> list[Send], fan-out)
    research_worker x N  (parallel Specialist Research Agents)
      |
      v (LangGraph fan-in: all dispatched instances must finish)
    synthesizer
      |
      v
    reviewer
      |
      v (conditional edge)
      +-- FAIL (not approved) & iteration < max_iterations --> back to planner
      |
      v -- PASS, or iteration budget exhausted
    finalizer            (Supervisor: produces the Final Answer)
      |
      v
    END

This module is the "Supervisor" in the architecture diagram in the sense
that it owns the overall coordination: there is no separate
``supervisor.py`` file because the Supervisor's job here is exactly what
``StateGraph`` orchestration + two small deterministic nodes
(``supervisor_entry``, ``finalizer``) already do -- it is not itself an LLM
agent, it is the fixed control flow around the LLM-backed agents.

No MCP, no external DB, no long-term memory/RAG, no human-in-the-loop --
all state lives only for the duration of one ``graph.invoke()`` call (unless
a checkpointer is supplied purely for persistence/traceability, exactly like
experiment 5; a checkpointer is not "memory" in the RAG/long-term sense --
see docs/14-MULTI-AGENT-RESEARCH.md, question 9's note).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.multi_agent_research.planner import fan_out_to_workers, make_planner_node
from src.multi_agent_research.research_worker import make_worker_node
from src.multi_agent_research.reviewer import make_reviewer_node, route_after_review
from src.multi_agent_research.state import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
    MultiAgentResearchState,
    ReviewVerdict,
    WorkerResult,
    initial_state,
)
from src.multi_agent_research.synthesizer import make_synthesizer_node
from src.reliability.errors import ControlledFailure
from src.reliability.timeout import ToolTimeoutError, run_with_timeout
from src.specialists.llm import TextLLMCall

GRAPH_NAME = "Multi-Agent Research System"


def make_supervisor_entry_node():
    """Node: Supervisor (entry). Deterministic -- no LLM call. Its only job
    is to record that coordination has started; it never writes any domain
    field (``tasks``/``synthesis``/``review``/``final_answer``), only its
    own ``trace`` entry."""

    def supervisor_entry(state: MultiAgentResearchState) -> MultiAgentResearchState:
        update: MultiAgentResearchState = {  # type: ignore[typeddict-item]
            "trace": [f"Supervisor: coordinating research for question={state['question']!r}"],
        }
        return update

    return supervisor_entry


def make_finalizer_node():
    """Node: Supervisor (exit / Finalizer). Deterministic -- no LLM call.
    Owns ONLY ``final_answer``."""

    def finalizer(state: MultiAgentResearchState) -> MultiAgentResearchState:
        review = state["review"]
        header = state["synthesis"].strip()
        if review["approved"]:
            note = f"Review: approved after {state['iteration']} iteration(s)."
        else:
            note = (
                f"Review: NOT approved after {state['iteration']} iteration(s) "
                f"(max_iterations={state['max_iterations']}); delivering best-effort answer. "
                f"Outstanding feedback: {review['feedback']}"
            )

        update: MultiAgentResearchState = {  # type: ignore[typeddict-item]
            "final_answer": f"{header}\n\n[{note}]",
            "trace": [f"Supervisor: finalized answer (approved={review['approved']!r})"],
        }
        return update

    return finalizer


def build_graph(
    planner_llm_call: TextLLMCall,
    worker_llm_call: TextLLMCall,
    synthesizer_llm_call: TextLLMCall,
    reviewer_llm_call: TextLLMCall,
    checkpointer: Any = None,
):
    """Assemble the StateGraph. Pass a checkpointer to enable persistence
    (traceable, recoverable state history); pass ``None`` to compile
    without one."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(MultiAgentResearchState)
    builder.add_node("supervisor_entry", make_supervisor_entry_node())
    builder.add_node("planner", make_planner_node(planner_llm_call))
    builder.add_node("research_worker", make_worker_node(worker_llm_call))
    builder.add_node("synthesizer", make_synthesizer_node(synthesizer_llm_call))
    builder.add_node("reviewer", make_reviewer_node(reviewer_llm_call))
    builder.add_node("finalizer", make_finalizer_node())

    builder.add_edge(START, "supervisor_entry")
    builder.add_edge("supervisor_entry", "planner")
    builder.add_conditional_edges("planner", fan_out_to_workers, ["research_worker"])
    builder.add_edge("research_worker", "synthesizer")
    builder.add_edge("synthesizer", "reviewer")
    builder.add_conditional_edges("reviewer", route_after_review, ["planner", "finalizer"])
    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)


def print_mermaid_diagram(graph) -> None:
    print(graph.get_graph().draw_mermaid())


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentResearchOutcome:
    question: str
    tasks: list[dict]
    worker_results: list[WorkerResult]
    synthesis: str
    review: ReviewVerdict
    final_answer: str
    iteration: int
    trace: list[str] = field(default_factory=list)
    thread_id: str = ""
    # Reliability requirement: "Agent loop 设置 max_iterations，超过后进入
    # controlled failure." Populated (non-None) only when the Planner<->
    # Reviewer loop exhausted its iteration budget without the Reviewer
    # approving -- a deliberate, structured stop, distinct from the
    # best-effort ``final_answer`` this graph still returns in that case.
    # ``None`` means the run reached a normal PASS (or failed for an
    # unrelated reason before ever reaching this check).
    controlled_failure: ControlledFailure | None = None


class MultiAgentResearchTimeoutError(TimeoutError):
    """Raised when a whole run exceeds ``run_timeout_seconds`` (Configuration
    & Secrets / Reliability requirement: "Timeout 必须可配置")."""


def _run_with_timeout(func, timeout_seconds: float | None):
    """Run ``func()`` bounded by ``timeout_seconds``.

    Delegates to the single canonical implementation in
    ``src/reliability/timeout.py`` and re-raises its
    ``reliability.timeout.ToolTimeoutError`` as this module's own
    ``MultiAgentResearchTimeoutError`` so existing callers/tests keep seeing
    the same public exception type as before this module started sharing
    the timeout primitive with the rest of the codebase.
    """
    try:
        return run_with_timeout(func, timeout_seconds, tool_name="multi_agent_research_run")
    except ToolTimeoutError as exc:
        raise MultiAgentResearchTimeoutError(
            f"Multi-agent research run exceeded the overall {timeout_seconds}s budget."
        ) from exc


def run_multi_agent_research(
    question: str,
    planner_llm_call: TextLLMCall,
    worker_llm_call: TextLLMCall,
    synthesizer_llm_call: TextLLMCall,
    reviewer_llm_call: TextLLMCall,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    per_worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    checkpointer: Any = None,
    thread_id: str | None = None,
    run_timeout_seconds: float | None = None,
) -> MultiAgentResearchOutcome:
    """Run the full graph once (potentially looping Planner<->Reviewer up to
    ``max_iterations`` times internally) and return a structured outcome.

    ``run_timeout_seconds`` (typically sourced from
    ``config.Settings.limits.run_timeout_seconds``) bounds the *entire* run
    -- independent of ``per_worker_timeout_seconds``, which only bounds each
    individual worker call. Pass ``None`` to disable the overall deadline.
    Raises ``MultiAgentResearchTimeoutError`` if the budget is exceeded.
    """
    graph = build_graph(planner_llm_call, worker_llm_call, synthesizer_llm_call, reviewer_llm_call, checkpointer=checkpointer)
    state = initial_state(
        question,
        max_workers=max_workers,
        max_iterations=max_iterations,
        per_worker_timeout_seconds=per_worker_timeout_seconds,
    )
    thread_id = thread_id or uuid.uuid4().hex
    config = {
        "configurable": {"thread_id": thread_id},
        # Generous recursion budget: each iteration is
        # supervisor/planner/N workers/synthesizer/reviewer supersteps;
        # max_iterations rounds of that, plus the final finalizer step.
        "recursion_limit": max_iterations * (max_workers + 6) + 10,
    }

    def invoke():
        return graph.invoke(state, config)

    result = _run_with_timeout(invoke, run_timeout_seconds)

    controlled_failure: ControlledFailure | None = None
    if not result["review"]["approved"] and result["iteration"] >= max_iterations:
        # Requirement: exceeding max_iterations must produce a deliberate,
        # structured controlled failure -- not silently indistinguishable
        # from a normal PASS. The graph itself still returns a best-effort
        # ``final_answer`` (unchanged, backward-compatible contract); this
        # is the additional structured signal callers should check/log/trace.
        controlled_failure = ControlledFailure(
            reason="review not approved within max_iterations",
            iterations_used=result["iteration"],
            max_iterations=max_iterations,
            last_feedback=result["review"]["feedback"],
        )

    return MultiAgentResearchOutcome(
        question=result["question"],
        tasks=result["tasks"],
        worker_results=result["worker_results"],
        synthesis=result["synthesis"],
        review=result["review"],
        final_answer=result["final_answer"],
        iteration=result["iteration"],
        trace=result["trace"],
        thread_id=thread_id,
        controlled_failure=controlled_failure,
    )
