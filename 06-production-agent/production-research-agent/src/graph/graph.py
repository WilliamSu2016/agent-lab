"""Graph assembly for the Production Research Agent.

    START
      |
      v
    supervisor_entry   (Layer 1 Input Guardrail; deterministic bookkeeping)
      |
      v (conditional: blocked -> finalizer, else -> planner)
    planner              (dynamic Research Tasks; capped at max_workers)
      |
      v (conditional edge -> list[Send], fan-out)
    researcher x N        (parallel Specialist Research Agents, each making
      |                    a real LOW-risk tool call through the 3-layer
      |                    guardrail pipeline before its LLM call)
      v (LangGraph fan-in: all dispatched instances must finish)
    synthesizer
      |
      v
    reviewer
      |
      v (conditional edge)
      +-- FAIL (not approved) & iteration < max_iterations --> back to planner
      |
      v -- PASS, blocked, or iteration budget exhausted
    finalizer             (Layer 2 Output Validation; produces final_answer)
      |
      v
    END

Every node is wrapped in an ``src.observability.tracing.Tracer`` span, so
a completed run's execution tree directly answers every Observability
requirement (which agents ran, which tools were called, how long each
took, token/cost usage per agent). The graph is compiled with a
production-capable ``BaseCheckpointSaver`` (never ``InMemorySaver`` -- see
``checkpointer.py``), so every completed superstep is durable and a
crashed run can be resumed via ``recovery.resume`` without re-running
already-completed nodes.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from src.agents.llm import TextLLMCall
from src.agents.reviewer import make_reviewer_node
from src.agents.researcher import make_researcher_node
from src.agents.supervisor import (
    controlled_failure_for,
    fan_out_to_workers,
    make_finalizer_node,
    make_notify_node,
    make_planner_node,
    make_supervisor_entry_node,
    route_after_entry,
    route_after_review,
)
from src.agents.synthesizer import make_synthesizer_node
from src.cost.budget import BudgetExceededError, BudgetTracker
from src.cost.policy import ExecutionMode, get_policy
from src.graph.state import (
    IdentityDict,
    ProductionResearchState,
    ReviewVerdict,
    WorkerResult,
    empty_review_verdict,
    initial_state,
)
from src.observability.logging import configure_json_logging
from src.observability.tracing import ExecutionContext, Span, SpanKind, Tracer, bind_execution_context
from src.reliability.errors import ControlledFailure
from src.reliability.timeout import ToolTimeoutError, run_with_timeout
from src.security.authorization import ApprovalStore
from src.security.guardrails import AgentWorkflowGuardrail, ToolGuardrail
from src.security.tool_policy import ToolRegistry, build_default_registry
from src.tools.actions import build_send_email_tool
from src.tools.search import build_search_tool

if TYPE_CHECKING:
    from src.memory.store import SessionMemoryStore

GRAPH_NAME = "Production Research Agent"


def build_graph(
    planner_llm_call: TextLLMCall,
    researcher_llm_call: TextLLMCall,
    synthesizer_llm_call: TextLLMCall,
    reviewer_llm_call: TextLLMCall,
    *,
    checkpointer: Any = None,
    tracer: Optional[Tracer] = None,
    tool_registry: Optional[ToolRegistry] = None,
    approval_store: Optional[ApprovalStore] = None,
    search_tool_fn=None,
    send_email_tool_fn=None,
    logger: Optional[logging.Logger] = None,
):
    """Assemble the StateGraph. Pass a production checkpointer (see
    ``checkpointer.sqlite_checkpointer``) to enable durable execution."""
    from langgraph.graph import END, START, StateGraph

    tracer = tracer or Tracer()
    tool_registry = tool_registry or build_default_registry()
    approval_store = approval_store or ApprovalStore()
    search_tool_fn = search_tool_fn or build_search_tool()
    send_email_tool_fn = send_email_tool_fn or build_send_email_tool()
    logger = logger or configure_json_logging(logger_name="agent.graph")

    workflow_guardrail = AgentWorkflowGuardrail(tool_registry, approval_store)
    tool_guardrail = ToolGuardrail(tool_registry)

    builder = StateGraph(ProductionResearchState)
    builder.add_node("supervisor_entry", make_supervisor_entry_node(tracer=tracer, logger=logger))
    builder.add_node("planner", make_planner_node(planner_llm_call, tracer=tracer))
    builder.add_node(
        "researcher",
        make_researcher_node(
            researcher_llm_call,
            search_tool_fn=search_tool_fn,
            workflow_guardrail=workflow_guardrail,
            tool_guardrail=tool_guardrail,
            tracer=tracer,
        ),
    )
    builder.add_node("synthesizer", make_synthesizer_node(synthesizer_llm_call, tracer=tracer))
    builder.add_node("reviewer", make_reviewer_node(reviewer_llm_call, tracer=tracer))
    builder.add_node("finalizer", make_finalizer_node(workflow_guardrail, tracer=tracer, logger=logger))
    builder.add_node("notify_action", make_notify_node(send_email_tool_fn, workflow_guardrail, tool_guardrail, tracer=tracer))

    builder.add_edge(START, "supervisor_entry")
    builder.add_conditional_edges("supervisor_entry", route_after_entry, ["planner", "finalizer"])
    builder.add_conditional_edges("planner", fan_out_to_workers, ["researcher"])
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "reviewer")
    builder.add_conditional_edges("reviewer", route_after_review, ["planner", "finalizer"])
    builder.add_edge("finalizer", "notify_action")
    builder.add_edge("notify_action", END)

    return builder.compile(checkpointer=checkpointer)


def print_mermaid_diagram(graph) -> None:
    print(graph.get_graph().draw_mermaid())


@dataclass
class ProductionResearchOutcome:
    question: str
    tasks: list[dict]
    worker_results: list[WorkerResult]
    synthesis: str
    review: ReviewVerdict
    final_answer: str
    iteration: int
    blocked: bool
    tokens_used: int
    cost_usd: float
    trace: list[str] = field(default_factory=list)
    thread_id: str = ""
    mode: str = "balanced"
    controlled_failure: ControlledFailure | None = None
    root_span: Optional[Span] = None


class ProductionResearchTimeoutError(TimeoutError):
    pass


def _run_with_timeout(func, timeout_seconds: float | None):
    try:
        return run_with_timeout(func, timeout_seconds, tool_name="production_research_run")
    except ToolTimeoutError as exc:
        raise ProductionResearchTimeoutError(f"Run exceeded the overall {timeout_seconds}s budget.") from exc


def run_production_research(
    question: str,
    identity: IdentityDict,
    planner_llm_call: TextLLMCall,
    researcher_llm_call: TextLLMCall,
    synthesizer_llm_call: TextLLMCall,
    reviewer_llm_call: TextLLMCall,
    *,
    mode: str = "balanced",
    checkpointer: Any = None,
    tracer: Optional[Tracer] = None,
    tool_registry: Optional[ToolRegistry] = None,
    approval_store: Optional[ApprovalStore] = None,
    search_tool_fn=None,
    thread_id: str | None = None,
    execution_context: Optional[ExecutionContext] = None,
    memory_store: Optional["SessionMemoryStore"] = None,
) -> ProductionResearchOutcome:
    """Run the full graph once, bounded by the chosen mode's
    ``src.cost.policy.ExecutionPolicy`` budget. Callers wanting automatic
    Quality -> Balanced -> Fast degradation should use
    ``src.cost.policy.run_with_degradation`` around this function (see
    ``docs/07-COST-LATENCY.md``'s pattern, reused unchanged) -- this
    function itself only enforces the *one* mode it is given.

    ``memory_store`` (optional): if supplied and this exact
    ``(tenant_id, question)`` already has a completed answer cached (see
    ``src.memory.store.SessionMemoryStore``), the whole graph is skipped
    entirely and the cached answer is returned immediately at zero
    additional cost/latency -- a cheap, safe cost/latency win for an exact
    repeat request, never a substitute for real retrieval/RAG (see
    ``docs/architecture.md``'s Memory section).
    """
    tracer = tracer or Tracer()
    policy = get_policy(ExecutionMode(mode))
    budget_tracker = BudgetTracker(policy.budget)

    if memory_store is not None:
        cached = memory_store.recall(identity["tenant_id"], question)
        if cached is not None:
            return ProductionResearchOutcome(
                question=question,
                tasks=[],
                worker_results=[],
                synthesis="",
                review=empty_review_verdict(),
                final_answer=cached.answer,
                iteration=0,
                blocked=False,
                tokens_used=0,
                cost_usd=0.0,
                trace=[f"Supervisor: served cached answer from memory (original thread_id={cached.thread_id!r})"],
                thread_id=cached.thread_id,
                mode=mode,
                controlled_failure=None,
                root_span=None,
            )

    graph = build_graph(
        planner_llm_call,
        researcher_llm_call,
        synthesizer_llm_call,
        reviewer_llm_call,
        checkpointer=checkpointer,
        tracer=tracer,
        tool_registry=tool_registry,
        approval_store=approval_store,
        search_tool_fn=search_tool_fn,
    )
    thread_id = thread_id or uuid.uuid4().hex
    context = execution_context or ExecutionContext(
        request_id=uuid.uuid4().hex,
        trace_id=uuid.uuid4().hex,
        user_id=identity["user_id"],
        session_id=thread_id,
        agent_version="dev",
        environment="development",
    )
    state = initial_state(
        question,
        identity,
        context=context.to_dict(),
        mode=mode,
        max_workers=policy.max_workers,
        max_iterations=policy.max_agent_iterations,
        per_worker_timeout_seconds=min(30.0, policy.budget.timeout_seconds / max(policy.max_agent_iterations, 1)),
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": policy.max_agent_iterations * (policy.max_workers + 6) + 10,
    }

    def invoke():
        with bind_execution_context(context):
            with tracer.span(SpanKind.WORKFLOW, "production_research_run", context=context) as root:
                result = graph.invoke(state, config)
                budget_tracker.usage.tokens_used = result["tokens_used"]
                budget_tracker.usage.cost_usd = result["cost_usd"]
                budget_tracker.usage.agent_iterations = result["iteration"]
                budget_tracker.usage.workers_used = len(result["tasks"])
                budget_tracker.check_all()
                root.attributes["final_status"] = "blocked" if result["blocked"] else "completed"
                return result, root

    result, root_span = _run_with_timeout(invoke, policy.budget.timeout_seconds)

    controlled_failure = None if result["blocked"] else controlled_failure_for(result, policy.max_agent_iterations)

    if memory_store is not None and not result["blocked"] and result["final_answer"]:
        memory_store.remember(identity["tenant_id"], question, result["final_answer"], thread_id)

    return ProductionResearchOutcome(
        question=result["question"],
        tasks=result["tasks"],
        worker_results=result["worker_results"],
        synthesis=result["synthesis"],
        review=result["review"],
        final_answer=result["final_answer"],
        iteration=result["iteration"],
        blocked=result["blocked"],
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        trace=result["trace"],
        thread_id=thread_id,
        mode=mode,
        controlled_failure=controlled_failure,
        root_span=root_span,
    )


__all__ = [
    "GRAPH_NAME",
    "build_graph",
    "print_mermaid_diagram",
    "ProductionResearchOutcome",
    "ProductionResearchTimeoutError",
    "run_production_research",
]
