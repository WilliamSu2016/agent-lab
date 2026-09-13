"""Graph assembly for the Durable Execution experiment.

    START
      |
      v
    research_a   -- side-effecting research task "A"
      |
      v
    research_b   -- side-effecting research task "B"
      |
      v
    research_c   -- side-effecting research task "C"
      |
      v
    finalize     -- deterministic, combines results into final_report
      |
      v
    END

Each ``research_<x>`` node is intentionally a *separate graph superstep*
(a straight chain of edges, not one node looping over a task list): with a
checkpointer attached, LangGraph persists a checkpoint after every
superstep, so "Research A succeeded" and "Research B succeeded" are each
individually durable *before* "Research C" ever starts -- that is what lets
recovery skip re-running A and B after a crash during C (see
``recovery.py``).

Every research node performs one *side-effecting* call (``side_effects[id]``,
injected by the caller) wrapped in ``reliability.idempotency.idempotent``
so that even if the *node itself* has to be replayed from scratch after a
crash (LangGraph only checkpoints whole completed supersteps, never partial
node execution), the underlying side effect is never executed twice for the
same task_id -- this is the "不重复执行已经成功且具有副作用的操作" requirement,
composed with checkpoint-level durability rather than duplicating that
logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from src.durable.state import TASK_IDS, DurableResearchState, initial_state
from src.reliability.idempotency import IdempotencyStore, idempotent

if TYPE_CHECKING:
    from src.durable.recovery import CrashInjectorProtocol

GRAPH_NAME = "Durable Execution Research Pipeline"

SideEffect = Callable[..., Any]


def _default_side_effect(*, task_id: str, content: str) -> dict[str, str]:
    """Placeholder "real" side effect (e.g. calling an external research /
    publish API). Reference/demo implementation only -- tests inject their
    own call-counting stand-ins to verify de-duplication."""
    return {"task_id": task_id, "findings": content}


def make_research_node(
    task_id: str,
    idempotency_store: IdempotencyStore,
    side_effect: SideEffect = _default_side_effect,
    crash_injector: Optional["CrashInjectorProtocol"] = None,
):
    """Node factory for one research task. Owns ONLY ``results[task_id]``
    (plus its own ``trace`` entry).

    Idempotency key: the task_id itself -- stable, caller-known, never
    generated inside the side effect (exactly the pattern documented in
    ``src/reliability/idempotency.py``). Fingerprint: the question the task
    was researching, so a *conflicting* replay (same task_id, different
    question -- which should never legitimately happen for one thread_id)
    is rejected loudly instead of silently returning a stale result.
    """
    idempotent_call = idempotent(
        side_effect,
        store=idempotency_store,
        key_fn=lambda **kwargs: kwargs["task_id"],
        fingerprint_fn=lambda **kwargs: kwargs["content"],
    )

    def research_node(state: DurableResearchState) -> DurableResearchState:
        if crash_injector is not None:
            crash_injector.before(task_id)

        content = f"researching aspect {task_id!r} of question={state['question']!r}"
        result = idempotent_call(task_id=task_id, content=content)

        if crash_injector is not None:
            crash_injector.after(task_id)

        update: DurableResearchState = {  # type: ignore[typeddict-item]
            "results": {task_id: result},
            "trace": [f"research_{task_id.lower()}: completed (findings recorded)"],
        }
        return update

    return research_node


def make_finalize_node():
    """Node: Finalize. Deterministic -- combines every task's results into
    one report. Owns ONLY ``final_report``."""

    def finalize(state: DurableResearchState) -> DurableResearchState:
        lines = [f"Question: {state['question']}", ""]
        for task_id in TASK_IDS:
            result = state["results"].get(task_id)
            lines.append(f"- Task {task_id}: {result['findings'] if result else '<missing>'}")
        update: DurableResearchState = {  # type: ignore[typeddict-item]
            "final_report": "\n".join(lines),
            "trace": ["finalize: report assembled from all task results"],
        }
        return update

    return finalize


def build_durable_graph(
    idempotency_store: IdempotencyStore,
    side_effects: dict[str, SideEffect] | None = None,
    crash_injector: Optional["CrashInjectorProtocol"] = None,
    checkpointer: Any = None,
):
    """Assemble the StateGraph. Pass a checkpointer (e.g.
    ``checkpointer.sqlite_checkpointer(...)``) to enable durable execution;
    pass ``None`` to compile without persistence (no crash recovery is
    possible in that mode -- every step is lost the instant the process
    exits, same caveat as ``InMemorySaver``)."""
    from langgraph.graph import END, START, StateGraph

    side_effects = side_effects or {task_id: _default_side_effect for task_id in TASK_IDS}

    builder = StateGraph(DurableResearchState)
    for task_id in TASK_IDS:
        builder.add_node(
            f"research_{task_id.lower()}",
            make_research_node(task_id, idempotency_store, side_effects[task_id], crash_injector),
        )
    builder.add_node("finalize", make_finalize_node())

    builder.add_edge(START, "research_a")
    builder.add_edge("research_a", "research_b")
    builder.add_edge("research_b", "research_c")
    builder.add_edge("research_c", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_durable_graph", "initial_state", "make_research_node", "make_finalize_node", "GRAPH_NAME"]
