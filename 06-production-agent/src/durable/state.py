"""Shared state schema for the Durable Execution experiment.

Deliberately linear (Research A -> Research B -> Research C -> Finalize,
see ``graph.py``) rather than fanned out like
``src/multi_agent_research``'s Planner/Worker graph: the point of this
experiment is to make "exactly which steps already committed a checkpoint"
unambiguous and trivial to assert on in tests, not to explore parallelism
(that is experiment 4's job).

Ownership convention (same as ``src/multi_agent_research/state.py``): every
node returns a partial update containing only the key(s) it owns.

    node             owns (writes)
    ----             -------------
    research_a/b/c   results[<task_id>] (merge-reducer: each task
                      contributes its own key, never overwrites another
                      task's key)
    finalize         final_report

``trace`` is the one deliberately shared, append-only field (mirrors
``docs/14-MULTI-AGENT-RESEARCH.md`` question 8's rationale).
"""

from __future__ import annotations

import operator
from typing import Any

from typing_extensions import Annotated, TypedDict

TASK_IDS: tuple[str, ...] = ("A", "B", "C")


def merge_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer for ``results``: each research node contributes exactly one
    key (its own ``task_id``) and never touches another task's key, so a
    plain non-destructive merge is sufficient and correct -- unlike
    ``operator.add`` (which does not apply to dicts), this preserves every
    previously-recorded task result across every subsequent graph step,
    including across a crash/resume replay of an earlier-completed node
    (which never happens here since completed nodes are never re-run, but
    the reducer is written to be safe even if that ever changed)."""
    merged = dict(left)
    merged.update(right)
    return merged


class DurableResearchState(TypedDict):
    """The one channel every node in ``graph.py`` communicates through."""

    question: str

    # One entry per completed research task, keyed by task_id ("A"/"B"/"C").
    results: Annotated[dict[str, Any], merge_results]

    final_report: str

    # Append-only audit log -- also readable independently of the
    # checkpointer via each checkpoint's channel values (see
    # ``recovery.get_execution_history``).
    trace: Annotated[list[str], operator.add]


def initial_state(question: str) -> DurableResearchState:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    return {
        "question": question,
        "results": {},
        "final_report": "",
        "trace": [f"start: question={question!r} tasks={list(TASK_IDS)!r}"],
    }
