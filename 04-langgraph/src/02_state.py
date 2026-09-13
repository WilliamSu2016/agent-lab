"""LangGraph experiment #2: understanding State.

Still NO LLM, NO Agent, NO Tool, NO database, NO global variables.

This module turns the minimal graph from experiment #1 into a slightly
richer pipeline that shows how State moves between Nodes, and how each
Node should only touch the fields it owns:

    START -> research -> analysis -> finalize -> END

State:
    {
        "question": str,
        "research_notes": list[str],
        "analysis": str,
        "final_answer": str,
    }

- ``research``  owns ``research_notes`` (derived from ``question``).
- ``analysis``  owns ``analysis`` (derived from ``research_notes``).
- ``finalize``  owns ``final_answer`` (derived from ``analysis``).

See ``docs/02-STATE.md`` for the conceptual explanation (why State is the
core of LangGraph, why Nodes must not use global variables, State vs plain
function arguments, what happens when two Nodes write the same field, and
what Reducers are for).
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """The single shared State that flows through research -> analysis -> finalize."""

    question: str
    research_notes: list[str]
    analysis: str
    final_answer: str


def _print_state(label: str, state: ResearchState) -> None:
    """Pure logging helper: takes State in, prints it, mutates nothing."""
    print(f"[{label}] {state}")


def research(state: ResearchState) -> ResearchState:
    """Node 1: owns ``research_notes`` only.

    Reads ``question`` (owned by no one / set by the caller), derives a
    couple of research notes from it, and returns ONLY the field this
    Node is responsible for.
    """
    _print_state("research: before", state)

    question = state["question"]
    notes = [
        f"Note 1: the question is '{question}'.",
        f"Note 2: the question has {len(question.split())} word(s).",
    ]
    update: ResearchState = {"research_notes": notes}  # type: ignore[typeddict-item]

    after_preview: ResearchState = {**state, **update}
    _print_state("research: after", after_preview)
    return update


def analysis(state: ResearchState) -> ResearchState:
    """Node 2: owns ``analysis`` only.

    Reads ``research_notes`` (owned by ``research``) and produces the
    ``analysis`` field. It never touches ``research_notes`` or
    ``final_answer``.
    """
    _print_state("analysis: before", state)

    notes = state["research_notes"]
    summary = f"Analyzed {len(notes)} research note(s): " + " | ".join(notes)
    update: ResearchState = {"analysis": summary}  # type: ignore[typeddict-item]

    after_preview: ResearchState = {**state, **update}
    _print_state("analysis: after", after_preview)
    return update


def finalize(state: ResearchState) -> ResearchState:
    """Node 3: owns ``final_answer`` only.

    Reads ``analysis`` (owned by ``analysis``) and produces the final
    output field.
    """
    _print_state("finalize: before", state)

    update: ResearchState = {"final_answer": f"Final answer -> {state['analysis']}"}  # type: ignore[typeddict-item]

    after_preview: ResearchState = {**state, **update}
    _print_state("finalize: after", after_preview)
    return update


def build_graph():
    """Build and compile the ``START -> research -> analysis -> finalize -> END`` graph.

    Note: the Node is registered under the id ``"analyze"`` rather than
    ``"analysis"``, because LangGraph forbids a Node id that collides with an
    existing State key -- and ``"analysis"`` is already a field of
    ``ResearchState``. The Node still runs the ``analysis`` function; only
    its *registered name* differs from the State field it writes to.
    """
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ResearchState)

    graph_builder.add_node("research", research)
    graph_builder.add_node("analyze", analysis)
    graph_builder.add_node("finalize", finalize)

    graph_builder.add_edge(START, "research")
    graph_builder.add_edge("research", "analyze")
    graph_builder.add_edge("analyze", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()


def main() -> None:
    graph = build_graph()
    initial_state: ResearchState = {
        "question": "What is LangGraph State?",
        "research_notes": [],
        "analysis": "",
        "final_answer": "",
    }
    result = graph.invoke(initial_state)
    print("=" * 60)
    print("FINAL STATE:", result)


if __name__ == "__main__":
    main()
