"""The first, minimal LangGraph graph.

This module intentionally contains NO LLM call, NO Agent, and NO Tool.
It only demonstrates the core LangGraph building blocks:

    START -> greet -> format -> END

State:
    {
        "name": str,
        "message": str,
    }

- ``greet``  fills ``message`` with a plain greeting built from ``name``.
- ``format`` transforms ``message`` into the final output format.

See ``docs/01-MINIMAL-GRAPH.md`` for a detailed explanation of every
LangGraph concept used here (Node, State, Edge, START, END, compile, invoke).
"""

from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class GraphState(TypedDict):
    """The single shared State that flows through every node in the graph."""

    name: str
    message: str


def greet(state: GraphState) -> GraphState:
    """Node 1: build a greeting message from ``name``."""
    return {"message": f"Hello, {state['name']}!"}


def format_message(state: GraphState) -> GraphState:
    """Node 2: convert ``message`` into the final output format."""
    return {"message": f">>> {state['message']} <<<"}


def build_graph():
    """Build and compile the ``START -> greet -> format -> END`` graph."""
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("greet", greet)
    graph_builder.add_node("format", format_message)

    graph_builder.add_edge(START, "greet")
    graph_builder.add_edge("greet", "format")
    graph_builder.add_edge("format", END)

    return graph_builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"name": "LangGraph", "message": ""})
    print(result)


if __name__ == "__main__":
    main()
