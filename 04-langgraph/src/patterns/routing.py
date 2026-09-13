"""Routing, expressed as a LangGraph ``StateGraph`` with a Conditional Edge.

Anthropic pattern: a single Router step classifies the input, then program
code dispatches to exactly ONE specialized downstream step. The other
specialists never run for that input.

Graph:

    START -> router -> route_after_router -- "technical" --> technical --> END
                                          -- "business"  --> business  --> END
                                          -- "general"   --> general   --> END
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

Category = Literal["technical", "business", "general"]

TextLLMCall = Callable[[str, str], str]
SearchWebCall = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class RoutingState(TypedDict):
    question: str
    category: Category | None
    final_answer: str


def initial_state(question: str) -> RoutingState:
    return {"question": question, "category": None, "final_answer": ""}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTER_INSTRUCTIONS = """You are the Router. Classify the user's question into EXACTLY one
of: technical, business, general. Respond with only that single lowercase word, nothing
else.
- technical: technology/framework/SDK/architecture/implementation questions.
- business: market/pricing/competitor/adoption/strategy questions.
- general: broad or introductory knowledge questions that are neither of the above."""

SPECIALIST_INSTRUCTIONS: dict[Category, str] = {
    "technical": (
        "You are the Technical Researcher. Answer focusing on technical implementation, "
        "architecture, APIs, and ecosystem details."
    ),
    "business": (
        "You are the Business Researcher. Answer focusing on market, pricing, competitors, "
        "and adoption/business strategy."
    ),
    "general": (
        "You are the General Researcher. Answer with a clear, accessible explanation for "
        "a general audience."
    ),
}


def _parse_category(raw_text: str) -> Category:
    normalized = raw_text.strip().lower()
    for category in ("technical", "business", "general"):
        if category in normalized:
            return category  # type: ignore[return-value]
    raise ValueError(f"Router returned an unrecognized category: {raw_text!r}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_router(llm_call: TextLLMCall):
    """The one and only classification step. Owns ``category`` only."""

    def router(state: RoutingState) -> RoutingState:
        category = _parse_category(llm_call(ROUTER_INSTRUCTIONS, state["question"]))
        return {"category": category}  # type: ignore[typeddict-item]

    return router


def make_specialist(category: Category, llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    """One Node per fixed route. Owns ``final_answer`` only. Each Specialist may
    use ``search_web`` if it decides evidence is needed; the Node itself does
    not force a search (this mirrors the original Agent's own tool, kept
    optional per specialist)."""

    def specialist(state: RoutingState) -> RoutingState:
        final_answer = llm_call(SPECIALIST_INSTRUCTIONS[category], state["question"])
        return {"final_answer": final_answer}  # type: ignore[typeddict-item]

    return specialist


# ---------------------------------------------------------------------------
# Routing function (NOT a Node): reads the Router's classification and picks
# exactly one of the three fixed routes.
# ---------------------------------------------------------------------------


def route_after_router(state: RoutingState) -> Category:
    assert state["category"] is not None, "router must set a category before routing"
    return state["category"]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(RoutingState)
    graph_builder.add_node("router", make_router(llm_call))
    graph_builder.add_node("technical", make_specialist("technical", llm_call, search_web_call))
    graph_builder.add_node("business", make_specialist("business", llm_call, search_web_call))
    graph_builder.add_node("general", make_specialist("general", llm_call, search_web_call))

    graph_builder.add_edge(START, "router")
    graph_builder.add_conditional_edges(
        "router",
        route_after_router,
        {"technical": "technical", "business": "business", "general": "general"},
    )
    graph_builder.add_edge("technical", END)
    graph_builder.add_edge("business", END)
    graph_builder.add_edge("general", END)

    return graph_builder.compile()


def get_mermaid(graph) -> str:
    return graph.get_graph().draw_mermaid()
