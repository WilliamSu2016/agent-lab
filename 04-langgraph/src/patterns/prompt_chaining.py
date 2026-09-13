"""Prompt Chaining, expressed as a LangGraph ``StateGraph``.

Anthropic pattern: a task is decomposed into a **fixed sequence of steps**,
each step's LLM call consuming only the previous step's output. No step
decides whether the next step runs -- the order is hard-coded.

Graph:

    START -> planner -> researcher -> analyst -> writer -> END

Every edge is unconditional: no Conditional Edge is needed because the step
order never varies.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

TextLLMCall = Callable[[str, str], str]
SearchWebCall = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ResearchNote(TypedDict):
    dimension: str
    findings: str
    sources: list[str]


class PromptChainState(TypedDict):
    question: str
    dimensions: list[str]
    research_notes: list[ResearchNote]
    analysis: str
    final_answer: str


def initial_state(question: str) -> PromptChainState:
    return {
        "question": question,
        "dimensions": [],
        "research_notes": [],
        "analysis": "",
        "final_answer": "",
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTIONS = """You are the Planner step of a fixed research pipeline.
Given the user's question, return a JSON array of 3 to 6 short, distinct research
dimensions that must be investigated before comparing/answering. Return ONLY the
JSON array, nothing else."""

RESEARCHER_SYNTHESIS_INSTRUCTIONS = """You are the Researcher step. You are given one
research dimension and raw web search results for it. Write a short, factual findings
summary for this dimension based ONLY on the given search results."""

ANALYST_INSTRUCTIONS = """You are the Analyst step. You are given the original question
and research notes for each dimension. Write a dimension-by-dimension analysis
comparing the evidence. Do not write a final recommendation -- that is the Writer's job."""

WRITER_INSTRUCTIONS = """You are the Writer step. You are given the original question and
the Analyst's dimension-by-dimension analysis. Write the final answer: a clear
comparison plus one explicit, justified recommendation."""


def _parse_dimensions(raw_text: str) -> list[str]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return [line.strip("- ").strip() for line in raw_text.splitlines() if line.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Planner must return a JSON array of strings.")
    return parsed


def _format_research_notes(notes: list[ResearchNote]) -> str:
    sections = []
    for note in notes:
        section = f"### {note['dimension']}\n{note['findings']}"
        if note["sources"]:
            section += "\nSources: " + ", ".join(note["sources"])
        sections.append(section)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Nodes -- each a plain function of (state) -> partial state update, with the
# LLM/tool calls injected via closure (never a global variable).
# ---------------------------------------------------------------------------


def make_planner(llm_call: TextLLMCall):
    def planner(state: PromptChainState) -> PromptChainState:
        response_text = llm_call(PLANNER_INSTRUCTIONS, state["question"])
        dimensions = _parse_dimensions(response_text)
        return {"dimensions": dimensions}  # type: ignore[typeddict-item]

    return planner


def make_researcher(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    def researcher(state: PromptChainState) -> PromptChainState:
        notes: list[ResearchNote] = []
        for dimension in state["dimensions"]:
            search_result = search_web_call(dimension)
            results = search_result.get("results", [])
            findings = llm_call(
                RESEARCHER_SYNTHESIS_INSTRUCTIONS,
                f"Dimension: {dimension}\nSearch results: {json.dumps(results)}",
            )
            sources = [item["url"] for item in results if "url" in item]
            notes.append({"dimension": dimension, "findings": findings, "sources": sources})
        return {"research_notes": notes}  # type: ignore[typeddict-item]

    return researcher


def make_analyst(llm_call: TextLLMCall):
    def analyst(state: PromptChainState) -> PromptChainState:
        user_prompt = (
            f"Original question: {state['question']}\n\n"
            f"Research notes:\n{_format_research_notes(state['research_notes'])}"
        )
        return {"analysis": llm_call(ANALYST_INSTRUCTIONS, user_prompt)}  # type: ignore[typeddict-item]

    return analyst


def make_writer(llm_call: TextLLMCall):
    def writer(state: PromptChainState) -> PromptChainState:
        user_prompt = f"Original question: {state['question']}\n\nAnalysis:\n{state['analysis']}"
        return {"final_answer": llm_call(WRITER_INSTRUCTIONS, user_prompt)}  # type: ignore[typeddict-item]

    return writer


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    """Build and compile ``START -> planner -> researcher -> analyst -> writer -> END``."""
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(PromptChainState)
    graph_builder.add_node("planner", make_planner(llm_call))
    graph_builder.add_node("researcher", make_researcher(llm_call, search_web_call))
    graph_builder.add_node("analyst", make_analyst(llm_call))
    graph_builder.add_node("writer", make_writer(llm_call))

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "researcher")
    graph_builder.add_edge("researcher", "analyst")
    graph_builder.add_edge("analyst", "writer")
    graph_builder.add_edge("writer", END)

    return graph_builder.compile()


def get_mermaid(graph) -> str:
    """Return this compiled graph's Mermaid diagram source."""
    return graph.get_graph().draw_mermaid()
