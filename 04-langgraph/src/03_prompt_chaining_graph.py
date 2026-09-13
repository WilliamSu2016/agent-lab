"""Prompt Chaining, rebuilt as a LangGraph ``StateGraph``.

This is a LangGraph reimplementation of ``src/prompt_chaining.py``'s fixed
Planner -> Researcher -> Analyst -> Writer pipeline:

    START -> planner -> researcher -> analyst -> writer -> END

Design constraints for this experiment:

* ``StateGraph`` is used, with one Node per pipeline step and Edges
  connecting them in a fixed, linear order (no Conditional Edge yet).
* All intermediate results flow through a single, explicitly typed
  ``PromptChainState`` -- there is no other channel between steps.
* NO ``openai-agents`` SDK: no ``Agent``, no ``Runner``, no tool-call loop
  decided by an LLM. Each Node calls the LLM directly (see ``LLMCall``
  below) and, where a tool is needed, calls it directly as plain Python --
  the *fixed* pipeline code decides when ``search_web`` runs, not the model.
* NO LangChain ``Agent`` abstraction (no ``AgentExecutor``, no
  ``create_react_agent``, ...). Nodes are plain Python functions.
* NO multi-agent / handoffs: there is exactly one flow of control, moving
  step by step through this module's own Python code.
* The ``search_web`` tool from ``src/tools.py`` is preserved and is still the
  only source of external, factual information -- it is called directly by
  the ``researcher`` Node.
* No Conditional Edge, no Persistence/Checkpoint, no Human-in-the-loop yet
  (those are later experiments).

See ``docs/03-PROMPT-CHAINING-GRAPH.md`` for how each original Planner /
Researcher / Analyst / Writer step maps onto a Node here, and for why
LangGraph is a better fit for this kind of workflow than plain function
calls.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

# ---------------------------------------------------------------------------
# LLM call seam: a Node never talks to the OpenAI/LangChain SDKs directly.
#
# ``LLMCall`` is a plain callable: (system_prompt, user_prompt) -> response
# text. This is intentionally NOT an Agent/Runner/AgentExecutor -- it is the
# smallest possible seam that lets every Node stay a pure, testable function
# of (state, llm_call) while still allowing a real LLM to be plugged in for
# production use (see ``build_openai_llm_call`` below).
# ---------------------------------------------------------------------------

LLMCall = Callable[[str, str], str]
SearchWebCall = Callable[[str], dict[str, Any]]


def build_openai_llm_call(api_key: str, model_name: str, base_url: str) -> LLMCall:
    """Build a real ``LLMCall`` backed by the plain ``openai`` client.

    This talks Chat Completions directly (``client.chat.completions.create``)
    -- no ``agents.Agent``/``Runner`` and no LangChain ``Agent`` abstraction
    are involved, only a single request/response call per invocation.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or ""

    return call


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ResearchNote(TypedDict):
    dimension: str
    findings: str
    sources: list[str]


class PromptChainState(TypedDict):
    """Everything that flows between planner -> researcher -> analyst -> writer."""

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


def _print_state(label: str, state: PromptChainState) -> None:
    print(f"[{label}] {state}")


# ---------------------------------------------------------------------------
# Prompts -- one narrow instruction string per Node, mirroring the narrow
# ``instructions`` each Agent had in src/prompt_chaining.py.
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTIONS = """You are the Planner step of a fixed research pipeline.
Given the user's question, return a JSON array of 3 to 6 short, distinct research
dimensions that must be investigated before comparing/answering (for example:
["Concurrency model", "Ecosystem maturity", "Tooling"]). Return ONLY the JSON array,
nothing else. Do not answer the question yourself and do not invent facts."""

RESEARCHER_SYNTHESIS_INSTRUCTIONS = """You are the Researcher step of a fixed research
pipeline. You are given one research dimension and raw web search results for it.
Write a short, factual findings summary for this dimension based ONLY on the given
search results. Do not compare options or give an opinion -- that happens later."""

ANALYST_INSTRUCTIONS = """You are the Analyst step of a fixed research pipeline. You are
given the original question and research notes for each dimension. Write a dimension-by
-dimension analysis comparing the evidence. Do not call any tool and do not write a final
recommendation -- that is the Writer step's job."""

WRITER_INSTRUCTIONS = """You are the Writer step of a fixed research pipeline. You are
given the original question and the Analyst's dimension-by-dimension analysis. Write the
final answer: a clear comparison plus one explicit, justified recommendation."""


# ---------------------------------------------------------------------------
# Nodes -- each is a plain function of (state) -> partial state update.
#
# ``llm_call``/``search_web_call`` are captured by closure (dependency
# injection), not read from globals: swapping them (e.g. in tests) never
# touches module-level state.
# ---------------------------------------------------------------------------


def _parse_dimensions(raw_text: str) -> list[str]:
    """Parse the Planner's JSON array response into a list of dimension strings."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: one dimension per non-empty line, in case the model
        # didn't return strict JSON.
        return [line.strip("- ").strip() for line in raw_text.splitlines() if line.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("Planner must return a JSON array of strings.")
    return parsed


def make_planner(llm_call: LLMCall):
    """Node factory for the Planner step: question -> dimensions."""

    def planner(state: PromptChainState) -> PromptChainState:
        _print_state("planner: before", state)
        response_text = llm_call(PLANNER_INSTRUCTIONS, state["question"])
        dimensions = _parse_dimensions(response_text)
        update: PromptChainState = {"dimensions": dimensions}  # type: ignore[typeddict-item]
        _print_state("planner: after", {**state, **update})
        return update

    return planner


def make_researcher(llm_call: LLMCall, search_web_call: SearchWebCall = _web_search):
    """Node factory for the Researcher step: dimensions -> research_notes.

    The pipeline code (not an LLM) decides that ``search_web_call`` runs
    exactly once per dimension; the LLM is only used afterwards, to
    summarize the raw search results it is handed.
    """

    def researcher(state: PromptChainState) -> PromptChainState:
        _print_state("researcher: before", state)
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
        update: PromptChainState = {"research_notes": notes}  # type: ignore[typeddict-item]
        _print_state("researcher: after", {**state, **update})
        return update

    return researcher


def _format_research_notes(notes: list[ResearchNote]) -> str:
    sections: list[str] = []
    for note in notes:
        section = f"### {note['dimension']}\n{note['findings']}"
        if note["sources"]:
            section += "\nSources: " + ", ".join(note["sources"])
        sections.append(section)
    return "\n\n".join(sections)


def make_analyst(llm_call: LLMCall):
    """Node factory for the Analyst step: research_notes -> analysis."""

    def analyst(state: PromptChainState) -> PromptChainState:
        _print_state("analyst: before", state)
        user_prompt = (
            f"Original question: {state['question']}\n\n"
            f"Research notes:\n{_format_research_notes(state['research_notes'])}"
        )
        analysis_text = llm_call(ANALYST_INSTRUCTIONS, user_prompt)
        update: PromptChainState = {"analysis": analysis_text}  # type: ignore[typeddict-item]
        _print_state("analyst: after", {**state, **update})
        return update

    return analyst


def make_writer(llm_call: LLMCall):
    """Node factory for the Writer step: analysis -> final_answer."""

    def writer(state: PromptChainState) -> PromptChainState:
        _print_state("writer: before", state)
        user_prompt = f"Original question: {state['question']}\n\nAnalysis:\n{state['analysis']}"
        final_answer = llm_call(WRITER_INSTRUCTIONS, user_prompt)
        update: PromptChainState = {"final_answer": final_answer}  # type: ignore[typeddict-item]
        _print_state("writer: after", {**state, **update})
        return update

    return writer


# ---------------------------------------------------------------------------
# Graph assembly: START -> planner -> researcher -> analyst -> writer -> END
# ---------------------------------------------------------------------------


def build_graph(llm_call: LLMCall, search_web_call: SearchWebCall = _web_search):
    """Build and compile the fixed Prompt Chaining graph.

    Every edge here is unconditional (no Conditional Edge yet): the four
    steps always run in this exact order, exactly once each.
    """
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


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    question = " ".join(sys.argv[1:]).strip() or (
        "Compare Python, TypeScript, and Go for building AI agents, and give a recommendation."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    llm_call = build_openai_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    graph = build_graph(llm_call)

    result = graph.invoke(initial_state(question))

    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()
