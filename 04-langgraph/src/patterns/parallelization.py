"""Parallelization (sectioning), expressed as a LangGraph ``StateGraph``.

Anthropic pattern: an independently divisible task is split into fixed,
non-overlapping subtasks that run **at the same time**, then a Synthesizer
aggregates all of their results.

Graph (fan-out from START, fan-in into synthesizer):

    START --> research_python      --\
    START --> research_typescript   ---> synthesizer --> END
    START --> research_go          --/

LangGraph runs all three ``research_*`` Nodes in the same superstep (they
have no data dependency on each other), and only invokes ``synthesizer``
once ALL of them have completed -- the fan-in "join" is automatic.

Because all three Nodes write to the same ``findings`` State field
*concurrently*, that field MUST use a Reducer (``operator.add``) instead of
the default "last write wins" merge -- otherwise two of the three findings
would silently overwrite each other (see ``docs/02-STATE.md`` Q4/Q5, and
``docs/06-PATTERNS-IN-LANGGRAPH.md`` for why this pattern is the one that
actually needs a Reducer).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

Language = Literal["Python", "TypeScript", "Go"]
LANGUAGES: tuple[Language, ...] = ("Python", "TypeScript", "Go")

TextLLMCall = Callable[[str, str], str]
SearchWebCall = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class LanguageFindings(TypedDict):
    language: Language
    findings: str
    sources: list[str]


class ParallelizationState(TypedDict):
    question: str
    # Reducer required: research_python/typescript/go all write this field in
    # the SAME superstep. Without `operator.add`, LangGraph's default merge
    # (last write wins) would keep only one of the three findings.
    findings: Annotated[list[LanguageFindings], operator.add]
    final_answer: str


def initial_state(question: str) -> ParallelizationState:
    return {"question": question, "findings": [], "final_answer": ""}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESEARCHER_INSTRUCTIONS_TEMPLATE = """You are the {language} Researcher. Investigate ONLY
{language}'s suitability for building AI agents -- never compare it to, or even mention,
any other language. Focus on concurrency model, AI/agent libraries, and ecosystem
maturity. Do not write a final recommendation -- that is the Synthesizer's job."""

SYNTHESIZER_INSTRUCTIONS = """You are the Synthesizer. You receive independent findings
about several languages, gathered without any of the Researchers seeing each other's
work. Compare them and produce one final answer with an explicit, justified
recommendation. Do not introduce facts beyond what you were given."""


def _format_findings(findings: list[LanguageFindings]) -> str:
    sections = []
    for item in findings:
        section = f"### {item['language']}\n{item['findings']}"
        if item["sources"]:
            section += "\nSources: " + ", ".join(item["sources"])
        sections.append(section)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_researcher(language: Language, llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    """One Node per language. Each only ever sees its own language -- it has no
    visibility into the other Researchers' inputs or outputs (requirement:
    mutually independent subtasks)."""

    def researcher(state: ParallelizationState) -> ParallelizationState:
        search_result = search_web_call(f"{language} for AI agents")
        results = search_result.get("results", [])
        findings_text = llm_call(
            RESEARCHER_INSTRUCTIONS_TEMPLATE.format(language=language),
            f"Original question (context only): {state['question']}\nSearch results: {results}",
        )
        sources = [item["url"] for item in results if "url" in item]
        entry: LanguageFindings = {"language": language, "findings": findings_text, "sources": sources}
        return {"findings": [entry]}  # type: ignore[typeddict-item]

    return researcher


def make_synthesizer(llm_call: TextLLMCall):
    def synthesizer(state: ParallelizationState) -> ParallelizationState:
        user_prompt = (
            f"Original question: {state['question']}\n\nFindings:\n{_format_findings(state['findings'])}"
        )
        return {"final_answer": llm_call(SYNTHESIZER_INSTRUCTIONS, user_prompt)}  # type: ignore[typeddict-item]

    return synthesizer


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ParallelizationState)

    node_names = []
    for language in LANGUAGES:
        node_name = f"research_{language.lower()}"
        node_names.append(node_name)
        graph_builder.add_node(node_name, make_researcher(language, llm_call, search_web_call))

    graph_builder.add_node("synthesizer", make_synthesizer(llm_call))

    for node_name in node_names:
        graph_builder.add_edge(START, node_name)  # fan-out: all start together
        graph_builder.add_edge(node_name, "synthesizer")  # fan-in: joins once all finish

    graph_builder.add_edge("synthesizer", END)

    return graph_builder.compile()


def get_mermaid(graph) -> str:
    return graph.get_graph().draw_mermaid()
