"""Prompt Chaining pattern: Planner -> Researcher -> Analyst -> Writer.

This module implements the *Prompt Chaining* workflow from Anthropic's
"Building effective agents" (see ``docs/PATTERN-TAXONOMY.md``): a task is
decomposed into a fixed sequence of steps, and each step's LLM call consumes
only the previous step's output.

Design constraints (all enforced by this module's own code, not by the SDK):

* Exactly four steps, always run in this fixed order:
  ``Planner -> Researcher -> Analyst -> Writer``.
* Each step is a *separate* ``agents.Agent`` with its own narrow
  ``instructions`` -- one responsibility per step.
* Step ``N``'s input is built *only* from step ``N-1``'s output (plus the
  original question, which never changes). No step is allowed to see raw
  material that skips a pipeline stage.
* No multi-agent handoffs (``agents.handoff``/``Agent.handoffs`` is never
  used) -- steps are invoked directly by this module's Python code, not by an
  LLM deciding to hand off control to another agent.
* No orchestrator and no dynamic task decomposition: the number and order of
  steps is fixed in this file and never chosen or changed by a model at run
  time. The Planner LLM only *fills in* the research dimensions inside a
  fixed step; it does not decide whether the Researcher/Analyst/Writer steps
  run, or in what order.
* ``search_web`` (a thin wrapper around ``src.tools.web_search``) is the only
  tool, and it is only attached to the Researcher step.
* Each step's ``Runner.run_sync`` call is wrapped so that a failure is always
  re-raised as a ``PromptChainStepError`` naming exactly which step failed.

What the OpenAI Agents SDK still owns, per step: the LLM call, structured
output parsing/validation (via ``Agent.output_type``), the tool-call loop
(only the Researcher step has a tool, so only that step's model may loop),
and tracing. This module only decides *what* runs *in what order*, and *what
data* flows from one step to the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.tools import web_search as _web_search

PIPELINE_NAME = "Language choice research (prompt chaining)"


# ---------------------------------------------------------------------------
# Failure reporting (requirement 9: report exactly which step failed)
# ---------------------------------------------------------------------------


class PromptChainStepError(RuntimeError):
    """Raised when one fixed pipeline step fails, naming that step explicitly."""

    def __init__(self, step_name: str, cause: BaseException) -> None:
        super().__init__(f"Prompt chain failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Step input/output contracts
#
# Structured ``output_type``s make the boundary between steps explicit and
# machine-checkable: step N+1 cannot silently receive more (or less) than
# what step N actually produced.
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Step 1 (Planner) output: only the research dimensions, nothing else."""

    dimensions: list[str] = Field(
        description=(
            "Distinct, non-overlapping angles to research before comparing the "
            "languages (e.g. concurrency model, ecosystem maturity, tooling)."
        ),
        min_length=2,
    )


class DimensionFindings(BaseModel):
    """Research notes gathered for a single dimension."""

    dimension: str
    findings: str = Field(description="A factual summary of what was found for this dimension.")
    sources: list[str] = Field(default_factory=list)


class ResearchBundle(BaseModel):
    """Step 2 (Researcher) output: raw findings only, no comparison or opinion yet."""

    notes: list[DimensionFindings]


@dataclass
class PromptChainResult:
    """The full, inspectable trail of a single Prompt Chaining run."""

    question: str
    plan: ResearchPlan
    research: ResearchBundle
    analysis: str
    final_answer: str


# ---------------------------------------------------------------------------
# search_web tool -- attached only to the Researcher step
# ---------------------------------------------------------------------------


@function_tool
def search_web(query: str) -> dict[str, Any]:
    """Search the web for current, factual information about one research dimension.

    Args:
        query: A focused web search query covering a single research dimension.
    """
    return _web_search(query)


# ---------------------------------------------------------------------------
# Model factory (mirrors src/agent.py: a Chat Completions-only gateway)
# ---------------------------------------------------------------------------


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# Step 1 -- Planner: question -> ResearchPlan
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTIONS = """You are the Planner step in a fixed 4-step research pipeline
(Planner -> Researcher -> Analyst -> Writer). You always run first.

Your ONLY job: given the user's comparison question, produce a short list of distinct,
non-overlapping research dimensions that must be investigated before anyone can compare
the options meaningfully (for example: performance/concurrency model, ecosystem and
library maturity, tooling and IDE support, learning curve, hiring/talent pool, or
deployment story -- pick the dimensions that actually fit this question).

Rules:
- Do NOT answer the user's question yourself.
- Do NOT search the web or invent facts -- you have no tools and no evidence yet.
- Do NOT compare the options yourself -- that happens in a later step.
- Return between 3 and 6 dimensions, each a short phrase, with no duplicates.
"""


def build_planner(model: Model | str) -> Agent:
    return Agent(
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        model=model,
        output_type=ResearchPlan,
    )


# ---------------------------------------------------------------------------
# Step 2 -- Researcher: ResearchPlan -> ResearchBundle (may call search_web)
# ---------------------------------------------------------------------------

RESEARCHER_INSTRUCTIONS = """You are the Researcher step in a fixed 4-step research pipeline
(Planner -> Researcher -> Analyst -> Writer). You always run second, after the Planner.

You receive the original question and a fixed list of research dimensions decided by the
Planner step. Your ONLY job: use the `search_web` tool to gather factual, current
information for EACH dimension, and record what you found.

Rules:
- Call `search_web` at least once per dimension (more if the first results are thin).
- Do NOT skip a dimension and do NOT add new dimensions of your own.
- Do NOT compare the options or give an opinion/recommendation -- that is not your job.
- For each dimension, report a factual findings summary and the source URLs you used.
- If search results are insufficient for a dimension, say so explicitly in `findings`
  instead of guessing.
"""


def build_researcher(model: Model | str) -> Agent:
    return Agent(
        name="Researcher",
        instructions=RESEARCHER_INSTRUCTIONS,
        model=model,
        tools=[search_web],
        output_type=ResearchBundle,
    )


# ---------------------------------------------------------------------------
# Step 3 -- Analyst: ResearchBundle -> analysis text (no tools)
# ---------------------------------------------------------------------------

ANALYST_INSTRUCTIONS = """You are the Analyst step in a fixed 4-step research pipeline
(Planner -> Researcher -> Analyst -> Writer). You always run third, after the Researcher.

You receive the original question and the raw research notes gathered by the Researcher
step, organized by dimension. Your ONLY job: analyze and compare the options against each
dimension, noting concrete strengths/weaknesses and any conflicting or missing evidence.

Rules:
- Do NOT call any tools and do NOT search the web -- work only from the notes you were given.
- Do NOT write the final answer or a recommendation -- that is the Writer step's job.
- Structure your analysis dimension by dimension, referencing the evidence you were given.
- Be explicit whenever the evidence is thin or contradictory for a dimension.
"""


def build_analyst(model: Model | str) -> Agent:
    return Agent(
        name="Analyst",
        instructions=ANALYST_INSTRUCTIONS,
        model=model,
    )


# ---------------------------------------------------------------------------
# Step 4 -- Writer: analysis -> final answer text (no tools)
# ---------------------------------------------------------------------------

WRITER_INSTRUCTIONS = """You are the Writer step in a fixed 4-step research pipeline
(Planner -> Researcher -> Analyst -> Writer). You always run last, after the Analyst.

You receive the original question and the Analyst's dimension-by-dimension comparison.
Your ONLY job: write the final answer for the user -- a clear, well-organized comparison
plus one explicit, justified recommendation.

Rules:
- Do NOT call any tools and do NOT introduce new facts beyond the given analysis.
- Do NOT redo the research or second-guess the earlier pipeline steps.
- End with an explicit recommendation, stating which option to prefer and the top reasons.
- Respond in the same language as the user's original question.
"""


def build_writer(model: Model | str) -> Agent:
    return Agent(
        name="Writer",
        instructions=WRITER_INSTRUCTIONS,
        model=model,
    )


# ---------------------------------------------------------------------------
# The fixed pipeline itself
# ---------------------------------------------------------------------------


def _format_research_notes(bundle: ResearchBundle) -> str:
    """Render step 2's structured output as plain text for step 3's input."""
    sections: list[str] = []
    for note in bundle.notes:
        section = f"### {note.dimension}\n{note.findings}"
        if note.sources:
            section += "\nSources: " + ", ".join(note.sources)
        sections.append(section)
    return "\n\n".join(sections)


def run_prompt_chain(question: str, model: Model | str, *, max_turns: int = 6) -> PromptChainResult:
    """Run the fixed Planner -> Researcher -> Analyst -> Writer chain, in that order.

    The four steps are invoked sequentially by this function -- there is no
    Runner/LLM decision anywhere about whether a step runs, which step runs
    next, or how many steps exist. Each step's input is built only from the
    previous step's (validated) output plus the unchanging original question.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    # Step 1 -- Planner: question -> ResearchPlan
    try:
        planner = build_planner(model)
        planner_result = Runner.run_sync(
            planner,
            question,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        plan = planner_result.final_output_as(ResearchPlan, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the failing step named
        raise PromptChainStepError("1-planner", exc) from exc

    # Step 2 -- Researcher: ResearchPlan -> ResearchBundle
    try:
        researcher = build_researcher(model)
        researcher_input = (
            f"Original question: {question}\n"
            "Research dimensions to investigate (do not add or skip any):\n"
            + "\n".join(f"- {dimension}" for dimension in plan.dimensions)
        )
        researcher_result = Runner.run_sync(
            researcher,
            researcher_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        research = researcher_result.final_output_as(ResearchBundle, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001
        raise PromptChainStepError("2-researcher", exc) from exc

    # Step 3 -- Analyst: ResearchBundle -> analysis text
    try:
        analyst = build_analyst(model)
        analyst_input = (
            f"Original question: {question}\n\nResearch notes:\n{_format_research_notes(research)}"
        )
        analyst_result = Runner.run_sync(
            analyst,
            analyst_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        if not isinstance(analyst_result.final_output, str):
            raise TypeError("Analyst step must return plain text analysis.")
        analysis = analyst_result.final_output
    except Exception as exc:  # noqa: BLE001
        raise PromptChainStepError("3-analyst", exc) from exc

    # Step 4 -- Writer: analysis -> final answer text
    try:
        writer = build_writer(model)
        writer_input = f"Original question: {question}\n\nAnalysis:\n{analysis}"
        writer_result = Runner.run_sync(
            writer,
            writer_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        if not isinstance(writer_result.final_output, str):
            raise TypeError("Writer step must return plain text final answer.")
        final_answer = writer_result.final_output
    except Exception as exc:  # noqa: BLE001
        raise PromptChainStepError("4-writer", exc) from exc

    return PromptChainResult(
        question=question,
        plan=plan,
        research=research,
        analysis=analysis,
        final_answer=final_answer,
    )


def main() -> None:
    """CLI entry point: run the fixed 4-step chain once and print every step's output."""
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    enable_local_tracing()

    question = " ".join(sys.argv[1:]).strip() or (
        "比较 Python、TypeScript、Go 哪个更适合开发 AI Agent，并给出推荐。"
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)

    try:
        result = run_prompt_chain(question, model)
    except PromptChainStepError as exc:
        raise SystemExit(f"Pipeline failed at step: {exc.step_name}\n{exc}") from exc

    print("=" * 80)
    print("STEP 1 -- Planner: research dimensions")
    print("=" * 80)
    for dimension in result.plan.dimensions:
        print(f"- {dimension}")

    print("\n" + "=" * 80)
    print("STEP 2 -- Researcher: findings")
    print("=" * 80)
    print(_format_research_notes(result.research))

    print("\n" + "=" * 80)
    print("STEP 3 -- Analyst: comparison")
    print("=" * 80)
    print(result.analysis)

    print("\n" + "=" * 80)
    print("STEP 4 -- Writer: final answer")
    print("=" * 80)
    print(result.final_answer)


if __name__ == "__main__":
    main()
