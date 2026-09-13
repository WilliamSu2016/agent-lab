"""Routing pattern: classify a question, then dispatch to exactly one Specialist.

This module implements the *Routing* workflow from Anthropic's "Building
effective agents" (see ``docs/PATTERN-TAXONOMY.md``): a single Router step
classifies the input, and program code maps that classification to exactly
one specialized downstream Agent. The other two Specialists never run.

Design constraints (all enforced by this module's own code, not by the SDK):

* Exactly one Router step, using a structured ``output_type`` (``RouterDecision``)
  so the category is a validated enum value, not free-form text this module
  would have to parse itself.
* Router's ONLY job is classification -- it never answers the question, and
  its instructions explicitly say so.
* Program code (``run_routing``), not an LLM, maps the category to a
  specialist builder via a plain ``dict`` lookup (``_SPECIALIST_BUILDERS``).
  The Router's output value selects a key in that fixed mapping; it cannot
  invent a new route or make the mapping itself dynamic.
* Exactly one Specialist agent actually runs per question (requirement 7:
  "do not let every Agent handle the same question"). The other two
  Specialists are never instantiated, let alone called, for that question.
* No multi-agent handoffs (``agents.handoff``/``Agent.handoffs`` is never
  used) -- the Router does not hand off control to a Specialist; this
  module's Python code invokes the chosen Specialist directly via a second,
  separate ``Runner.run_sync`` call.
* No Orchestrator-Workers: there is no central LLM that dynamically breaks
  the question into an unknown number of sub-tasks. There are exactly three
  fixed, pre-defined routes (Technical / Business / General), and the Router
  only *selects* one of them; it never invents a new route or decomposes the
  question into multiple parallel sub-questions.
* No Parallelization: exactly one Specialist runs, never more than one, and
  never as a fan-out/fan-in of multiple simultaneous calls.
* No Evaluator-Optimizer: there is no feedback loop that critiques the
  Specialist's answer and asks it to revise; the Specialist's first answer is
  returned as-is.
* ``search_web`` (a thin wrapper around ``src.tools.web_search``) is attached
  to all three Specialists (each may use it if their own instructions call
  for it), but never to the Router -- the Router classifies from the
  question text alone, with no tool access.
* Every failure is re-raised as a ``RoutingStepError`` naming exactly which
  step failed ("router" or the specific specialist's route name).

What the OpenAI Agents SDK still owns: each LLM call, structured output
parsing/validation for the Router, the Specialist's own tool-call loop (if it
decides to call ``search_web``), and tracing. This module only decides *which
one* Specialist runs, based on the Router's validated classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.tools import web_search as _web_search

PIPELINE_NAME = "Research router (routing)"

Category = Literal["technical", "business", "general"]


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


class RoutingStepError(RuntimeError):
    """Raised when the Router or the chosen Specialist fails, naming that step."""

    def __init__(self, step_name: str, cause: BaseException) -> None:
        super().__init__(f"Routing failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Router's structured output: a validated category, nothing else
# ---------------------------------------------------------------------------


class RouterDecision(BaseModel):
    """Router step output: which single route to take, and why."""

    category: Category = Field(
        description=(
            "Exactly one of 'technical', 'business', or 'general', describing "
            "which specialist should handle this question."
        )
    )
    reason: str = Field(description="One short sentence explaining the classification.")


@dataclass
class RoutingResult:
    """The full, inspectable trail of a single Routing run."""

    question: str
    decision: RouterDecision
    final_answer: str


# ---------------------------------------------------------------------------
# search_web tool -- available to every Specialist, never to the Router
# ---------------------------------------------------------------------------


@function_tool
def search_web(query: str) -> dict[str, Any]:
    """Search the web for current, factual information relevant to the question.

    Args:
        query: A focused web search query.
    """
    return _web_search(query)


# ---------------------------------------------------------------------------
# Model factory (mirrors src/agent.py and src/prompt_chaining.py)
# ---------------------------------------------------------------------------


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# Router -- question -> RouterDecision (classification only, no tools)
# ---------------------------------------------------------------------------

ROUTER_INSTRUCTIONS = """You are the Router in a research routing system. You always run first,
and your ONLY job is to classify the user's question into exactly one category:

- "technical": questions about technology choices, frameworks, SDKs, libraries, APIs,
  architecture, implementation details, code, or how something works internally
  (e.g. comparing two frameworks/SDKs, "how does X work", "what is the architecture of Y").
- "business": questions about markets, business models, costs, pricing, competitors,
  opportunities, ROI, adoption, or go-to-market strategy for a product/technology.
- "general": broad or introductory knowledge questions that are not specifically about
  technical implementation nor about business/market strategy (e.g. "what is RAG?",
  "what does 'agent' mean in AI?").

Rules:
- Do NOT answer the question yourself. Do NOT search the web -- you have no tools.
- Choose exactly one category. If the question could fit more than one, pick the single
  best-fitting category rather than trying to cover all angles.
- Give one short reason for your classification.
"""


def build_router(model: Model | str) -> Agent:
    return Agent(
        name="Router",
        instructions=ROUTER_INSTRUCTIONS,
        model=model,
        output_type=RouterDecision,
    )


# ---------------------------------------------------------------------------
# Specialist: Technical Researcher
# ---------------------------------------------------------------------------

TECHNICAL_RESEARCHER_INSTRUCTIONS = """You are the Technical Researcher. You only run when the
Router has classified a question as "technical".

Focus areas: technical documentation, official docs/READMEs, GitHub repositories and issues,
release notes, source code, APIs, SDKs, architecture, design trade-offs, and implementation
details. When comparing two technologies, focus on concrete technical differences: APIs,
concurrency/execution model, extensibility, tooling, ecosystem maturity, and how they are
actually implemented -- not on market size, pricing, or business strategy.

Rules:
- Use `search_web` to find current, factual technical information (e.g. official docs,
  GitHub, changelogs). Prefer authoritative technical sources over general blog posts.
- Be explicit about version-specific or fast-changing details, since technical ecosystems
  change quickly.
- If search results are insufficient, say so explicitly instead of guessing.
- End with a concise, technically grounded answer, citing the source URLs you used.
"""


def build_technical_researcher(model: Model | str) -> Agent:
    return Agent(
        name="Technical Researcher",
        instructions=TECHNICAL_RESEARCHER_INSTRUCTIONS,
        model=model,
        tools=[search_web],
    )


# ---------------------------------------------------------------------------
# Specialist: Business Researcher
# ---------------------------------------------------------------------------

BUSINESS_RESEARCHER_INSTRUCTIONS = """You are the Business Researcher. You only run when the
Router has classified a question as "business".

Focus areas: market size and growth, business models, monetization, pricing, cost structure,
competitive landscape, market opportunities/gaps, customer segments, adoption trends, and
go-to-market strategy. When discussing a technology, focus on its business and market
implications -- not on its internal architecture or implementation details.

Rules:
- Use `search_web` to find current market data, competitor information, pricing, or industry
  analysis. Prefer recent, named sources (reports, funding news, pricing pages) over vague
  generalities.
- Distinguish clearly between verified findings and your own inference/estimate.
- If search results are insufficient or outdated, say so explicitly instead of guessing.
- End with a concise, business-focused answer, citing the source URLs you used.
"""


def build_business_researcher(model: Model | str) -> Agent:
    return Agent(
        name="Business Researcher",
        instructions=BUSINESS_RESEARCHER_INSTRUCTIONS,
        model=model,
        tools=[search_web],
    )


# ---------------------------------------------------------------------------
# Specialist: General Researcher
# ---------------------------------------------------------------------------

GENERAL_RESEARCHER_INSTRUCTIONS = """You are the General Researcher. You only run when the
Router has classified a question as "general" -- broad or introductory knowledge questions
that are not specifically about technical implementation nor about business/market strategy.

Focus areas: clear, accessible explanations of concepts, terms, and general knowledge
(e.g. "what is RAG?", "what does 'agent' mean in AI?").

Rules:
- Use `search_web` only if the question needs current or fact-checkable information; for
  well-established concepts, you may answer directly from your own knowledge.
- Keep the answer clear and approachable -- avoid unnecessary technical or business jargon.
- If you do search and results are insufficient, say so explicitly instead of guessing.
- If you used `search_web`, cite the source URLs you used.
"""


def build_general_researcher(model: Model | str) -> Agent:
    return Agent(
        name="General Researcher",
        instructions=GENERAL_RESEARCHER_INSTRUCTIONS,
        model=model,
        tools=[search_web],
    )


# ---------------------------------------------------------------------------
# The fixed category -> Specialist mapping (program code, not an LLM decision)
# ---------------------------------------------------------------------------

_SPECIALIST_BUILDERS: dict[Category, Callable[[Model | str], Agent]] = {
    "technical": build_technical_researcher,
    "business": build_business_researcher,
    "general": build_general_researcher,
}


def run_routing(question: str, model: Model | str, *, max_turns: int = 6) -> RoutingResult:
    """Classify the question, then run exactly one Specialist for it.

    Only the Router's ``category`` decides which Specialist runs; that
    decision is looked up in the fixed ``_SPECIALIST_BUILDERS`` mapping by
    this function's own code. There is no Orchestrator, no dynamic task
    decomposition, and no parallel fan-out -- exactly one Specialist agent is
    built and run per call.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    # Step 1 -- Router: question -> RouterDecision (classification only)
    try:
        router = build_router(model)
        router_result = Runner.run_sync(
            router,
            question,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        decision = router_result.final_output_as(RouterDecision, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the failing step named
        raise RoutingStepError("router", exc) from exc

    # Step 2 -- exactly one Specialist, chosen by fixed dict lookup on decision.category
    build_specialist = _SPECIALIST_BUILDERS[decision.category]
    try:
        specialist = build_specialist(model)
        specialist_result = Runner.run_sync(
            specialist,
            question,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        if not isinstance(specialist_result.final_output, str):
            raise TypeError(f"{specialist.name} must return a plain-text final answer.")
        final_answer = specialist_result.final_output
    except Exception as exc:  # noqa: BLE001
        raise RoutingStepError(decision.category, exc) from exc

    return RoutingResult(question=question, decision=decision, final_answer=final_answer)


def main() -> None:
    """CLI entry point: route one question and print the Router's decision and the answer."""
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    enable_local_tracing()

    question = " ".join(sys.argv[1:]).strip() or "什么是 RAG？"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the router.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)

    try:
        result = run_routing(question, model)
    except RoutingStepError as exc:
        raise SystemExit(f"Routing failed at step: {exc.step_name}\n{exc}") from exc

    print("=" * 80)
    print(f"ROUTER DECISION: category={result.decision.category!r}")
    print(f"Reason: {result.decision.reason}")
    print("=" * 80)
    print(result.final_answer)


if __name__ == "__main__":
    main()
