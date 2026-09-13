"""Evaluator-Optimizer pattern: a Generator drafts an AI Agent Framework Comparison
Report, an independent Evaluator critiques it against fixed criteria, and -- if the
draft is not yet good enough -- the Evaluator's feedback is fed back into the
Generator for a revision. This repeats until the score clears a fixed bar or a hard
iteration cap is reached.

This module implements the *Evaluator-Optimizer* workflow from Anthropic's
"Building effective agents" (see ``docs/PATTERN-TAXONOMY.md``): unlike
Orchestrator-Workers (``src/orchestrator_workers.py``), there is no dynamic task
decomposition and no concurrent dispatch -- this is a strictly sequential
generate -> critique -> (maybe) regenerate loop over ONE artifact (the report).

Design constraints (all enforced by this module's own code, not left to the model):

* Generator and Evaluator are two separate ``Agent`` objects with different
  instructions and different responsibilities. Neither ever hands off to the
  other (``agents.handoff``/``Agent.handoffs`` is never used) -- this module's own
  Python code (``run_evaluator_optimizer``) is what decides whether to loop again
  and what to feed into the next Generator call.
* The Evaluator NEVER rewrites the report itself. It only ever returns a
  structured ``EvaluationResult`` (score, pass/fail, feedback, missing points). If
  it tried to also emit report text, that would blur Generator/Evaluator
  responsibilities and this module has no code path that would ever use such text.
* The Generator is the only agent that ever produces report text, and on every
  revision it is explicitly given the *previous draft* plus the Evaluator's
  *feedback* and *missing_points* so it revises rather than starting over blind.
* The accept/reject decision is made by THIS MODULE'S CODE, not by trusting the
  Evaluator's own boolean ``passed`` field: the code recomputes
  ``score >= PASS_SCORE`` itself and uses that as the single source of truth for
  whether to stop. Any disagreement between the Evaluator's own ``passed`` flag
  and the code's score-based decision is recorded (not silently trusted) -- see
  docs/05-EVALUATOR-OPTIMIZER.md Q3.
* ``MAX_ITERATIONS`` (a hard, code-enforced cap, default 3) bounds the loop no
  matter what the Evaluator says -- see docs/05-EVALUATOR-OPTIMIZER.md Q4.
* Every iteration's draft and evaluation are kept (``IterationRecord``), and the
  FINAL report returned is the best-scoring iteration seen so far, not
  necessarily the last one -- see docs/05-EVALUATOR-OPTIMIZER.md Q2.
* ``search_web`` is attached only to the Generator (it does the actual research);
  the Evaluator has no tools -- it only judges the text it is given.
* Every failure is re-raised as an ``EvaluatorOptimizerStepError`` naming exactly
  which step failed (``"generator-iteration-<n>"`` or ``"evaluator-iteration-<n>"``).

What the OpenAI Agents SDK still owns: each LLM call, structured output
parsing/validation, the Generator's own ``search_web`` tool-call loop, and tracing.
This module only decides the fixed shape of the loop (generate, evaluate, decide
whether to stop) and enforces the iteration cap and the score-based decision rule.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from src.tools import web_search as _web_search

PIPELINE_NAME = "Research evaluator-optimizer"

# Hard, code-enforced cap on how many generate/evaluate rounds may ever run for a
# single report, independent of what the Evaluator says.
MAX_ITERATIONS = 3

# Score threshold (0-10) at or above which a draft is accepted and the loop stops.
# This is a plain code-level constant -- the loop never relies on the Evaluator's
# own "passed" boolean as the sole authority (see module docstring and Q3).
PASS_SCORE = 8


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


class EvaluatorOptimizerStepError(RuntimeError):
    """Raised when a Generator or Evaluator call fails at a specific iteration."""

    def __init__(self, step_name: str, cause: BaseException) -> None:
        super().__init__(f"Evaluator-Optimizer failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Structured Evaluator output
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """The Evaluator's structured critique of one draft.

    The JSON key is ``pass`` (as requested), exposed here as the Python attribute
    ``passed`` because ``pass`` is a reserved keyword and cannot be a Python
    identifier. ``populate_by_name=True`` lets code also construct this model
    using the ``passed=`` keyword directly (e.g. in tests) while the wire format
    (and the JSON schema shown to the model) still uses ``pass``.
    """

    model_config = ConfigDict(populate_by_name=True)

    passed: bool = Field(
        alias="pass",
        description="Whether, in the Evaluator's own judgement, this draft is good enough to ship.",
    )
    score: int = Field(
        ge=0,
        le=10,
        description="Overall quality score from 0 (unacceptable) to 10 (excellent), evaluated "
        "against: (1) answers the user's question, (2) covers the major comparison "
        "dimensions, (3) has no obvious factual errors, (4) is backed by sufficient "
        "evidence, (5) is clearly structured, (6) does not omit important content.",
    )
    feedback: list[str] = Field(
        default_factory=list,
        description="Specific, actionable critique points the Generator should address on revision.",
    )
    missing_points: list[str] = Field(
        default_factory=list,
        description="Concrete comparison dimensions or content that are missing from the draft.",
    )


# ---------------------------------------------------------------------------
# Iteration bookkeeping and final result
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One full generate+evaluate round, kept for inspection regardless of outcome."""

    iteration: int
    draft: str
    evaluation: EvaluationResult
    accepted: bool
    score_pass_mismatch: bool = field(default=False)


@dataclass
class EvaluatorOptimizerResult:
    """The full, inspectable trail of a single Evaluator-Optimizer run."""

    topic: str
    iterations: list[IterationRecord]
    best_iteration: int
    final_report: str
    stopped_reason: str  # "passed_threshold" or "max_iterations_reached"


# ---------------------------------------------------------------------------
# search_web tool -- attached only to the Generator
# ---------------------------------------------------------------------------


@function_tool
def search_web(query: str) -> dict[str, Any]:
    """Search the web for current, factual information to support the report.

    Args:
        query: A focused web search query.
    """
    return _web_search(query)


# ---------------------------------------------------------------------------
# Model factory (mirrors src/agent.py, src/prompt_chaining.py, src/routing.py,
# src/parallelization.py, src/orchestrator_workers.py)
# ---------------------------------------------------------------------------


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# Generator -- writes (or revises) the report. Never judges its own work.
# ---------------------------------------------------------------------------

GENERATOR_INSTRUCTIONS = """You are the Generator in a research report writing system. Your ONLY
job is to write (or revise) an "AI Agent Framework Comparison Report" that answers the given
topic/question.

Rules:
- Use `search_web` to gather current, factual evidence before writing or revising.
- If you are given a PREVIOUS DRAFT plus EVALUATOR FEEDBACK and MISSING POINTS, you are
  revising, not starting over: preserve what was already good, and explicitly address every
  piece of feedback and every missing point. Do not ignore any of them.
- You never evaluate or score your own report, and you never claim it is "final" or "ready" --
  that is the Evaluator's job, not yours.
- Respond with the report text itself (plain text), well-structured with clear sections.
"""


def build_generator(model: Model | str) -> Agent:
    return Agent(
        name="Generator",
        instructions=GENERATOR_INSTRUCTIONS,
        model=model,
        tools=[search_web],
    )


# ---------------------------------------------------------------------------
# Evaluator -- critiques the report. Never rewrites it.
# ---------------------------------------------------------------------------

EVALUATOR_INSTRUCTIONS = f"""You are the Evaluator in a research report writing system. You are
independent from the Generator: you never write or rewrite any part of the report yourself, you
only critique the draft you are given.

Evaluate the draft strictly against these six criteria:
1. Does it actually answer the user's question?
2. Does it cover the major comparison dimensions relevant to the question?
3. Does it contain any obvious factual errors?
4. Is it backed by sufficient evidence (sources, concrete facts, not vague claims)?
5. Is it clearly structured (sections, comparisons, not a wall of text)?
6. Does it omit any important content a reader would expect?

Return a structured evaluation with:
- "pass": your own honest judgement of whether this draft is good enough to ship as-is.
- "score": an integer from 0 to 10 reflecting overall quality against all six criteria above.
  A score of {PASS_SCORE} or higher means the report is essentially ready to publish.
- "feedback": a list of specific, actionable critique points (empty if none).
- "missing_points": a list of concrete comparison dimensions or content that are missing
  (empty if none).

Do not include any report text of your own -- only the structured evaluation.
"""


def build_evaluator(model: Model | str) -> Agent:
    return Agent(
        name="Evaluator",
        instructions=EVALUATOR_INSTRUCTIONS,
        model=model,
        output_type=EvaluationResult,
    )


# ---------------------------------------------------------------------------
# One Generator call, one Evaluator call
# ---------------------------------------------------------------------------


def _build_generator_input(
    topic: str,
    *,
    previous_draft: str | None,
    feedback: list[str] | None,
    missing_points: list[str] | None,
) -> str:
    if previous_draft is None:
        return (
            "Write an AI Agent Framework Comparison Report that answers the following topic:\n"
            f"{topic}"
        )

    feedback_block = "\n".join(f"- {item}" for item in (feedback or [])) or "(none)"
    missing_block = "\n".join(f"- {item}" for item in (missing_points or [])) or "(none)"
    return (
        f"Topic: {topic}\n\n"
        f"PREVIOUS DRAFT:\n{previous_draft}\n\n"
        f"EVALUATOR FEEDBACK to address:\n{feedback_block}\n\n"
        f"MISSING POINTS to add:\n{missing_block}\n\n"
        "Revise the report, explicitly addressing every piece of feedback and every missing "
        "point above. Do not simply resubmit the previous draft unchanged."
    )


async def _run_generator(
    topic: str,
    model: Model | str,
    *,
    iteration: int,
    previous_draft: str | None,
    feedback: list[str] | None,
    missing_points: list[str] | None,
    max_turns: int,
) -> str:
    try:
        generator = build_generator(model)
        generator_input = _build_generator_input(
            topic, previous_draft=previous_draft, feedback=feedback, missing_points=missing_points
        )
        result = await Runner.run(
            generator,
            generator_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        if not isinstance(result.final_output, str):
            raise TypeError("Generator must return plain-text report content.")
        return result.final_output
    except Exception as exc:  # noqa: BLE001 -- re-raised with the failing iteration named
        raise EvaluatorOptimizerStepError(f"generator-iteration-{iteration}", exc) from exc


async def _run_evaluator(
    topic: str, draft: str, model: Model | str, *, iteration: int, max_turns: int
) -> EvaluationResult:
    try:
        evaluator = build_evaluator(model)
        evaluator_input = f"Topic: {topic}\n\nReport draft to evaluate:\n{draft}"
        result = await Runner.run(
            evaluator,
            evaluator_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        return result.final_output_as(EvaluationResult, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001
        raise EvaluatorOptimizerStepError(f"evaluator-iteration-{iteration}", exc) from exc


# ---------------------------------------------------------------------------
# The fixed generate -> evaluate -> (maybe) revise loop
# ---------------------------------------------------------------------------


async def run_evaluator_optimizer(
    topic: str,
    model: Model | str,
    *,
    max_iterations: int = MAX_ITERATIONS,
    pass_score: int = PASS_SCORE,
    max_turns: int = 6,
) -> EvaluatorOptimizerResult:
    """Run the fixed Generator -> Evaluator -> (maybe revise) loop.

    The *shape* of this control flow (generate, evaluate, decide whether to loop)
    is fixed by this function and by ``max_iterations``/``pass_score``, both of
    which are plain code-level parameters -- the Evaluator's output can influence
    WHAT the next draft looks like (via feedback/missing_points), but it can never
    make the loop run more than ``max_iterations`` times, and it never single-
    handedly decides to stop: the code recomputes ``score >= pass_score`` itself.
    """
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    iterations: list[IterationRecord] = []
    previous_draft: str | None = None
    feedback: list[str] | None = None
    missing_points: list[str] | None = None
    stopped_reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        draft = await _run_generator(
            topic,
            model,
            iteration=iteration,
            previous_draft=previous_draft,
            feedback=feedback,
            missing_points=missing_points,
            max_turns=max_turns,
        )
        evaluation = await _run_evaluator(topic, draft, model, iteration=iteration, max_turns=max_turns)

        # The code -- not the Evaluator's own "passed" flag -- has the final say on
        # whether this draft is accepted. Any disagreement is recorded, not hidden.
        accepted = evaluation.score >= pass_score
        mismatch = accepted != evaluation.passed
        iterations.append(
            IterationRecord(
                iteration=iteration,
                draft=draft,
                evaluation=evaluation,
                accepted=accepted,
                score_pass_mismatch=mismatch,
            )
        )

        if accepted:
            stopped_reason = "passed_threshold"
            break

        previous_draft = draft
        feedback = evaluation.feedback
        missing_points = evaluation.missing_points

    # Final report is the BEST-scoring iteration seen, not necessarily the last one
    # -- if scores regress or plateau, we do not silently ship a worse revision.
    best = max(iterations, key=lambda record: record.evaluation.score)

    return EvaluatorOptimizerResult(
        topic=topic,
        iterations=iterations,
        best_iteration=best.iteration,
        final_report=best.draft,
        stopped_reason=stopped_reason,
    )


def run_evaluator_optimizer_sync(
    topic: str,
    model: Model | str,
    *,
    max_iterations: int = MAX_ITERATIONS,
    pass_score: int = PASS_SCORE,
    max_turns: int = 6,
) -> EvaluatorOptimizerResult:
    """Synchronous convenience wrapper around ``run_evaluator_optimizer`` for CLI/tests."""
    return asyncio.run(
        run_evaluator_optimizer(
            topic, model, max_iterations=max_iterations, pass_score=pass_score, max_turns=max_turns
        )
    )


def main() -> None:
    """CLI entry point: run the loop once and print every iteration plus the final report."""
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    enable_local_tracing()

    topic = " ".join(sys.argv[1:]).strip() or (
        "Compare the leading AI Agent frameworks (e.g. OpenAI Agents SDK, LangGraph, "
        "AutoGen/AG2, CrewAI) for building production research agents, and recommend one."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)

    try:
        result = run_evaluator_optimizer_sync(topic, model)
    except EvaluatorOptimizerStepError as exc:
        raise SystemExit(f"Pipeline failed at step: {exc.step_name}\n{exc}") from exc

    print("=" * 80)
    print(f"TOPIC: {result.topic}")
    print("=" * 80)

    for record in result.iterations:
        print(
            f"\n--- Iteration {record.iteration} "
            f"(score={record.evaluation.score}, pass={record.evaluation.passed}, "
            f"accepted={record.accepted}"
            + (", MISMATCH between score and pass flag" if record.score_pass_mismatch else "")
            + ") ---"
        )
        if record.evaluation.feedback:
            print("Feedback:")
            for item in record.evaluation.feedback:
                print(f"  - {item}")
        if record.evaluation.missing_points:
            print("Missing points:")
            for item in record.evaluation.missing_points:
                print(f"  - {item}")

    print("\n" + "=" * 80)
    print(
        f"STOPPED: {result.stopped_reason} | best iteration: {result.best_iteration} "
        f"(score={result.iterations[result.best_iteration - 1].evaluation.score})"
    )
    print("=" * 80)
    print(result.final_report)


if __name__ == "__main__":
    main()
