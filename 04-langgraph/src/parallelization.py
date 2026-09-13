"""Parallelization pattern (sectioning): three independent Researchers run concurrently,
then a Synthesizer combines their results into one final answer.

This module implements the *Parallelization* workflow from Anthropic's
"Building effective agents" (see ``docs/PATTERN-TAXONOMY.md``), specifically
the *sectioning* variant: an independently divisible task (comparing three
languages) is split into fixed, non-overlapping subtasks that are run at the
same time and then aggregated programmatically.

Design constraints (all enforced by this module's own code, not by the SDK):

* Exactly three Researcher agents -- Python / TypeScript / Go -- each with
  its own narrow ``instructions`` covering only its own language. None of
  them is told about, or has any way to see, the other two languages'
  research (requirement 1: mutually independent).
* The three Researchers are launched together via ``asyncio.gather`` over
  ``Runner.run`` (the SDK's async entry point), so they execute concurrently
  from this process's point of view (requirement 2 & 4: actual parallel
  execution via asyncio, not just "conceptually parallel" sequential calls).
* Each Researcher's input is only the language name plus the original
  question -- never another Researcher's output. There is no data
  dependency between the three Researcher calls, which is what makes running
  them concurrently valid in the first place (see docs/03-PARALLELIZATION.md
  Q2 for what would break if there were a dependency).
* A single Synthesizer agent, run *after* all three Researchers have
  completed (via ``asyncio.gather``'s join semantics), receives all three
  findings and produces the final comparison + recommendation.
* No Orchestrator: the three subtasks (research Python / TypeScript / Go) are
  fixed in this file ahead of time. No LLM decides at run time how many
  Researchers to spawn or what each one should investigate -- that division
  of labor is hard-coded, unlike Orchestrator-Workers where an LLM decides
  the breakdown dynamically.
* No Routing: all three Researchers always run for every question; there is
  no classification step that picks only one of them.
* No Evaluator-Optimizer: the Synthesizer's first output is the final
  answer; there is no critique-and-revise loop.
* ``search_web`` (a thin wrapper around ``src.tools.web_search``) is attached
  to all three Researchers, never to the Synthesizer (which only reasons
  over the three findings it is given).
* Every step's start/end timestamps are recorded (``StepTiming``) so
  sequential vs. parallel latency can be measured and compared, per the
  task's requirement to log each agent's start/end time.

What the OpenAI Agents SDK still owns: each LLM call, structured output
parsing/validation, each Researcher's own ``search_web`` tool-call loop, and
tracing. This module only decides *how many* Researchers exist, *what data*
each receives, and *how* (concurrently vs. sequentially) they are invoked --
the concurrency itself is provided by Python's ``asyncio``, not by the SDK.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.tools import web_search as _web_search

PIPELINE_NAME = "Language comparison (parallelization)"

Language = Literal["Python", "TypeScript", "Go"]
LANGUAGES: tuple[Language, ...] = ("Python", "TypeScript", "Go")


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


class ParallelizationStepError(RuntimeError):
    """Raised when a Researcher or the Synthesizer fails, naming that step."""

    def __init__(self, step_name: str, cause: BaseException) -> None:
        super().__init__(f"Parallelization failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Timing -- requirement: record each agent's start/end time, to compare
# sequential vs. parallel latency.
# ---------------------------------------------------------------------------


@dataclass
class StepTiming:
    """Wall-clock start/end time (seconds, ``time.perf_counter()``) for one step."""

    name: str
    started_at: float
    ended_at: float

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class TimingReport:
    """All recorded step timings for one run, plus the overall wall-clock duration."""

    steps: list[StepTiming] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    def add(self, timing: StepTiming) -> None:
        self.steps.append(timing)

    def summary_lines(self) -> list[str]:
        lines = [
            f"- {t.name}: started={t.started_at:.3f}s ended={t.ended_at:.3f}s "
            f"duration={t.duration_seconds:.3f}s"
            for t in self.steps
        ]
        lines.append(f"- TOTAL wall-clock: {self.wall_clock_seconds:.3f}s")
        return lines


# ---------------------------------------------------------------------------
# Structured output: one Researcher's findings for its single language
# ---------------------------------------------------------------------------


class LanguageFindings(BaseModel):
    """One Researcher's independent findings about its single assigned language."""

    language: Language
    findings: str = Field(
        description="A factual summary of this language's suitability for building AI agents."
    )
    sources: list[str] = Field(default_factory=list)


@dataclass
class ParallelizationResult:
    """The full, inspectable trail of a single Parallelization run."""

    question: str
    findings: list[LanguageFindings]
    final_answer: str
    timing: TimingReport


# ---------------------------------------------------------------------------
# search_web tool -- attached to all three Researchers, never to the Synthesizer
# ---------------------------------------------------------------------------


@function_tool
def search_web(query: str) -> dict[str, Any]:
    """Search the web for current, factual information about one programming language.

    Args:
        query: A focused web search query about a single language.
    """
    return _web_search(query)


# ---------------------------------------------------------------------------
# Model factory (mirrors src/agent.py, src/prompt_chaining.py, src/routing.py)
# ---------------------------------------------------------------------------


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# Researcher -- one per language, all built from the same template but each
# instructed to look at ONLY its own language (requirement 3 & 1).
# ---------------------------------------------------------------------------

RESEARCHER_INSTRUCTIONS_TEMPLATE = """You are the {language} Researcher. You investigate
ONLY {language}'s suitability for building AI agents -- never compare it to other
languages, and never mention other languages by name.

Focus areas: {language}'s concurrency/execution model, relevant AI/agent SDKs and
libraries, ecosystem maturity, tooling, and any notable strengths or weaknesses for
building AI agents specifically.

Rules:
- Use `search_web` to find current, factual information about {language} for this purpose.
- Do NOT discuss or compare against any other programming language -- you have no
  visibility into what the other Researchers find, and neither should your answer.
- Do NOT write a final recommendation -- that is the Synthesizer's job, not yours.
- If search results are insufficient, say so explicitly instead of guessing.
"""


def build_researcher(model: Model | str, language: Language) -> Agent:
    return Agent(
        name=f"{language} Researcher",
        instructions=RESEARCHER_INSTRUCTIONS_TEMPLATE.format(language=language),
        model=model,
        tools=[search_web],
        output_type=LanguageFindings,
    )


# ---------------------------------------------------------------------------
# Synthesizer -- runs once, after all three Researchers have finished
# ---------------------------------------------------------------------------

SYNTHESIZER_INSTRUCTIONS = """You are the Synthesizer. You run once, after three independent
Researchers (Python, TypeScript, and Go) have each investigated their own language in
isolation. You receive all three findings together for the first time.

Your ONLY job: compare the three languages against each other using the findings you were
given, and produce a final, well-organized answer with one explicit, justified
recommendation for building AI agents.

Rules:
- Do NOT call any tools and do NOT introduce new facts beyond the three findings you were given.
- Do NOT redo the research or second-guess what each Researcher reported.
- Be explicit about trade-offs between the three languages, not just a single winner.
- End with an explicit recommendation, stating which language to prefer and the top reasons.
- Respond in the same language as the user's original question.
"""


def build_synthesizer(model: Model | str) -> Agent:
    return Agent(
        name="Synthesizer",
        instructions=SYNTHESIZER_INSTRUCTIONS,
        model=model,
    )


# ---------------------------------------------------------------------------
# Concurrent execution of the three Researchers
# ---------------------------------------------------------------------------


async def _run_researcher(
    language: Language,
    question: str,
    model: Model | str,
    *,
    max_turns: int,
    timing: TimingReport,
) -> LanguageFindings:
    """Run one Researcher and record its own start/end wall-clock time."""
    started_at = time.perf_counter()
    try:
        researcher = build_researcher(model, language)
        result = await Runner.run(
            researcher,
            f"Original question (for context only): {question}\nInvestigate: {language}",
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        findings = result.final_output_as(LanguageFindings, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the failing step named
        raise ParallelizationStepError(f"researcher-{language.lower()}", exc) from exc
    finally:
        timing.add(StepTiming(f"{language} Researcher", started_at, time.perf_counter()))
    return findings


async def run_researchers_in_parallel(
    question: str, model: Model | str, *, max_turns: int, timing: TimingReport
) -> list[LanguageFindings]:
    """Launch all three Researchers concurrently via ``asyncio.gather`` and await them all.

    This is the actual parallelism (requirement 4: implemented with asyncio):
    all three ``Runner.run`` coroutines are scheduled together, so their
    network-bound LLM calls overlap in wall-clock time instead of running
    one after another.
    """
    results = await asyncio.gather(
        *(
            _run_researcher(language, question, model, max_turns=max_turns, timing=timing)
            for language in LANGUAGES
        )
    )
    return list(results)


async def run_researchers_sequentially(
    question: str, model: Model | str, *, max_turns: int, timing: TimingReport
) -> list[LanguageFindings]:
    """Run the same three Researchers one after another (for latency comparison only).

    This function exists purely to measure sequential latency as a baseline;
    the actual pipeline (``run_parallelization``) always uses
    ``run_researchers_in_parallel``, never this function.
    """
    results: list[LanguageFindings] = []
    for language in LANGUAGES:
        results.append(
            await _run_researcher(language, question, model, max_turns=max_turns, timing=timing)
        )
    return results


async def run_parallelization(
    question: str, model: Model | str, *, max_turns: int = 6
) -> ParallelizationResult:
    """Run the three Researchers concurrently, then the Synthesizer once.

    Control flow is fixed by this function, not by any LLM: exactly three
    Researchers always run (in parallel), and exactly one Synthesizer always
    runs afterward, over all three results. No step decides whether another
    step runs.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    timing = TimingReport()
    wall_clock_start = time.perf_counter()

    findings = await run_researchers_in_parallel(question, model, max_turns=max_turns, timing=timing)

    # Step 2 -- Synthesizer: all three findings -> final answer (runs only after
    # every Researcher has completed; asyncio.gather's join guarantees this).
    started_at = time.perf_counter()
    try:
        synthesizer = build_synthesizer(model)
        synthesizer_input = f"Original question: {question}\n\nFindings:\n" + "\n\n".join(
            f"### {f.language}\n{f.findings}"
            + (f"\nSources: {', '.join(f.sources)}" if f.sources else "")
            for f in findings
        )
        synthesizer_result = await Runner.run(
            synthesizer,
            synthesizer_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        if not isinstance(synthesizer_result.final_output, str):
            raise TypeError("Synthesizer must return a plain-text final answer.")
        final_answer = synthesizer_result.final_output
    except Exception as exc:  # noqa: BLE001
        raise ParallelizationStepError("synthesizer", exc) from exc
    finally:
        timing.add(StepTiming("Synthesizer", started_at, time.perf_counter()))

    timing.wall_clock_seconds = time.perf_counter() - wall_clock_start

    return ParallelizationResult(
        question=question, findings=findings, final_answer=final_answer, timing=timing
    )


def run_parallelization_sync(
    question: str, model: Model | str, *, max_turns: int = 6
) -> ParallelizationResult:
    """Synchronous convenience wrapper around ``run_parallelization`` for CLI/tests."""
    return asyncio.run(run_parallelization(question, model, max_turns=max_turns))


async def measure_sequential_vs_parallel(
    question: str, model: Model | str, *, max_turns: int = 6
) -> tuple[TimingReport, TimingReport]:
    """Run the three Researchers once sequentially and once in parallel, and return both timings.

    This is only for measurement/comparison (see docs/03-PARALLELIZATION.md);
    it runs six Researcher calls in total (three sequential + three parallel),
    so it costs roughly double a single normal run.
    """
    sequential_timing = TimingReport()
    seq_start = time.perf_counter()
    await run_researchers_sequentially(question, model, max_turns=max_turns, timing=sequential_timing)
    sequential_timing.wall_clock_seconds = time.perf_counter() - seq_start

    parallel_timing = TimingReport()
    par_start = time.perf_counter()
    await run_researchers_in_parallel(question, model, max_turns=max_turns, timing=parallel_timing)
    parallel_timing.wall_clock_seconds = time.perf_counter() - par_start

    return sequential_timing, parallel_timing


def main() -> None:
    """CLI entry point: run the pipeline once and print findings, final answer, and timing."""
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
        result = run_parallelization_sync(question, model)
    except ParallelizationStepError as exc:
        raise SystemExit(f"Pipeline failed at step: {exc.step_name}\n{exc}") from exc

    print("=" * 80)
    print("RESEARCHER FINDINGS (ran concurrently)")
    print("=" * 80)
    for f in result.findings:
        print(f"\n### {f.language}\n{f.findings}")
        if f.sources:
            print("Sources: " + ", ".join(f.sources))

    print("\n" + "=" * 80)
    print("SYNTHESIZER: final answer")
    print("=" * 80)
    print(result.final_answer)

    print("\n" + "=" * 80)
    print("TIMING (start/end per agent, seconds since process start)")
    print("=" * 80)
    for line in result.timing.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
