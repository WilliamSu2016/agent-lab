"""Orchestrator-Workers pattern: a central LLM dynamically decomposes the question
into a bounded set of research tasks, dispatches one Worker per task, and a
Synthesizer combines all Worker results into the final answer.

This module implements the *Orchestrator-Workers* workflow from Anthropic's
"Building effective agents" (see ``docs/PATTERN-TAXONOMY.md``): unlike
Parallelization (``src/parallelization.py``), the number and content of the
sub-tasks are NOT fixed ahead of time in this file -- they are decided at run
time by the Orchestrator LLM, based on whatever question the user actually
asks. This module deliberately never hardcodes any particular research
topics (e.g. "Python"/"TypeScript"/"Go"); the Orchestrator must derive tasks
from the input question itself.

Design constraints (all enforced by this module's own code, not by the SDK):

* The Orchestrator's ONLY job is to read the question and produce a
  structured ``ResearchPlan`` (a list of ``WorkerTask``s) -- it has no tools,
  never does research itself, and never decides more than that plan.
* The Orchestrator's plan is a *hint*, not a guarantee, about how many tasks
  there are. This module's own code (``_assert_within_worker_cap``) enforces
  a hard ceiling of ``MAX_WORKERS`` regardless of what the model returns --
  see docs/04-ORCHESTRATOR-WORKERS.md Q5 for why this cannot be left to the
  model, the prompt, or even the output schema alone.
* Each Worker is built from one single, generic template (``build_worker``);
  there is no per-topic Worker like Parallelization's per-language
  Researchers, because the topics themselves are not known until the
  Orchestrator produces them at run time.
* A Worker only ever sees its own assigned ``WorkerTask`` (plus the original
  question, for background context) -- never another Worker's task or
  results. Workers are independent of each other, same as in Parallelization,
  but *which* tasks they are given is decided dynamically here, not fixed in
  this file.
* Workers are dispatched via ``asyncio.gather`` for efficiency, but running
  them concurrently is an *implementation detail*, not the defining feature
  of this pattern (see docs/04-ORCHESTRATOR-WORKERS.md Q1) -- they could just
  as well be run one at a time without changing what pattern this is.
* No Handoffs (``agents.handoff``/``Agent.handoffs`` is never used anywhere)
  and no Agent-as-a-tool wiring of Workers into the Orchestrator's own tool
  loop: the Orchestrator produces a plan and stops; this module's own Python
  code (not any LLM's tool-calling decision) is what actually dispatches the
  Workers and calls the Synthesizer. This keeps the "how many workers, with
  what tasks, and when do they run" logic fully explicit and auditable,
  rather than hidden inside a single agentic loop.
* Exactly one Synthesizer runs, after all Workers have completed, and
  produces the final answer -- no critique/revise loop (that would be
  Evaluator-Optimizer, not used here).
* ``search_web`` (a thin wrapper around ``src.tools.web_search``) is attached
  to the generic Worker, never to the Orchestrator or the Synthesizer.
* Every failure is re-raised as an ``OrchestratorWorkersStepError`` naming
  exactly which step failed ("orchestrator", "worker-<task_id>", or
  "synthesizer").

What the OpenAI Agents SDK still owns: each LLM call, structured output
parsing/validation, each Worker's own ``search_web`` tool-call loop, and
tracing. This module only decides how the Orchestrator's plan is turned into
actual Worker invocations, and enforces the hard cap on how many of those can
ever run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.tools import web_search as _web_search

PIPELINE_NAME = "Research orchestrator-workers"

# Hard, code-enforced ceiling on how many Workers may ever be dispatched for a
# single question, independent of what the Orchestrator's plan asks for.
MAX_WORKERS = 5


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


class OrchestratorWorkersStepError(RuntimeError):
    """Raised when the Orchestrator, a Worker, or the Synthesizer fails."""

    def __init__(self, step_name: str, cause: BaseException) -> None:
        super().__init__(f"Orchestrator-Workers failed at step '{step_name}': {cause}")
        self.step_name = step_name
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Structured Worker Task (Orchestrator's output) and Worker Result (Worker's output)
# ---------------------------------------------------------------------------


class WorkerTask(BaseModel):
    """One structured research task, fully self-contained for an independent Worker."""

    task_id: str = Field(description="A short, unique identifier for this task, e.g. 'task-1'.")
    objective: str = Field(
        description=(
            "One clear, self-contained research objective. A Worker will see ONLY this "
            "objective (plus optional guidance) and must be able to act on it without any "
            "other context about the rest of the plan."
        )
    )
    guidance: str = Field(
        default="",
        description=(
            "Optional notes for the Worker: what to focus on, what to avoid, or what kind "
            "of evidence/sources to prefer."
        ),
    )


class ResearchPlan(BaseModel):
    """Orchestrator step output: a dynamically sized, non-overlapping set of tasks.

    Intentionally has no ``max_length`` on ``tasks`` here -- the upper bound on
    worker count is enforced in code (``_assert_within_worker_cap``), not left
    to the output schema alone. See module docstring and
    docs/04-ORCHESTRATOR-WORKERS.md Q5.
    """

    tasks: list[WorkerTask] = Field(
        min_length=1,
        description=(
            f"The distinct, non-overlapping research tasks genuinely needed to answer the "
            f"question. Use as few or as many as the question actually requires, up to "
            f"{MAX_WORKERS} -- do not pad the list to reach a fixed number, and do not exceed "
            f"the limit."
        ),
    )


class WorkerFindings(BaseModel):
    """One Worker's structured findings for its single assigned task."""

    summary: str = Field(description="A concise, factual summary addressing the task's objective.")
    key_findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


@dataclass
class WorkerResult:
    """A Worker's result, paired with the task it was assigned.

    ``task_id``/``objective`` are attached by this module's own code (from the
    ``WorkerTask`` that was dispatched), not parsed from the model's own
    output -- so a Worker's result can never be silently misattributed to the
    wrong task due to the model mis-echoing an id.
    """

    task_id: str
    objective: str
    findings: WorkerFindings


@dataclass
class OrchestratorWorkersResult:
    """The full, inspectable trail of a single Orchestrator-Workers run."""

    question: str
    plan: ResearchPlan
    worker_results: list[WorkerResult]
    final_answer: str


# ---------------------------------------------------------------------------
# Hard worker-count cap -- enforced by this module's own code
# ---------------------------------------------------------------------------


def _assert_within_worker_cap(tasks: list[WorkerTask]) -> None:
    """Refuse to proceed if the Orchestrator's plan violates the hard worker cap.

    This check is independent of, and does not rely on, the Orchestrator's own
    instructions, the model's willingness to follow them, or any JSON-schema
    hint on ``ResearchPlan.tasks``. It is the one place that actually
    guarantees no more than ``MAX_WORKERS`` Workers are ever dispatched.
    """
    if len(tasks) > MAX_WORKERS:
        raise ValueError(
            f"Orchestrator produced {len(tasks)} tasks, exceeding the hard cap of "
            f"{MAX_WORKERS} workers. Refusing to dispatch any workers."
        )
    if len(tasks) < 1:
        raise ValueError("Orchestrator must produce at least one research task.")


# ---------------------------------------------------------------------------
# search_web tool -- attached only to the Worker
# ---------------------------------------------------------------------------


@function_tool
def search_web(query: str) -> dict[str, Any]:
    """Search the web for current, factual information relevant to one research task.

    Args:
        query: A focused web search query for a single research task.
    """
    return _web_search(query)


# ---------------------------------------------------------------------------
# Model factory (mirrors src/agent.py, src/prompt_chaining.py, src/routing.py,
# src/parallelization.py)
# ---------------------------------------------------------------------------


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


# ---------------------------------------------------------------------------
# Orchestrator -- question -> ResearchPlan (dynamic decomposition, no tools)
# ---------------------------------------------------------------------------

ORCHESTRATOR_INSTRUCTIONS = f"""You are the Orchestrator in a research system. You always run
first, and your ONLY job is to read the user's research question -- which can be about
ANYTHING, not just software or programming languages -- and break it down into a set of
distinct, non-overlapping research tasks that, together, would let someone answer the
question well.

Rules:
- Do NOT do any research yourself and do NOT answer the question -- you have no tools.
- Decide the number of tasks based entirely on what THIS question actually needs. Do not
  assume any fixed topic, category, or scope in advance (for example, do not assume the
  question is about programming languages, or default to any particular fixed list of
  technologies) -- derive the tasks from the question's actual content.
- Use as few or as many tasks as are genuinely necessary to cover the question, with a hard
  maximum of {MAX_WORKERS} tasks. Do not pad the list to reach a fixed number, and never exceed
  the maximum.
- Each task's objective must be self-contained: the worker who receives it will NOT see the
  other tasks, so write the objective so it can be understood and acted on in isolation.
- Make sure tasks do not overlap -- each task should cover a genuinely distinct angle.
"""


def build_orchestrator(model: Model | str) -> Agent:
    return Agent(
        name="Orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        model=model,
        output_type=ResearchPlan,
    )


# ---------------------------------------------------------------------------
# Worker -- one generic template, reused for however many tasks the
# Orchestrator produced. Content is entirely driven by the input message,
# never baked into the instructions (topics are not known ahead of time).
# ---------------------------------------------------------------------------

WORKER_INSTRUCTIONS = """You are a Research Worker. You execute exactly ONE assigned research
task for this turn, given to you as an "objective" (plus optional "guidance") in the input.
You have no visibility into the user's original question beyond the background text you are
given, and no visibility into any other worker's task or findings.

Rules:
- Use `search_web` to find current, factual information relevant to YOUR objective only.
- Do NOT expand scope beyond your assigned objective, and do NOT attempt to answer any
  broader question than the one you were given.
- Report a concise summary, a short list of key findings, and the source URLs you used.
- If search results are insufficient, say so explicitly instead of guessing.
"""


def build_worker(model: Model | str) -> Agent:
    return Agent(
        name="Worker",
        instructions=WORKER_INSTRUCTIONS,
        model=model,
        tools=[search_web],
        output_type=WorkerFindings,
    )


# ---------------------------------------------------------------------------
# Synthesizer -- runs once, after every Worker has completed
# ---------------------------------------------------------------------------

SYNTHESIZER_INSTRUCTIONS = """You are the Synthesizer. You run once, after every Worker
dispatched by the Orchestrator has completed its own independent research task. You receive
all of their findings together for the first time.

Your ONLY job: combine the workers' findings into one clear, well-organized final answer to
the user's original question, resolving overlaps and noting any gaps or contradictions
between workers.

Rules:
- Do NOT call any tools and do NOT introduce new facts beyond what the workers reported.
- Do NOT redo the research or second-guess what each worker found.
- Structure your answer around the question, not around a mechanical list of worker outputs.
- If the question calls for a recommendation or conclusion, end with one, explicitly justified.
- Respond in the same language as the user's original question.
"""


def build_synthesizer(model: Model | str) -> Agent:
    return Agent(
        name="Synthesizer",
        instructions=SYNTHESIZER_INSTRUCTIONS,
        model=model,
    )


# ---------------------------------------------------------------------------
# Dispatching Workers (concurrently, via asyncio -- an implementation detail)
# ---------------------------------------------------------------------------


async def _run_worker(
    task: WorkerTask, question: str, model: Model | str, *, max_turns: int
) -> WorkerResult:
    """Run one Worker for one task. Independent of every other Worker's call."""
    try:
        worker = build_worker(model)
        worker_input = (
            f"Original question (for background context only): {question}\n"
            f"Your assigned objective: {task.objective}\n"
        )
        if task.guidance:
            worker_input += f"Guidance: {task.guidance}\n"

        result = await Runner.run(
            worker,
            worker_input,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        findings = result.final_output_as(WorkerFindings, raise_if_incorrect_type=True)
    except Exception as exc:  # noqa: BLE001 -- re-raised with the failing task named
        raise OrchestratorWorkersStepError(f"worker-{task.task_id}", exc) from exc

    return WorkerResult(task_id=task.task_id, objective=task.objective, findings=findings)


def _format_worker_results(worker_results: list[WorkerResult]) -> str:
    """Render all Worker results as plain text for the Synthesizer's input."""
    sections: list[str] = []
    for result in worker_results:
        section = f"### Task {result.task_id}: {result.objective}\n{result.findings.summary}"
        if result.findings.key_findings:
            section += "\nKey findings:\n" + "\n".join(f"- {kf}" for kf in result.findings.key_findings)
        if result.findings.sources:
            section += "\nSources: " + ", ".join(result.findings.sources)
        sections.append(section)
    return "\n\n".join(sections)


async def run_orchestrator_workers(
    question: str, model: Model | str, *, max_turns: int = 6
) -> OrchestratorWorkersResult:
    """Run the fixed Orchestrator -> (dynamic N Workers, in parallel) -> Synthesizer flow.

    The *shape* of this control flow (plan, then dispatch, then synthesize) is
    fixed by this function; only *how many* Workers exist and *what* each one
    investigates is decided by the Orchestrator LLM, and even that is capped
    by ``_assert_within_worker_cap`` regardless of what the LLM produces.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    # Step 1 -- Orchestrator: question -> ResearchPlan (dynamic decomposition)
    try:
        orchestrator = build_orchestrator(model)
        orchestrator_result = await Runner.run(
            orchestrator,
            question,
            max_turns=max_turns,
            run_config=RunConfig(workflow_name=PIPELINE_NAME),
        )
        plan = orchestrator_result.final_output_as(ResearchPlan, raise_if_incorrect_type=True)
        _assert_within_worker_cap(plan.tasks)
    except Exception as exc:  # noqa: BLE001
        raise OrchestratorWorkersStepError("orchestrator", exc) from exc

    # Step 2 -- exactly len(plan.tasks) Workers (<= MAX_WORKERS), dispatched concurrently.
    worker_results = list(
        await asyncio.gather(
            *(
                _run_worker(task, question, model, max_turns=max_turns)
                for task in plan.tasks
            )
        )
    )

    # Step 3 -- Synthesizer: all worker results -> final answer (runs only after
    # every Worker has completed; asyncio.gather's join guarantees this).
    try:
        synthesizer = build_synthesizer(model)
        synthesizer_input = (
            f"Original question: {question}\n\nWorker findings:\n"
            + _format_worker_results(worker_results)
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
        raise OrchestratorWorkersStepError("synthesizer", exc) from exc

    return OrchestratorWorkersResult(
        question=question, plan=plan, worker_results=worker_results, final_answer=final_answer
    )


def run_orchestrator_workers_sync(
    question: str, model: Model | str, *, max_turns: int = 6
) -> OrchestratorWorkersResult:
    """Synchronous convenience wrapper around ``run_orchestrator_workers`` for CLI/tests."""
    return asyncio.run(run_orchestrator_workers(question, model, max_turns=max_turns))


def main() -> None:
    """CLI entry point: run the pipeline once and print the plan, worker results, and final answer."""
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    enable_local_tracing()

    question = " ".join(sys.argv[1:]).strip() or "全面分析 2026 年 AI Agent 开发技术栈。"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)

    try:
        result = run_orchestrator_workers_sync(question, model)
    except OrchestratorWorkersStepError as exc:
        raise SystemExit(f"Pipeline failed at step: {exc.step_name}\n{exc}") from exc

    print("=" * 80)
    print(f"ORCHESTRATOR PLAN ({len(result.plan.tasks)} task(s), cap={MAX_WORKERS})")
    print("=" * 80)
    for task in result.plan.tasks:
        print(f"- [{task.task_id}] {task.objective}")
        if task.guidance:
            print(f"    guidance: {task.guidance}")

    print("\n" + "=" * 80)
    print("WORKER RESULTS (dispatched concurrently)")
    print("=" * 80)
    print(_format_worker_results(result.worker_results))

    print("\n" + "=" * 80)
    print("SYNTHESIZER: final answer")
    print("=" * 80)
    print(result.final_answer)


if __name__ == "__main__":
    main()
