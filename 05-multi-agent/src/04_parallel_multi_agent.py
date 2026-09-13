"""Experiment 4 (Multi-Agent): Parallel Multi-Agent Research, built on LangGraph.

Task: "全面研究 AI Agent Framework." A Planner dynamically decides *what*
needs to be researched (it is not hard-coded which frameworks/aspects show
up -- see ``PLANNER_INSTRUCTIONS`` and the tests, which use a fake planner
call that returns an arbitrary, made-up set of aspects to prove nothing is
hard-wired). One Research Agent is spun up per aspect; all Research Agents
run in parallel as LangGraph fan-out branches over the *same* ``worker``
node (each invocation gets only its own task, never another worker's task or
result); a Synthesizer then fans back in over every ``ResearchResult`` to
produce the final report.

Graph:

    START
      |
      v
    planner  (LLM decides the research aspects -- dynamic, capped at MAX_WORKERS)
      |
      v (conditional edge returns one langgraph.types.Send per task -- fan-out)
    worker (x N, running concurrently; each is its own isolated Research Agent)
      |
      v (LangGraph waits for every dispatched `worker` instance -- fan-in)
    synthesizer
      |
      v
     END

Design constraints:

* ``StateGraph`` + explicit state + Nodes + a conditional edge that returns
  ``list[Send]`` for dynamic fan-out (Requirement 1, 7, 8). No hand-written
  ``asyncio.gather``/thread pool orchestrates the workers -- LangGraph's own
  Pregel scheduler runs every dispatched ``worker`` task in the same
  superstep, concurrently, and only proceeds to ``synthesizer`` once every
  one of them has finished or failed (verified empirically: four workers
  each sleeping 1 second finish in ~1 second total, not ~4 -- see
  ``docs/12-PARALLEL-MULTI-AGENT.md``).
* Every worker instance receives *only* its own ``ResearchTask`` (via the
  ``Send`` payload) -- never the full task list, never another worker's
  in-flight or completed result (Requirement 4).
* ``results`` uses an ``operator.add`` reducer so results merge safely across
  concurrent branches without a race (Requirement 5 & 8's fan-in half).
* ``MAX_WORKERS`` hard-caps how many ``Send`` payloads the fan-out edge ever
  emits, no matter how many aspects the Planner's LLM call proposes
  (Requirement 9).
* Each worker enforces its own wall-clock deadline
  (``per_worker_timeout_seconds``) around its own LLM call using a bounded
  thread executor, and never lets an exception escape the node -- both
  timeouts and ordinary failures are caught and turned into a ``"failed"``/
  ``"timeout"`` ``ResearchResult`` instead of raising (Requirement 11). This
  matters because an *uncaught* exception in a single fan-out branch crashes
  the *entire* ``graph.invoke()`` call for every other in-flight worker too
  (verified empirically -- see ``docs/12-PARALLEL-MULTI-AGENT.md``).
* ``run_parallel_research`` additionally wraps the whole graph run in a
  hard, overall wall-clock timeout (``max_total_seconds``, Requirement 10),
  independent of the per-worker timeouts, so a hang anywhere in the pipeline
  (e.g. the Synthesizer's own call) cannot block the caller forever either.
* No MCP, no LangGraph memory/checkpointer (Requirement 12 & 13): state lives
  only for the duration of a single ``graph.invoke()`` call, nothing is
  persisted across runs.

See ``docs/12-PARALLEL-MULTI-AGENT.md`` for the full walkthrough, including
the requested comparison against plain Parallelization and the "why is this
Multi-Agent" discussion.
"""

from __future__ import annotations

import json
import operator
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import Annotated, TypedDict

from src.specialists.llm import TextLLMCall

GRAPH_NAME = "Parallel Multi-Agent Research"

# Requirement 9: hard cap on how many Research Agents can ever be spawned for
# one run, regardless of how many aspects the Planner's LLM proposes.
MAX_WORKERS = 6

# Requirement 10: default wall-clock budgets. Both are independent knobs:
# a slow single worker cannot silently eat the whole budget (each worker is
# bounded on its own), and a hang anywhere in the pipeline cannot block the
# caller past MAX_TOTAL_SECONDS either.
MAX_TOTAL_SECONDS = 90.0
MAX_SECONDS_PER_WORKER = 30.0


PLANNER_INSTRUCTIONS = """你是一个研究任务规划器（Planner）。给定一个宽泛的研究主题，
你需要把它拆解成若干个具体、互不重叠的"研究方面"（aspect），每个方面都足够具体，
可以交给一个独立的研究者去单独调查，不需要了解其他方面的内容。

不要假设任何固定的候选清单——具体要拆出哪些方面，完全取决于这个主题本身。

只输出一个 JSON 数组，数组元素是字符串，每个字符串是一个研究方面的简短描述
（不超过 15 个字）。不要输出任何其他文字、解释或 Markdown 代码块标记，只输出
JSON 数组本身，例如：

["方面一", "方面二", "方面三"]
"""

RESEARCH_INSTRUCTIONS_TEMPLATE = """你是一名研究者，只负责调查一个具体的研究方面，
不需要、也不应该关心整体主题下的其他方面（有其他独立的研究者在负责）。

总体研究主题（仅供你理解背景）：{topic}
你被分配的研究方面：{aspect}

请只针对"{aspect}"这一个方面给出研究结论：关键事实、原理、优缺点或适用场景。
不要泛泛而谈整体主题，也不要涉及其他方面。如果信息不足，明确说明缺口，不要编造。"""

SYNTHESIZER_INSTRUCTIONS = """你是汇总者（Synthesizer），会收到多个独立研究者对同一个
主题下不同方面的研究结果（其中可能有个别方面因超时或失败而没有结论）。

请把这些结果整合成一份连贯的研究报告：
1. 先给出总体概述。
2. 按方面分节陈述关键发现。
3. 如果有方面缺失或失败，在报告中明确指出这是"未能完成的研究项"，不要假装它存在
   结论，也不要替它编造内容。
4. 最后给出一段总结性的结论。"""


def _run_with_timeout(func, timeout_seconds: float):
    """Run ``func()`` bounded by ``timeout_seconds`` using a daemon thread.

    This deliberately avoids ``concurrent.futures.ThreadPoolExecutor``: that
    module keeps a *global* registry of every worker thread it has ever
    created and registers an ``atexit`` hook (``concurrent.futures.thread.
    _python_exit``) that joins *all* of them before the interpreter is
    allowed to exit -- even threads that belong to an executor instance you
    already called ``shutdown(wait=False)`` on. In practice this means a
    single hung/slow call (e.g. a real network request past its timeout)
    still blocks the whole *process* from exiting, only surfacing much later
    than the timeout actually enforced by ``future.result(timeout=...)``
    (observed live: worker timeouts fired correctly at ~30s each, but the
    overall script still took several minutes to actually exit). A plain
    ``threading.Thread(daemon=True)`` has no such global join-at-exit hook:
    if it is still running when ``timeout_seconds`` elapses, we raise
    ``TimeoutError`` immediately and the abandoned thread is simply killed by
    the OS whenever the process eventually exits, without blocking on it.
    """

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = func()
        except Exception as exc:  # noqa: BLE001 -- re-raised on the caller's side
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"Call exceeded the {timeout_seconds}s timeout.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


# ---------------------------------------------------------------------------
# Structured task / result types (Requirement 5: every worker returns a
# structured ResearchResult, not just raw text buried in a message blob).
# ---------------------------------------------------------------------------


class ResearchTask(TypedDict):
    """One dynamically generated unit of work -- exactly what one worker sees."""

    task_id: str
    aspect: str


ResearchStatus = Literal["completed", "failed", "timeout"]


class ResearchResult(TypedDict):
    """The structured result of exactly one worker's research on its own aspect."""

    task_id: str
    aspect: str
    status: ResearchStatus
    summary: str
    error: str | None


def _add_results(left: list[ResearchResult], right: list[ResearchResult]) -> list[ResearchResult]:
    """Reducer for concurrent fan-in: append, never overwrite (Requirement 8)."""
    return [*left, *right]


class ParallelResearchState(TypedDict):
    """Top-level graph state. Every field here is scoped to a single run --
    no memory, no persistence across runs (Requirement 13)."""

    topic: str
    max_workers: int
    per_worker_timeout_seconds: float
    tasks: list[ResearchTask]
    results: Annotated[list[ResearchResult], _add_results]
    final_report: str


class WorkerInput(TypedDict):
    """What one ``worker`` branch actually receives via ``Send`` -- Requirement 4:
    only its own task, nothing about the other tasks or their results."""

    topic: str
    task: ResearchTask
    per_worker_timeout_seconds: float


def initial_state(
    topic: str,
    *,
    max_workers: int = MAX_WORKERS,
    per_worker_timeout_seconds: float = MAX_SECONDS_PER_WORKER,
) -> ParallelResearchState:
    if not topic.strip():
        raise ValueError("topic must not be empty")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    return {
        "topic": topic,
        "max_workers": max_workers,
        "per_worker_timeout_seconds": per_worker_timeout_seconds,
        "tasks": [],
        "results": [],
        "final_report": "",
    }


# ---------------------------------------------------------------------------
# Planner node -- dynamically decides the research tasks (Requirement 1).
# ---------------------------------------------------------------------------


def _parse_aspects(raw: str) -> list[str]:
    """Parse the Planner LLM's response into a flat list of aspect strings.

    Tries strict JSON first (the documented contract); falls back to
    splitting on newlines/bullets so a slightly malformed response does not
    crash the whole pipeline outright.
    """
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    lines = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.、） )").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def make_planner_node(text_llm_call: TextLLMCall):
    """Node 1: ask the LLM what should be researched, then cap and tag tasks.

    Nothing about *which* aspects come out is hard-coded here -- this
    function only enforces the MAX_WORKERS cap (Requirement 9) and assigns
    task ids. The actual breakdown is entirely up to ``text_llm_call``.
    """

    def planner(state: ParallelResearchState) -> ParallelResearchState:
        raw = text_llm_call(PLANNER_INSTRUCTIONS, state["topic"])
        aspects = _parse_aspects(raw)
        if not aspects:
            raise ValueError("Planner produced no research aspects; cannot continue.")

        capped = aspects[: state["max_workers"]]
        tasks: list[ResearchTask] = [
            {"task_id": f"task-{index}-{uuid.uuid4().hex[:8]}", "aspect": aspect}
            for index, aspect in enumerate(capped)
        ]

        update: ParallelResearchState = {"tasks": tasks}  # type: ignore[typeddict-item]
        return update

    return planner


def fan_out_to_workers(state: ParallelResearchState):
    """Conditional edge (NOT a Node): turn each planned task into one ``Send``.

    This is the dynamic fan-out (Requirement 1, 2, 7, 8): the number and
    content of the ``Send`` calls depends entirely on what the Planner
    produced at run time -- nothing here is a fixed list of worker names.
    Each ``Send`` payload is a ``WorkerInput`` containing only that one task,
    never the full task list (Requirement 4).
    """
    from langgraph.types import Send

    return [
        Send(
            "worker",
            {
                "topic": state["topic"],
                "task": task,
                "per_worker_timeout_seconds": state["per_worker_timeout_seconds"],
            },
        )
        for task in state["tasks"]
    ]


# ---------------------------------------------------------------------------
# Worker node -- one isolated Research Agent per task (Requirement 2, 3, 4).
# ---------------------------------------------------------------------------


def make_worker_node(text_llm_call: TextLLMCall):
    """Node 2 (fanned out): research exactly one task, in isolation.

    Requirement 11: a bounded thread executor enforces
    ``per_worker_timeout_seconds`` around the LLM call, and *any* exception
    (timeout or otherwise) is caught here and converted into a ``ResearchResult``
    with ``status="failed"``/``"timeout"`` -- this node never raises. That
    matters because letting an exception escape a single fan-out branch
    crashes ``graph.invoke()`` for every other still-running worker too (see
    ``docs/12-PARALLEL-MULTI-AGENT.md`` for the empirical proof).
    """

    def worker(state: WorkerInput) -> ParallelResearchState:
        task = state["task"]
        timeout_seconds = state["per_worker_timeout_seconds"]

        def call_llm() -> str:
            instructions = RESEARCH_INSTRUCTIONS_TEMPLATE.format(topic=state["topic"], aspect=task["aspect"])
            return text_llm_call(instructions, task["aspect"])

        result: ResearchResult
        # See _run_with_timeout's docstring: a plain daemon thread is used
        # here instead of ThreadPoolExecutor so an abandoned, still-running
        # call cannot block process exit later.
        try:
            summary = _run_with_timeout(call_llm, timeout_seconds)
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "completed",
                "summary": summary,
                "error": None,
            }
        except TimeoutError:
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "timeout",
                "summary": "",
                "error": f"Worker exceeded the {timeout_seconds}s timeout for this task.",
            }
        except Exception as exc:  # noqa: BLE001 -- any single worker failure must be recoverable
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "failed",
                "summary": "",
                "error": str(exc),
            }

        update: ParallelResearchState = {"results": [result]}  # type: ignore[typeddict-item]
        return update

    return worker


# ---------------------------------------------------------------------------
# Synthesizer node -- fans back in over every ResearchResult (Requirement 6).
# ---------------------------------------------------------------------------


def _format_results(results: list[ResearchResult]) -> str:
    sections = []
    for result in results:
        if result["status"] == "completed":
            sections.append(f"### {result['aspect']}\n{result['summary']}")
        else:
            sections.append(f"### {result['aspect']}\n[{result['status'].upper()}] {result['error']}")
    return "\n\n".join(sections) if sections else "(no research results were produced)"


def make_synthesizer_node(text_llm_call: TextLLMCall):
    """Node 3: combine every worker's ResearchResult into one final report.

    Runs exactly once, after LangGraph's own fan-in has collected every
    dispatched worker's contribution into ``state["results"]`` (Requirement 6).
    """

    def synthesizer(state: ParallelResearchState) -> ParallelResearchState:
        user_prompt = f"研究主题：{state['topic']}\n\n各方面的研究结果：\n{_format_results(state['results'])}"
        report = text_llm_call(SYNTHESIZER_INSTRUCTIONS, user_prompt)

        update: ParallelResearchState = {"final_report": report}  # type: ignore[typeddict-item]
        return update

    return synthesizer


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(planner_llm_call: TextLLMCall, worker_llm_call: TextLLMCall, synthesizer_llm_call: TextLLMCall):
    """Compile the graph: START -> planner -> (fan-out) -> worker* -> synthesizer -> END.

    Three separate ``TextLLMCall``s are accepted (rather than one shared
    call) purely so tests can script the Planner, every Worker, and the
    Synthesizer independently; in ``main()`` all three are the same real
    model call.
    """
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ParallelResearchState)

    graph_builder.add_node("planner", make_planner_node(planner_llm_call))
    graph_builder.add_node("worker", make_worker_node(worker_llm_call))
    graph_builder.add_node("synthesizer", make_synthesizer_node(synthesizer_llm_call))

    graph_builder.add_edge(START, "planner")
    graph_builder.add_conditional_edges("planner", fan_out_to_workers, ["worker"])
    graph_builder.add_edge("worker", "synthesizer")
    graph_builder.add_edge("synthesizer", END)

    return graph_builder.compile()


def print_mermaid_diagram(graph) -> str:
    """Return (and print) the compiled graph's Mermaid diagram source."""
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)
    return mermaid


# ---------------------------------------------------------------------------
# Top-level run helper -- overall wall-clock budget (Requirement 10).
# ---------------------------------------------------------------------------


class ParallelResearchTimeoutError(TimeoutError):
    """Raised when the whole pipeline exceeds ``max_total_seconds``."""


@dataclass(frozen=True)
class ParallelResearchOutcome:
    topic: str
    tasks: list[ResearchTask]
    results: list[ResearchResult]
    final_report: str


def run_parallel_research(
    topic: str,
    planner_llm_call: TextLLMCall,
    worker_llm_call: TextLLMCall,
    synthesizer_llm_call: TextLLMCall,
    *,
    max_workers: int = MAX_WORKERS,
    per_worker_timeout_seconds: float = MAX_SECONDS_PER_WORKER,
    max_total_seconds: float = MAX_TOTAL_SECONDS,
) -> ParallelResearchOutcome:
    """Run the full graph once, enforcing an overall wall-clock deadline.

    ``max_total_seconds`` is independent of ``per_worker_timeout_seconds``:
    the latter bounds each individual worker (Requirement 11), while this
    bounds the *entire* run (Planner + all workers + Synthesizer) so a hang
    anywhere in the pipeline cannot block the caller past this budget either
    (Requirement 10).
    """
    graph = build_graph(planner_llm_call, worker_llm_call, synthesizer_llm_call)
    state = initial_state(topic, max_workers=max_workers, per_worker_timeout_seconds=per_worker_timeout_seconds)

    # See _run_with_timeout's docstring: ThreadPoolExecutor-based timeouts
    # still block *process exit* on an abandoned thread even after
    # shutdown(wait=False); a daemon thread does not.
    def invoke() -> ParallelResearchState:
        return graph.invoke(state, {"recursion_limit": max_workers * 4 + 10})

    try:
        result = _run_with_timeout(invoke, max_total_seconds)
    except TimeoutError as exc:
        raise ParallelResearchTimeoutError(
            f"Parallel research exceeded the overall {max_total_seconds}s budget."
        ) from exc

    return ParallelResearchOutcome(
        topic=result["topic"],
        tasks=result["tasks"],
        results=result["results"],
        final_report=result["final_report"],
    )


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    topic = " ".join(sys.argv[1:]).strip() or "全面研究 AI Agent Framework。"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)

    print("=" * 80)
    print("GRAPH (Mermaid)")
    print("=" * 80)
    print_mermaid_diagram(build_graph(text_llm_call, text_llm_call, text_llm_call))

    try:
        outcome = run_parallel_research(topic, text_llm_call, text_llm_call, text_llm_call)
    except ParallelResearchTimeoutError as exc:
        raise SystemExit(str(exc)) from exc

    print("=" * 80)
    print(f"TOPIC: {topic}")
    print("=" * 80)
    print(f"Dynamically planned {len(outcome.tasks)} research task(s):")
    for task in outcome.tasks:
        print(f"  - {task['aspect']}")

    print("\n" + "=" * 80)
    print("RESULTS (ran in parallel)")
    print("=" * 80)
    for result in outcome.results:
        print(f"\n### [{result['status']}] {result['aspect']}")
        print(result["summary"] or result["error"])

    print("\n" + "=" * 80)
    print("FINAL REPORT (Synthesizer)")
    print("=" * 80)
    print(outcome.final_report)


if __name__ == "__main__":
    main()
