"""Orchestrator-Workers, expressed as a LangGraph ``StateGraph`` using the
dynamic fan-out (``Send``) API.

Anthropic pattern: a central LLM (the Orchestrator) dynamically breaks a task
into an *unknown-at-build-time* number of subtasks, dispatches a Worker for
each one (they run concurrently), and a Synthesizer aggregates all the
Worker results. This is the key difference from Parallelization: there the
subtasks are fixed in advance (three languages); here the Orchestrator
decides both *how many* subtasks exist and *what* each one investigates, at
run time.

Graph:

    START -> orchestrator -> assign_workers (routing fn, returns Send(...) per task)
               |                    |
               |                    v
               |                worker  (one invocation per task, all concurrent)
               |                    |
               `--------------------+--> synthesizer --> END

``assign_workers`` is a Conditional Edge whose routing function returns a
*list* of ``langgraph.types.Send`` objects -- one per task the Orchestrator
produced -- instead of a single fixed node name. This is LangGraph's native
way to fan out to a dynamic number of Worker invocations.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Callable

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

MAX_WORKERS = 5

TextLLMCall = Callable[[str, str], str]
SearchWebCall = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class WorkerTask(TypedDict):
    task_id: str
    objective: str


class WorkerResult(TypedDict):
    task_id: str
    summary: str


class OrchestratorWorkersState(TypedDict):
    question: str
    tasks: list[WorkerTask]
    # Reducer required: every dynamically-spawned `worker` invocation writes
    # this field concurrently; without a Reducer, only one Worker's result
    # would survive the merge.
    worker_results: Annotated[list[WorkerResult], operator.add]
    final_answer: str


def initial_state(question: str) -> OrchestratorWorkersState:
    return {"question": question, "tasks": [], "worker_results": [], "final_answer": ""}


# A single Worker invocation (via Send) only needs to see its own task -- it
# has no visibility into the other tasks or their results.
class WorkerInput(TypedDict):
    task: WorkerTask


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_INSTRUCTIONS = f"""You are the Orchestrator. Given the user's question, break
it into a JSON array of 1 to {MAX_WORKERS} self-contained research tasks, each with
"task_id" and "objective". Each objective must be understandable in isolation -- the
Worker who receives it will NOT see the other tasks. Return ONLY the JSON array."""

WORKER_INSTRUCTIONS = """You are a Worker. You are given exactly one self-contained research
task. Investigate it and write a concise, factual summary addressing its objective."""

SYNTHESIZER_INSTRUCTIONS = """You are the Synthesizer. You receive the results of every
Worker task the Orchestrator dispatched. Combine them into one coherent final answer to
the original question."""


def _parse_tasks(raw_text: str) -> list[WorkerTask]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Orchestrator did not return valid JSON: {raw_text!r}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Orchestrator must return a non-empty JSON array of tasks.")
    if len(parsed) > MAX_WORKERS:
        # Hard, code-enforced cap -- never trust the model/prompt alone.
        raise ValueError(f"Orchestrator produced {len(parsed)} tasks, exceeding the cap of {MAX_WORKERS}.")
    tasks: list[WorkerTask] = []
    for item in parsed:
        tasks.append({"task_id": item["task_id"], "objective": item["objective"]})
    return tasks


def _format_worker_results(results: list[WorkerResult]) -> str:
    return "\n\n".join(f"### {r['task_id']}\n{r['summary']}" for r in results)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_orchestrator(llm_call: TextLLMCall):
    """Owns ``tasks`` only. Decides, dynamically, how many Workers will run."""

    def orchestrator(state: OrchestratorWorkersState) -> OrchestratorWorkersState:
        tasks = _parse_tasks(llm_call(ORCHESTRATOR_INSTRUCTIONS, state["question"]))
        return {"tasks": tasks}  # type: ignore[typeddict-item]

    return orchestrator


def make_worker(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    """One Worker invocation per task (dispatched dynamically via ``Send``).
    Owns (appends to) ``worker_results`` only, and sees only its own task."""

    def worker(state: WorkerInput) -> OrchestratorWorkersState:
        task = state["task"]
        search_result = search_web_call(task["objective"])
        summary = llm_call(
            WORKER_INSTRUCTIONS,
            f"Task: {task['objective']}\nSearch results: {search_result.get('results', [])}",
        )
        return {"worker_results": [{"task_id": task["task_id"], "summary": summary}]}  # type: ignore[typeddict-item]

    return worker


def make_synthesizer(llm_call: TextLLMCall):
    def synthesizer(state: OrchestratorWorkersState) -> OrchestratorWorkersState:
        user_prompt = (
            f"Original question: {state['question']}\n\n"
            f"Worker results:\n{_format_worker_results(state['worker_results'])}"
        )
        return {"final_answer": llm_call(SYNTHESIZER_INSTRUCTIONS, user_prompt)}  # type: ignore[typeddict-item]

    return synthesizer


# ---------------------------------------------------------------------------
# Routing function (NOT a Node): fans out to exactly ``len(tasks)`` Worker
# invocations, one ``Send`` per task -- a dynamic Conditional Edge.
# ---------------------------------------------------------------------------


def assign_workers(state: OrchestratorWorkersState) -> list[Any]:
    from langgraph.types import Send

    return [Send("worker", {"task": task}) for task in state["tasks"]]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(llm_call: TextLLMCall, search_web_call: SearchWebCall = _web_search):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(OrchestratorWorkersState)
    graph_builder.add_node("orchestrator", make_orchestrator(llm_call))
    graph_builder.add_node("worker", make_worker(llm_call, search_web_call))
    graph_builder.add_node("synthesizer", make_synthesizer(llm_call))

    graph_builder.add_edge(START, "orchestrator")
    graph_builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    graph_builder.add_edge("worker", "synthesizer")
    graph_builder.add_edge("synthesizer", END)

    return graph_builder.compile()


def get_mermaid(graph) -> str:
    return graph.get_graph().draw_mermaid()
