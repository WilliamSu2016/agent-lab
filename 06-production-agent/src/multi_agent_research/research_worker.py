"""Research Worker: one isolated Specialist Research Agent per task.

Fanned out by ``planner.fan_out_to_workers`` via LangGraph's ``Send`` API
(one concurrent instance of this node per planned task, all running in the
same superstep -- LangGraph's own Pregel scheduler parallelizes them; see
experiment 4's doc for the empirical proof this relies on).

Ownership: this node only ever returns ``worker_results`` (one single-item
list, merged via the ``operator.add`` reducer) plus its own ``trace`` entry.
It never sees another worker's task or result -- its input (``WorkerInput``)
is only ever the one task it was dispatched with.

Reliability: every call is bounded by ``per_worker_timeout_seconds`` and
every exception (timeout or otherwise) is caught here and turned into a
``WorkerResult`` with ``status="timeout"``/``"failed"`` -- this node never
raises. That matters because an uncaught exception in a single fan-out
branch crashes the whole ``graph.invoke()`` call for every other in-flight
worker too (verified in experiment 4).

The timeout enforcement itself now delegates to the single canonical
implementation in ``src/reliability/timeout.py`` (``ToolTimeoutError`` is a
``TimeoutError`` subclass, so the ``except TimeoutError`` handling below is
unaffected) instead of this module's own copy of the daemon-thread helper.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from src.multi_agent_research.state import MultiAgentResearchState, ResearchTask, WorkerResult
from src.reliability.timeout import run_with_timeout
from src.specialists.llm import TextLLMCall

RESEARCH_INSTRUCTIONS_TEMPLATE = """你是一名研究者（Research Worker），只负责调查一个具体的
研究方面，不需要、也不应该关心整体问题下的其他方面（有其他独立的研究者在负责）。

总体研究问题（仅供你理解背景）：{question}
你被分配的研究方面：{aspect}
（背景：这个任务的产生原因是 "{reason}"）

请只针对"{aspect}"这一个方面给出研究结论：关键事实、数据、原理、优缺点或适用场景，
并尽量说明证据来源/依据（例如"根据官方文档"、"根据社区实践"等，即使无法给出精确引用）。
不要泛泛而谈整体问题，也不要涉及其他方面。如果信息不足，明确说明缺口，不要编造。"""


class WorkerInput(TypedDict):
    """The ONLY thing a research_worker instance ever receives -- one task,
    never the full task list or another worker's data."""

    question: str
    task: ResearchTask
    per_worker_timeout_seconds: float


def make_worker_node(text_llm_call: TextLLMCall):
    """Node: Research Worker (fanned out). Owns ONLY ``worker_results``."""

    def research_worker(state: WorkerInput) -> MultiAgentResearchState:
        task = state["task"]
        timeout_seconds = state["per_worker_timeout_seconds"]

        def call_llm() -> str:
            instructions = RESEARCH_INSTRUCTIONS_TEMPLATE.format(
                question=state["question"], aspect=task["aspect"], reason=task["reason"]
            )
            return text_llm_call(instructions, task["aspect"])

        result: WorkerResult
        try:
            findings = run_with_timeout(call_llm, timeout_seconds, tool_name=f"research_worker[{task['task_id']}]")
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "completed",
                "findings": findings,
                "error": None,
            }
        except TimeoutError:
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "timeout",
                "findings": "",
                "error": f"Worker exceeded the {timeout_seconds}s timeout for this task.",
            }
        except Exception as exc:  # noqa: BLE001 -- any single worker failure must be recoverable
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "failed",
                "findings": "",
                "error": str(exc),
            }

        update: MultiAgentResearchState = {  # type: ignore[typeddict-item]
            "worker_results": [result],
            "trace": [f"ResearchWorker[{task['task_id']}]: status={result['status']} aspect={task['aspect']!r}"],
        }
        return update

    return research_worker
