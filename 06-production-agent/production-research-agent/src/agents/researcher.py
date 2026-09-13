"""Researcher: one isolated Specialist Research Agent per fanned-out task.

Renamed from the original experiment's ``research_worker.py``. The key
change beyond the rename: every researcher invocation now makes a real
tool call (``search_web``, LOW risk) through the full three-layer
guardrail pipeline (``src.security.guardrails.secure_tool_call``) *before*
calling the LLM -- closing the Final Review's finding that this pipeline
never actually exercised ``src/security/tool_policy.py``'s risk tiers.

Reliability is unchanged from the original: every call is bounded by
``per_worker_timeout_seconds`` and every exception (timeout, guardrail
failure, or otherwise) is caught here and turned into a structured
``WorkerResult`` -- this node never raises, so one failed/blocked
researcher branch never crashes every other in-flight branch in the same
fan-out superstep.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from src.agents.common import llm_usage_delta
from src.agents.llm import TextLLMCall
from src.graph.state import ExecutionContextDict, ProductionResearchState, ResearchTask, WorkerResult
from src.observability.tracing import ExecutionContext, SpanKind, Tracer, bind_execution_context
from src.reliability.timeout import run_with_timeout
from src.security.authorization import (
    AuthorizationError,
    Identity,
)
from src.security.guardrails import (
    AgentWorkflowGuardrail,
    ToolCallRequest,
    ToolGuardrail,
)
from src.security.tool_policy import ToolArgumentValidationError

RESEARCH_INSTRUCTIONS_TEMPLATE = """你是一名研究者（Research Worker），只负责调查一个具体的
研究方面，不需要、也不应该关心整体问题下的其他方面（有其他独立的研究者在负责）。

总体研究问题（仅供你理解背景）：{question}
你被分配的研究方面：{aspect}
（背景：这个任务的产生原因是 "{reason}"）

以下是一次网络检索工具返回的参考信息（可能不完整或不相关，仅供参考，不要盲目照抄）：
{search_snippet}

请只针对"{aspect}"这一个方面给出研究结论：关键事实、数据、原理、优缺点或适用场景，
并尽量说明证据来源/依据。不要泛泛而谈整体问题，也不要涉及其他方面。如果信息不足，
明确说明缺口，不要编造。"""


class WorkerInput(TypedDict):
    question: str
    task: ResearchTask
    per_worker_timeout_seconds: float
    identity: dict
    context: ExecutionContextDict
    mode: str


def make_researcher_node(
    text_llm_call: TextLLMCall,
    *,
    search_tool_fn,
    workflow_guardrail: AgentWorkflowGuardrail,
    tool_guardrail: ToolGuardrail,
    tracer: Tracer,
):
    """Node: Researcher (fanned out). Owns ONLY ``worker_results``,
    ``tokens_used``, ``cost_usd``."""

    def researcher(state: WorkerInput) -> ProductionResearchState:
        task = state["task"]
        timeout_seconds = state["per_worker_timeout_seconds"]
        identity = Identity(
            user_id=state["identity"]["user_id"],
            tenant_id=state["identity"]["tenant_id"],
            roles=frozenset(state["identity"]["roles"]),
        )

        def do_research() -> tuple[str, int, float]:
            # This closure runs on a fresh raw thread (see
            # ``src.reliability.timeout.run_with_timeout``'s docstring),
            # which does NOT inherit the parent thread's ``contextvars`` --
            # so the ExecutionContext and parent Span must be reconstructed
            # explicitly and passed into every ``tracer.span(...)`` call
            # here (see ``ExecutionContextDict``'s docstring in
            # ``src.graph.state``).
            exec_context = ExecutionContext(**state["context"])
            trace_root = tracer.get_trace(exec_context.trace_id)
            with bind_execution_context(exec_context):
                with tracer.span(
                    SpanKind.AGENT, "researcher", context=exec_context, parent=trace_root, task_id=task["task_id"]
                ) as agent_span:
                    with tracer.span(
                        SpanKind.TOOL, "search_web", context=exec_context, parent=agent_span
                    ) as tool_span:
                        request = ToolCallRequest(tool_name="search_web", arguments={"query": task["aspect"]}, identity=identity)
                        tool = workflow_guardrail.authorize_tool_call(request)
                        workflow_guardrail.enforce_high_risk_approval(request, tool)
                        tool_guardrail.validate_arguments(request.tool_name, request.arguments)
                        raw_output = search_tool_fn(query=task["aspect"])
                        sanitized = tool_guardrail.sanitize_output(str(raw_output))
                        tool_span.attributes["risk_level"] = tool.risk_level.value
                        search_snippet = sanitized.sanitized_text if sanitized.allowed else "[search result withheld: sensitive data]"

                    instructions = RESEARCH_INSTRUCTIONS_TEMPLATE.format(
                        question=state["question"], aspect=task["aspect"], reason=task["reason"], search_snippet=search_snippet
                    )
                    with tracer.span(
                        SpanKind.LLM, "researcher_llm", context=exec_context, parent=agent_span, agent_name="researcher"
                    ) as llm_span:
                        findings = text_llm_call(instructions, task["aspect"])
                        tokens, cost = llm_usage_delta(instructions, findings)
                        llm_span.attributes.update(prompt_tokens=tokens, completion_tokens=0, cost_usd=cost)
                    return findings, tokens, cost

        result: WorkerResult
        tokens_used = 0
        cost_usd = 0.0
        try:
            findings, tokens_used, cost_usd = run_with_timeout(
                do_research, timeout_seconds, tool_name=f"researcher[{task['task_id']}]"
            )
            result = {"task_id": task["task_id"], "aspect": task["aspect"], "status": "completed", "findings": findings, "error": None}
        except TimeoutError:
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "timeout",
                "findings": "",
                "error": f"Researcher exceeded the {timeout_seconds}s timeout for this task.",
            }
        except (AuthorizationError, ToolArgumentValidationError) as exc:
            result = {
                "task_id": task["task_id"],
                "aspect": task["aspect"],
                "status": "blocked",
                "findings": "",
                "error": f"Tool guardrail blocked this task: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 -- any single researcher failure must be recoverable
            result = {"task_id": task["task_id"], "aspect": task["aspect"], "status": "failed", "findings": "", "error": str(exc)}

        update: ProductionResearchState = {  # type: ignore[typeddict-item]
            "worker_results": [result],
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "trace": [f"Researcher[{task['task_id']}]: status={result['status']} aspect={task['aspect']!r}"],
        }
        return update

    return researcher


__all__ = ["WorkerInput", "make_researcher_node", "RESEARCH_INSTRUCTIONS_TEMPLATE"]
