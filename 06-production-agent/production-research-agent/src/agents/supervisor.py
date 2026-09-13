"""Supervisor: entry guardrail + task planning + routing + finalization.

Owns the overall coordination -- there is no separate ``planner.py``
because in this architecture the "Supervisor" pattern folds task
decomposition into the same node family that owns entry/exit control
flow (mirrors the original Multi-Agent Research System's own framing:
"there is no separate supervisor.py file because the Supervisor's job
here is exactly what StateGraph orchestration + two small deterministic
nodes already do" -- here we simply gather all of that coordination logic,
including planning, into one module named for the role it plays).

Three guardrail integration points live here (closing the Final Review's
P0 #2 "guardrails never wired into the request path"):

1. ``supervisor_entry`` -- Layer 1 Input Guardrail (prompt injection /
   sensitive-data scan) on the raw user question. A flagged question sets
   ``blocked=True`` and routes straight to ``finalizer`` -- it never
   reaches the Planner/Researcher/Synthesizer/Reviewer loop.
2. ``planner`` -- Layer 2 authorization is implicitly satisfied here since
   planning issues no tool calls itself (research task authorization
   happens per-task in ``src/agents/researcher.py``).
3. ``finalizer`` -- Layer 2 output validation
   (``AgentWorkflowGuardrail.validate_output``) on the final answer before
   it is ever returned to the caller.
"""

from __future__ import annotations

import json
import logging
import uuid

from src.graph.state import ProductionResearchState, ResearchTask
from src.observability.logging import log_event
from src.observability.tracing import SpanKind, Tracer
from src.reliability.errors import ControlledFailure
from src.security.guardrails import AgentWorkflowGuardrail, scan_prompt_injection
from src.agents.common import llm_usage_delta
from src.agents.llm import TextLLMCall

PLANNER_INSTRUCTIONS = """你是一个研究任务规划器（Planner）。给定一个宽泛的研究问题，
你需要把它拆解成若干个具体、互不重叠的"研究方面"（aspect），每个方面都足够具体，
可以交给一个独立的研究者去单独调查，不需要了解其他方面的内容。

不要假设任何固定的候选清单——具体要拆出哪些方面，完全取决于这个问题本身。

只输出一个 JSON 数组，数组元素是字符串，每个字符串是一个研究方面的简短描述
（不超过 20 个字）。不要输出任何其他文字、解释或 Markdown 代码块标记，只输出
JSON 数组本身，例如：

["方面一", "方面二", "方面三"]
"""

PLANNER_REFINE_INSTRUCTIONS = """你是一个研究任务规划器（Planner），这一次你的任务不是
从头拆解问题，而是根据审阅者（Reviewer）指出的具体缺口，生成一组"补充研究任务"。

你会收到：原始问题、审阅者认为缺失的方面列表（missing_aspects）、以及审阅者的
具体反馈（feedback）。请只针对这些缺口生成新的、具体的研究方面，不要重复已经
覆盖过的内容，也不要生成与缺口无关的新方面。

只输出一个 JSON 数组，数组元素是字符串，格式要求与拆解阶段相同，不要输出任何
其他文字：

["补充方面一", "补充方面二"]
"""


def _parse_aspects(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    lines = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.、） )").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def make_supervisor_entry_node(*, tracer: Tracer, logger, input_block_threshold: float = 0.5):
    """Node: Supervisor (entry). Layer 1 Input Guardrail runs here, on the
    raw user question, before anything else in the graph executes."""

    def supervisor_entry(state: ProductionResearchState) -> ProductionResearchState:
        with tracer.span(SpanKind.AGENT, "supervisor_entry"):
            result = scan_prompt_injection(state["question"], block_threshold=input_block_threshold)
            update: ProductionResearchState = {  # type: ignore[typeddict-item]
                "trace": [
                    f"Supervisor: coordinating research for question={state['question']!r} "
                    f"(input_guardrail risk_score={result.risk_score:.2f} allowed={result.allowed!r})"
                ],
            }
            if not result.allowed:
                log_event(
                    logger,
                    logging.WARNING,
                    "input_guardrail_blocked",
                    risk_score=result.risk_score,
                    matched_rules=list(result.matched_rules),
                )
                update["blocked"] = True
                update["blocked_reason"] = result.reason or "input guardrail blocked this request"
            return update

    return supervisor_entry


def route_after_entry(state: ProductionResearchState) -> str:
    return "finalizer" if state["blocked"] else "planner"


def make_planner_node(text_llm_call: TextLLMCall, *, tracer: Tracer):
    """Node: Planner. Owns ONLY ``tasks``, ``iteration``, ``tokens_used``,
    ``cost_usd``."""

    def planner(state: ProductionResearchState) -> ProductionResearchState:
        with tracer.span(SpanKind.AGENT, "planner") as span:
            iteration = state.get("iteration", 0) + 1
            review = state.get("review")

            if iteration == 1 or not review or not review.get("missing_aspects"):
                instructions, user_prompt = PLANNER_INSTRUCTIONS, state["question"]
                reason = "initial decomposition"
            else:
                instructions = PLANNER_REFINE_INSTRUCTIONS
                user_prompt = (
                    f"原始问题：{state['question']}\n\n"
                    f"审阅者认为缺失的方面：{json.dumps(review['missing_aspects'], ensure_ascii=False)}\n\n"
                    f"审阅者的反馈：{review['feedback']}"
                )
                reason = f"gap-filling for iteration {iteration} (Reviewer feedback)"

            with tracer.span(SpanKind.LLM, "planner_llm", agent_name="planner") as llm_span:
                raw = text_llm_call(instructions, user_prompt)
                tokens, cost = llm_usage_delta(instructions + user_prompt, raw)
                llm_span.attributes.update(prompt_tokens=tokens, completion_tokens=0, cost_usd=cost)

            aspects = _parse_aspects(raw)
            if not aspects:
                raise ValueError("Planner produced no research aspects; cannot continue.")

            capped = aspects[: state["max_workers"]]
            tasks: list[ResearchTask] = [
                {"task_id": f"iter{iteration}-task{index}-{uuid.uuid4().hex[:8]}", "aspect": aspect, "reason": reason}
                for index, aspect in enumerate(capped)
            ]
            span.attributes["iteration"] = iteration
            span.attributes["task_count"] = len(tasks)

            update: ProductionResearchState = {  # type: ignore[typeddict-item]
                "tasks": tasks,
                "iteration": iteration,
                "tokens_used": tokens,
                "cost_usd": cost,
                "trace": [f"Planner: iteration {iteration} produced {len(tasks)} task(s) ({reason})"],
            }
            return update

    return planner


def fan_out_to_workers(state: ProductionResearchState):
    from langgraph.types import Send

    return [
        Send(
            "researcher",
            {
                "question": state["question"],
                "task": task,
                "per_worker_timeout_seconds": state["per_worker_timeout_seconds"],
                "identity": state["identity"],
                "context": state["context"],
                "mode": state["mode"],
            },
        )
        for task in state["tasks"]
    ]


def make_finalizer_node(workflow_guardrail: AgentWorkflowGuardrail, *, tracer: Tracer, logger):
    """Node: Supervisor (exit / Finalizer). Layer 2 Output Validation runs
    here -- the last checkpoint before ``final_answer`` leaves this graph."""

    def finalizer(state: ProductionResearchState) -> ProductionResearchState:
        with tracer.span(SpanKind.AGENT, "finalizer"):
            if state["blocked"] and not state["synthesis"]:
                update: ProductionResearchState = {  # type: ignore[typeddict-item]
                    "final_answer": f"[BLOCKED] {state['blocked_reason']}",
                    "trace": ["Supervisor: request blocked by input guardrail; no research was performed."],
                }
                return update

            review = state["review"]
            header = state["synthesis"].strip()
            if review["approved"]:
                note = f"Review: approved after {state['iteration']} iteration(s)."
            else:
                note = (
                    f"Review: NOT approved after {state['iteration']} iteration(s) "
                    f"(max_iterations={state['max_iterations']}); delivering best-effort answer. "
                    f"Outstanding feedback: {review['feedback']}"
                )
            candidate = f"{header}\n\n[{note}]"

            validation = workflow_guardrail.validate_output(candidate)
            if not validation.allowed:
                log_event(logger, logging.ERROR, "output_guardrail_blocked", reason=validation.blocked_reason)
                update = {  # type: ignore[typeddict-item]
                    "final_answer": (
                        "[BLOCKED] The generated answer contained high-severity sensitive data and "
                        "was withheld. Please contact support with request trace details."
                    ),
                    "blocked": True,
                    "blocked_reason": validation.blocked_reason or "output guardrail blocked this response",
                    "trace": ["Supervisor: finalized answer BLOCKED by output guardrail."],
                }
                return update

            update = {  # type: ignore[typeddict-item]
                "final_answer": validation.sanitized_text,
                "trace": [f"Supervisor: finalized answer (approved={review['approved']!r})"],
            }
            return update

    return finalizer


def make_notify_node(
    send_email_tool_fn,
    workflow_guardrail: AgentWorkflowGuardrail,
    tool_guardrail,
    *,
    tracer: Tracer,
):
    """Node: optional HIGH-risk "notify by email" side effect -- the
    concrete proof that this graph enforces ``Agent -> approval -> Tool``
    (never ``Agent -> Tool``) for a HIGH-risk tool, not just at the
    standalone guardrail-unit-test level.

    A no-op unless the run was started with ``notify_email`` set (never
    fires during a normal research run). When it does fire:

    1. First invocation: submits (or looks up, idempotently keyed by this
       run's ``trace_id``) an :class:`~src.security.authorization
       .ApprovalRequest` and finds it still ``PENDING`` -- lets
       :class:`ApprovalRequiredError` propagate *uncaught*, exactly like a
       genuine worker crash (see ``src.agents.researcher``'s module
       docstring for the contrast: that node deliberately swallows
       failures for resilience; this node deliberately does NOT, because
       "pending human approval" must pause the run, not silently continue
       past the gate). The run is reported ``"interrupted"`` by
       ``src.api.runs.RunRegistry`` and becomes resumable exactly like a
       crashed run.
    2. An operator approves (or denies) via ``POST
       /approvals/{request_id}/decide``.
    3. ``POST /runs/{run_id}/resume`` re-invokes this same node (LangGraph
       never marks a step "done" if it raised) -- this time
       ``require_approved`` either succeeds (tool call proceeds through
       Layer 3 argument validation + output sanitization) or raises
       :class:`ApprovalDeniedError`, which this node DOES catch (a denial
       is a legitimate terminal outcome, not a crash).
    """

    def notify_action(state: ProductionResearchState) -> ProductionResearchState:
        if not state.get("notify_email"):
            return {}

        from src.security.authorization import ApprovalDeniedError, Identity
        from src.security.guardrails import ToolCallRequest

        with tracer.span(SpanKind.AGENT, "notify_action"):
            identity = Identity(
                user_id=state["identity"]["user_id"],
                tenant_id=state["identity"]["tenant_id"],
                roles=frozenset(state["identity"]["roles"]),
            )
            arguments = {
                "to": state["notify_email"],
                "subject": f"Research complete: {state['question'][:60]}",
                "body": state["final_answer"][:2000],
            }

            # Idempotent submission keyed by trace_id: a retry of this same
            # node (via resume() after an earlier ApprovalRequiredError)
            # must poll the SAME pending request instead of creating a new
            # one every retry.
            idempotency_key = f"notify:{state['context']['trace_id']}"
            approval = workflow_guardrail.approvals.submit_or_get(idempotency_key, "send_email", arguments, identity)

            request = ToolCallRequest(
                tool_name="send_email",
                arguments=arguments,
                identity=identity,
                approval_request_id=approval.request_id,
            )
            tool = workflow_guardrail.authorize_tool_call(request)

            try:
                workflow_guardrail.enforce_high_risk_approval(request, tool)
            except ApprovalDeniedError:
                return {
                    "notify_status": "denied",
                    "trace": [f"Supervisor: notify_action denied (request_id={approval.request_id})."],
                }
            # ApprovalRequiredError (still PENDING) is intentionally left
            # to propagate uncaught -- see docstring above.

            tool_guardrail.validate_arguments(tool.name, arguments)
            raw_output = send_email_tool_fn(**arguments)
            tool_guardrail.sanitize_output(str(raw_output))
            return {
                "notify_status": "sent",
                "trace": [f"Supervisor: notify_action sent email to={arguments['to']!r}."],
            }

    return notify_action


def route_after_review(state: ProductionResearchState) -> str:
    if state["review"]["approved"]:
        return "finalizer"
    if state["iteration"] >= state["max_iterations"]:
        return "finalizer"
    return "planner"


def controlled_failure_for(state: ProductionResearchState, max_iterations: int) -> ControlledFailure | None:
    if not state["review"]["approved"] and state["iteration"] >= max_iterations:
        return ControlledFailure(
            reason="review not approved within max_iterations",
            iterations_used=state["iteration"],
            max_iterations=max_iterations,
            last_feedback=state["review"]["feedback"],
        )
    return None


__all__ = [
    "make_supervisor_entry_node",
    "route_after_entry",
    "make_planner_node",
    "fan_out_to_workers",
    "make_finalizer_node",
    "make_notify_node",
    "route_after_review",
    "controlled_failure_for",
]
