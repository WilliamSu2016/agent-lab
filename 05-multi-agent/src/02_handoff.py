"""Experiment 2 (Multi-Agent): Handoff, built on the OpenAI Agents SDK.

TriageAgent classifies an incoming request as either ``research`` or
``coding`` and hands the *entire conversation* off to the matching specialist
agent, which then produces the final answer itself.

Design constraints for this experiment:

* OpenAI Agents SDK only (``agents.Agent`` / ``agents.Runner`` /
  ``agents.handoff``). No LangGraph here.
* Real SDK ``handoffs`` (``Agent(handoffs=[...])`` built via ``agents.handoff``),
  not "Agents as Tools" (``agent.as_tool(...)``) and not a hand-written
  Supervisor/Manager that inspects the classification and dispatches by hand.
  TriageAgent never reads its own classification result and manually calls
  ResearchAgent/CodingAgent -- the SDK's ``Runner`` performs the control
  transfer once TriageAgent's model emits a handoff tool call.
* TriageAgent never produces the final answer: its only job is to call one of
  the two handoff tools. Once a handoff tool call happens, the SDK switches
  the *active agent* for the rest of the run to the target specialist, which
  is the one whose model turn produces ``final_output``.
* Each specialist agent (ResearchAgent, CodingAgent) has a
  ``handoff_description`` (Requirement 9) so TriageAgent's model can decide
  which one fits an incoming request, without either specialist knowing about
  the other.
* Tracing stays fully enabled (Requirement 10) -- see ``src/tracing.py``. Every
  run produces one trace containing, in order: the TriageAgent's agent-run
  span, a handoff span (``transfer_to_research`` or ``transfer_to_coding``),
  and the specialist agent's agent-run span. See ``docs/10-HANDOFF.md`` for a
  walkthrough of what to look for in ``traces/trace.jsonl``.

Flow:

    User -> TriageAgent --handoff (transfer_to_research)--> ResearchAgent -> Final Answer
    User -> TriageAgent --handoff (transfer_to_coding)-->   CodingAgent   -> Final Answer
"""

from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.items import HandoffCallItem, HandoffOutputItem
from agents.models.interface import Model
from agents.result import RunResult
from openai import AsyncOpenAI

TRIAGE_AGENT_NAME = "TriageAgent"
RESEARCH_AGENT_NAME = "ResearchAgent"
CODING_AGENT_NAME = "CodingAgent"

# Requirement 9: each specialist agent that can be handed off to needs a
# handoff_description -- this is what the *calling* agent's model reads to
# decide which target fits an incoming request. It describes the specialist
# from the outside; it is not the specialist's own instructions.
RESEARCH_HANDOFF_DESCRIPTION = (
    "接手技术调研 / 资料研究类请求：解释某项技术是什么、原理是什么、"
    "以及不同技术方案之间的比较、优缺点、适用场景。不写代码。"
)
CODING_HANDOFF_DESCRIPTION = (
    "接手编写代码 / 实现方案类请求：根据需求写出具体代码、设计模块结构、"
    "给出可运行或可直接使用的实现。不做背景技术调研或方案比较。"
)

TRIAGE_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

你是 TriageAgent，一个分诊入口，只负责判断用户请求属于下面两类中的哪一类，
然后调用对应的转接工具，把整个对话交给对应的专家 Agent：

- research：用户在问某项技术是什么、原理、优缺点、多个技术方案之间的比较等
  研究类问题。
- coding：用户要求写代码、实现某个功能、给出具体的代码或模块设计。

规则（必须遵守）：
1. 你自己绝不回答用户的问题，也绝不生成任何关于 research 或 coding 内容本身的
   文字。你唯一的输出就是调用 transfer_to_research 或 transfer_to_coding 中的
   一个转接工具。
2. 如果请求同时包含调研和编码两部分，选择请求中最主要、最先需要处理的那一部分
   对应的专家。
3. 不要在对用户的回复中提及“转接”“分诊”这类内部机制。
"""

RESEARCH_AGENT_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

你是 ResearchAgent，一名只负责技术资料研究的专家。你会通过转接收到用户的原始
请求，请直接把它当作最终需要回答的问题来处理。

职责范围（只做这些）：
- 研究并回答技术原理、概念解释、多个技术方案之间的比较、优缺点权衡。
- 明确区分“比较确定的事实”和“不确定、需要进一步验证的内容”。

明确不做的事情：
- 不写实现代码、不给出文件结构或伪代码。
- 不再把请求转接给其他 Agent；你是这次对话的最终回答者，必须直接给出完整答案。

只依据用户的请求作答；如果信息不足以形成可靠结论，明确说明缺口，不要编造事实。"""

CODING_AGENT_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

你是 CodingAgent，一名只负责根据需求编写代码方案的专家。你会通过转接收到用户的
原始请求，请直接把它当作最终需要回答的问题来处理。

职责范围（只做这些）：
- 根据用户需求，给出具体的代码实现、关键模块划分、核心函数/类设计。
- 视需要给出简要的使用说明。

明确不做的事情：
- 不做背景技术调研或多方案比较（那不是你的职责；如果需求已经给出足够背景，直接
  基于它给出实现）。
- 不再把请求转接给其他 Agent；你是这次对话的最终回答者，必须直接给出完整答案。

只依据用户的需求作答；如果需求信息不足以给出可靠实现，明确说明缺口，不要臆造需求。"""


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_research_agent(model: Model | str) -> Agent:
    """Build ResearchAgent: a specialist that answers research requests directly.

    ResearchAgent has no ``handoffs`` of its own and never imports or
    references CodingAgent/TriageAgent -- it does not know they exist.
    """
    return Agent(
        name=RESEARCH_AGENT_NAME,
        handoff_description=RESEARCH_HANDOFF_DESCRIPTION,
        instructions=RESEARCH_AGENT_INSTRUCTIONS,
        model=model,
    )


def build_coding_agent(model: Model | str) -> Agent:
    """Build CodingAgent: a specialist that answers coding requests directly.

    CodingAgent has no ``handoffs`` of its own and never imports or
    references ResearchAgent/TriageAgent -- it does not know they exist.
    """
    return Agent(
        name=CODING_AGENT_NAME,
        handoff_description=CODING_HANDOFF_DESCRIPTION,
        instructions=CODING_AGENT_INSTRUCTIONS,
        model=model,
    )


def build_triage_agent(model: Model | str, *, research_agent: Agent, coding_agent: Agent) -> Agent:
    """Build TriageAgent: the only agent with ``handoffs``.

    Tool names are pinned to ``transfer_to_research`` / ``transfer_to_coding``
    via ``tool_name_override`` (rather than the SDK's default
    ``transfer_to_<agent_name>`` derivation) to match this experiment's
    required trace shape:

        Triage -> transfer_to_research -> ResearchAgent
        Triage -> transfer_to_coding   -> CodingAgent

    No Supervisor/Manager logic lives here: TriageAgent's model itself decides
    which handoff tool (if any) to call; this function only wires the tools up.
    """
    return Agent(
        name=TRIAGE_AGENT_NAME,
        instructions=TRIAGE_INSTRUCTIONS,
        model=model,
        handoffs=[
            handoff(research_agent, tool_name_override="transfer_to_research"),
            handoff(coding_agent, tool_name_override="transfer_to_coding"),
        ],
    )


@dataclass(frozen=True)
class HandoffAgents:
    """The three agents wired together for this experiment."""

    triage: Agent
    research: Agent
    coding: Agent


def build_agents(
    triage_model: Model | str,
    research_model: Model | str | None = None,
    coding_model: Model | str | None = None,
) -> HandoffAgents:
    """Build all three agents.

    ``research_model``/``coding_model`` default to ``triage_model`` (the usual
    case: one real model shared by all agents). Tests pass distinct scripted
    models per agent, since a handoff switches *which agent's model* answers
    the next turn.
    """
    research_agent = build_research_agent(research_model if research_model is not None else triage_model)
    coding_agent = build_coding_agent(coding_model if coding_model is not None else triage_model)
    triage_agent = build_triage_agent(triage_model, research_agent=research_agent, coding_agent=coding_agent)
    return HandoffAgents(triage=triage_agent, research=research_agent, coding=coding_agent)


def run_handoff(question: str, triage_agent: Agent, *, max_turns: int = 10) -> RunResult:
    """Run the multi-agent handoff flow and return the full ``RunResult``.

    The full ``RunResult`` (not just the final text) is returned so callers
    can inspect ``result.last_agent`` (which specialist actually answered) and
    ``result.new_items`` (the handoff call/output items) -- i.e. the trace of
    control transfer described in Requirement 6 ("Specialist Agent 接管后负责
    最终回答") and observed in ``docs/10-HANDOFF.md``.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    return Runner.run_sync(
        triage_agent,
        question,
        max_turns=max_turns,
        run_config=RunConfig(workflow_name=TRIAGE_AGENT_NAME),
    )


@dataclass(frozen=True)
class HandoffEvent:
    """One observed control transfer: ``source_agent --tool_name--> target_agent``."""

    source_agent: str
    tool_name: str
    target_agent: str


def extract_handoff_trace(result: RunResult) -> list[HandoffEvent]:
    """Read the handoff events out of a ``RunResult``'s run items, in order.

    Every handoff shows up as two adjacent run items: a ``HandoffCallItem``
    (the ``transfer_to_*`` tool call, produced by the source agent) followed
    by a ``HandoffOutputItem`` (recording ``source_agent``/``target_agent``).
    This reconstructs the human-readable chain, e.g.
    ``TriageAgent -> transfer_to_research -> ResearchAgent``.
    """
    events: list[HandoffEvent] = []
    pending_tool_name: str | None = None

    for item in result.new_items:
        if isinstance(item, HandoffCallItem):
            pending_tool_name = item.raw_item.name
        elif isinstance(item, HandoffOutputItem):
            tool_name = pending_tool_name or "<unknown>"
            events.append(
                HandoffEvent(
                    source_agent=item.source_agent.name,
                    tool_name=tool_name,
                    target_agent=item.target_agent.name,
                )
            )
            pending_tool_name = None

    return events


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    trace_file = enable_local_tracing()

    question = " ".join(sys.argv[1:]).strip() or "帮我写一个 LangGraph Agent。"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the handoff experiment.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)
    agents = build_agents(model)

    result = run_handoff(question, agents.triage)

    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    for event in extract_handoff_trace(result):
        print(f"HANDOFF: {event.source_agent} --{event.tool_name}--> {event.target_agent}")
    print(f"ANSWERED BY: {result.last_agent.name}")
    print("-" * 80)
    print(result.final_output)
    print("=" * 80)
    print(f"Full trace written to: {trace_file}")


if __name__ == "__main__":
    main()
