"""Experiment 3 (Multi-Agent): Supervisor / Manager via Agents-as-Tools.

SupervisorAgent stays in control of the whole conversation for its entire
run. It can call any of three specialist agents -- wrapped as ordinary
function tools via ``Agent.as_tool()`` -- one or more times, then writes the
final answer itself.

Design constraints for this experiment:

* OpenAI Agents SDK only (``agents.Agent`` / ``agents.Runner`` /
  ``Agent.as_tool()``). No LangGraph here.
* Agents-as-Tools, not Handoff: each specialist is wrapped with
  ``Agent.as_tool(tool_name=..., tool_description=...)`` and passed to
  SupervisorAgent's ``tools=[...]`` list -- not its ``handoffs=[...]`` list.
  Per the SDK's own doc-comment on ``Agent.as_tool`` (see
  ``docs/03-SUPERVISOR.md`` for the full quote), this differs from Handoff in
  two ways: (1) the specialist receives *generated input* (whatever text the
  Supervisor's model puts in the tool call), not the full conversation
  history; (2) the *Supervisor* continues the conversation after the tool
  call returns -- the specialist never takes over.
* No Supervisor/Manager hand-written dispatch logic decides which specialist
  to call. SupervisorAgent's own model decides, via ordinary tool-calling,
  which of ``research_expert`` / ``coding_expert`` / ``review_expert`` (zero,
  one, or several, in any order) to invoke for a given request.
* Requirement 1-5 follow directly from using ``as_tool()`` instead of
  ``handoffs``: SupervisorAgent is the *only* agent whose model turn ever
  produces ``final_output`` for the run (Requirement 1 & 5); each specialist's
  own model output becomes a tool result string fed back to Supervisor, never
  a message shown to the user directly (Requirement 2 & 3); Supervisor's
  model can invoke any subset of the three tools, including several in the
  same or different turns (Requirement 4).
* Tracing stays fully enabled (Requirement 10) -- see ``src/tracing.py``.
  Because each specialist call is a *nested agent run* (not a handoff), the
  trace shows Supervisor's agent-run span as the parent of each specialist's
  own nested agent-run span, with ordinary function-call/function-output
  spans around them -- see ``docs/03-SUPERVISOR.md`` for a walkthrough.

Flow (for a research + review request, as tested by
"比较 LangGraph 和 OpenAI Agents SDK，并给出推荐。"):

    User -> SupervisorAgent -> [tool call] research_expert -> ResearchAgent -> tool result
                             -> [tool call] review_expert   -> ReviewAgent   -> tool result
                             -> Final Answer (written by SupervisorAgent itself)
"""

from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.models.interface import Model
from agents.result import RunResult
from openai import AsyncOpenAI

SUPERVISOR_AGENT_NAME = "SupervisorAgent"
RESEARCH_AGENT_NAME = "ResearchAgent"
CODING_AGENT_NAME = "CodingAgent"
REVIEW_AGENT_NAME = "ReviewAgent"

# The tool names SupervisorAgent sees. These are deliberately *not* the agent
# names (Requirement 6/2: specialists are tools, invoked with a generic
# "expert" identity from the Supervisor's point of view -- the Supervisor
# does not need to know it is calling a full nested Agent run under the hood).
RESEARCH_TOOL_NAME = "research_expert"
CODING_TOOL_NAME = "coding_expert"
REVIEW_TOOL_NAME = "review_expert"

RESEARCH_TOOL_DESCRIPTION = (
    "调研某项技术：解释技术原理、比较多个技术方案的优缺点和适用场景。"
    "输入：一个具体的调研问题或需要比较的技术点。输出：调研结论文本。"
)
CODING_TOOL_DESCRIPTION = (
    "根据需求编写代码方案：给出具体实现、模块划分、关键函数/类设计。"
    "输入：一个具体的编码需求。输出：代码设计方案文本。"
)
REVIEW_TOOL_DESCRIPTION = (
    "审查一段研究结论或代码方案：检查其正确性、完整性、内部一致性，"
    "指出问题和缺口，并给出通过/需要修改的结论。"
    "输入：需要审查的材料原文（以及可选的背景说明）。输出：审查结果文本。"
)

RESEARCH_AGENT_INSTRUCTIONS = """你是 ResearchAgent，一名只负责技术资料研究的专家。
你收到的输入是别的 Agent（Supervisor）转述或整理过的调研请求，不是用户原始的完整
对话；把它当作需要独立回答的问题来处理。

职责范围（只做这些）：
- 研究并回答技术原理、概念解释、多个技术方案之间的比较、优缺点权衡。
- 明确区分“比较确定的事实”和“不确定、需要进一步验证的内容”。

明确不做的事情：
- 不写实现代码，不给出文件结构或伪代码。
- 不审查任何材料，不给“通过/不通过”的结论。
- 不直接与用户对话；你的输出会被调用方（Supervisor）读取、整合后再展示给用户，
  所以必须自洽、完整，但不需要包含寒暄或"以下是我的回答"这类客套话。

只依据你被给定的问题作答；如果信息不足以形成可靠结论，明确说明缺口，不要编造事实。"""

CODING_AGENT_INSTRUCTIONS = """你是 CodingAgent，一名只负责根据需求编写代码方案的专家。
你收到的输入是别的 Agent（Supervisor）转述或整理过的编码需求，不是用户原始的完整
对话；把它当作需要独立回答的问题来处理。

职责范围（只做这些）：
- 根据需求，给出具体的代码实现、关键模块划分、核心函数/类设计。
- 视需要给出简要的使用说明。

明确不做的事情：
- 不做背景技术调研或多方案比较。
- 不审查任何材料，不给“通过/不通过”的结论。
- 不直接与用户对话；你的输出会被调用方（Supervisor）读取、整合后再展示给用户，
  所以必须自洽、完整，但不需要包含寒暄或客套话。

只依据你被给定的需求作答；如果信息不足以给出可靠实现，明确说明缺口，不要臆造需求。"""

REVIEW_AGENT_INSTRUCTIONS = """你是 ReviewAgent，一名只负责审查他人产出的专家。
你收到的输入是需要被审查的材料本身（可能是研究结论、也可能是代码方案），以及
可选的背景说明；不是用户原始的完整对话。

职责范围（只做这些）：
- 检查给定材料本身的正确性、完整性、内部一致性。
- 指出材料中未说明、含糊或有风险的地方。
- 给出明确的审查结论：“通过”或“需要修改”，并说明理由。

明确不做的事情：
- 不重新做研究、不补充你自己的研究结论去替代材料中缺失的内容。
- 不重新设计代码方案、不写替代实现；只能针对给定方案提出修改建议。
- 不直接与用户对话；你的输出会被调用方（Supervisor）读取、整合后再展示给用户。

只依据你被给定的材料作答，信息不足时明确指出缺口，而不是去猜测或编造。"""

SUPERVISOR_INSTRUCTIONS = """你是 SupervisorAgent，负责处理用户的请求并给出最终答案。
你自己不是调研或编码方面的专家，但可以调用下面这些专家工具来获取信息，然后由你自己
整合、判断、写出最终回复给用户：

- research_expert：调研某项技术、比较多个技术方案。
- coding_expert：根据需求编写代码方案。
- review_expert：审查一段研究结论或代码方案，指出问题并给出通过/需要修改的结论。

规则（必须遵守）：
1. 你可以按需调用零个、一个或多个专家工具，也可以对同一个专家工具调用多次。
   如果请求既需要调研又需要判断调研结论是否可靠，应该先调用 research_expert 拿到
   调研结果，再调用 review_expert 审查这份结果，而不是自己臆断。
2. 专家工具的输出只是提供给你的原始材料，不会被直接展示给用户；最终展示给用户的
   回复必须由你自己撰写、整合、给出结论和建议。
3. 不要在最终回复中原样粘贴专家工具的输出；你要综合它们，给出清晰、连贯、面向
   用户的最终答案（可以引用专家工具的关键结论，但语言应该是你自己的）。
4. 如果用户的请求要求"给出推荐/建议"，必须在最终回复中给出明确的推荐，并说明依据。
"""


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_research_agent(model: Model | str) -> Agent:
    """Build ResearchAgent. Has no tools/handoffs of its own; never talks to the user directly."""
    return Agent(name=RESEARCH_AGENT_NAME, instructions=RESEARCH_AGENT_INSTRUCTIONS, model=model)


def build_coding_agent(model: Model | str) -> Agent:
    """Build CodingAgent. Has no tools/handoffs of its own; never talks to the user directly."""
    return Agent(name=CODING_AGENT_NAME, instructions=CODING_AGENT_INSTRUCTIONS, model=model)


def build_review_agent(model: Model | str) -> Agent:
    """Build ReviewAgent. Has no tools/handoffs of its own; never talks to the user directly."""
    return Agent(name=REVIEW_AGENT_NAME, instructions=REVIEW_AGENT_INSTRUCTIONS, model=model)


def build_supervisor_agent(
    model: Model | str,
    *,
    research_agent: Agent,
    coding_agent: Agent,
    review_agent: Agent,
) -> Agent:
    """Build SupervisorAgent: the only agent with ``tools`` wrapping the specialists.

    Each specialist is turned into a plain ``FunctionTool`` via
    ``Agent.as_tool()`` -- Requirement 6. This is deliberately *not*
    ``handoffs=[...]``: SupervisorAgent's own agent run keeps going after each
    tool call returns, and it is SupervisorAgent's model (not the specialist)
    that ultimately produces ``final_output`` for the run (Requirements 1-5).
    """
    return Agent(
        name=SUPERVISOR_AGENT_NAME,
        instructions=SUPERVISOR_INSTRUCTIONS,
        model=model,
        tools=[
            research_agent.as_tool(
                tool_name=RESEARCH_TOOL_NAME,
                tool_description=RESEARCH_TOOL_DESCRIPTION,
            ),
            coding_agent.as_tool(
                tool_name=CODING_TOOL_NAME,
                tool_description=CODING_TOOL_DESCRIPTION,
            ),
            review_agent.as_tool(
                tool_name=REVIEW_TOOL_NAME,
                tool_description=REVIEW_TOOL_DESCRIPTION,
            ),
        ],
    )


@dataclass(frozen=True)
class SupervisorAgents:
    """The four agents wired together for this experiment."""

    supervisor: Agent
    research: Agent
    coding: Agent
    review: Agent


def build_agents(
    supervisor_model: Model | str,
    research_model: Model | str | None = None,
    coding_model: Model | str | None = None,
    review_model: Model | str | None = None,
) -> SupervisorAgents:
    """Build all four agents.

    ``research_model``/``coding_model``/``review_model`` default to
    ``supervisor_model`` (the usual case: one real model shared by all
    agents). Tests pass distinct scripted models per agent, since each
    ``as_tool()`` call triggers a *separate, nested* agent run with its own
    model turns, independent of Supervisor's own model turns.
    """
    research_agent = build_research_agent(research_model if research_model is not None else supervisor_model)
    coding_agent = build_coding_agent(coding_model if coding_model is not None else supervisor_model)
    review_agent = build_review_agent(review_model if review_model is not None else supervisor_model)
    supervisor_agent = build_supervisor_agent(
        supervisor_model,
        research_agent=research_agent,
        coding_agent=coding_agent,
        review_agent=review_agent,
    )
    return SupervisorAgents(
        supervisor=supervisor_agent,
        research=research_agent,
        coding=coding_agent,
        review=review_agent,
    )


def run_supervisor(question: str, supervisor_agent: Agent, *, max_turns: int = 10) -> RunResult:
    """Run the Supervisor/Agents-as-Tools flow and return the full ``RunResult``.

    The full ``RunResult`` is returned (not just the final text) so callers
    can inspect ``result.last_agent`` (must always be SupervisorAgent --
    Requirement 1) and ``result.new_items`` (the tool call/output items for
    each specialist invocation -- Requirement 3/4), i.e. the same kind of
    observation this experiment's docs (``docs/03-SUPERVISOR.md``) walk
    through for the Handoff experiment's trace.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    return Runner.run_sync(
        supervisor_agent,
        question,
        max_turns=max_turns,
        run_config=RunConfig(workflow_name=SUPERVISOR_AGENT_NAME),
    )


def extract_expert_tool_calls(result: RunResult) -> list[str]:
    """Return the ordered list of specialist tool names Supervisor invoked.

    Reads ``ToolCallItem``s out of ``result.new_items`` -- each
    ``research_expert``/``coding_expert``/``review_expert`` call shows up here
    as an ordinary function tool call (not a ``HandoffCallItem``), because
    Agents-as-Tools keeps every specialist invocation inside the ordinary
    tool-calling mechanism.
    """
    names: list[str] = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", None)
            if name is not None:
                names.append(name)
    return names


def extract_expert_tool_results(result: RunResult) -> list[str]:
    """Return the ordered list of raw text results returned by specialist tool calls.

    These are exactly what SupervisorAgent's model sees fed back into its own
    conversation (Requirement 3: "Specialist 的输出返回给 Supervisor") -- never
    shown to the user directly unless Supervisor's own final answer chooses to
    quote them.
    """
    outputs: list[str] = []
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem):
            outputs.append(str(item.output))
    return outputs


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    from src.tracing import enable_local_tracing

    load_dotenv()
    trace_file = enable_local_tracing()

    question = " ".join(sys.argv[1:]).strip() or "比较 LangGraph 和 OpenAI Agents SDK，并给出推荐。"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the supervisor experiment.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)
    agents = build_agents(model)

    result = run_supervisor(question, agents.supervisor)

    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    for tool_name in extract_expert_tool_calls(result):
        print(f"SUPERVISOR CALLED: {tool_name}")
    print(f"ANSWERED BY: {result.last_agent.name}")
    print("-" * 80)
    print(result.final_output)
    print("=" * 80)
    print(f"Full trace written to: {trace_file}")


if __name__ == "__main__":
    main()
