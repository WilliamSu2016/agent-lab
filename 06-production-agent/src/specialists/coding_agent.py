"""CodingAgent -- a Specialist Agent that designs code solutions from requirements.

Experiment 1 (Specialist Agents) constraints, enforced by this module's design:

* Exactly one Specialist Agent, expressed as a one-node LangGraph
  ``StateGraph`` (``START -> design -> END``). No loops, no tool calls.
* No Handoff, no Supervisor/Manager, no Shared State, and no MCP: this module
  never imports ``research_agent`` or ``review_agent``, never receives a
  reference to them, and has no notion that they exist.
* Explicit input/output contract: input is a single ``requirement: str``;
  output is a single ``design_proposal: str`` written in a fixed section
  format (see ``INSTRUCTIONS``). No other fields cross the boundary.

See ``docs/01-SPECIALIST-AGENTS.md`` for the design rationale.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from src.specialists.llm import TextLLMCall

AGENT_NAME = "CodingAgent"

INSTRUCTIONS = """你是 CodingAgent，一名只负责根据需求设计代码方案的专家。

职责范围（只做这些）：
- 针对用户给出的需求，设计可落地的代码实现方案：模块/文件划分、关键数据结构、
  核心函数/类的职责、关键流程的伪代码或简要示例代码。
- 指出该方案的主要权衡取舍（为什么这样设计，放弃了哪些替代方案）。

明确不做的事情：
- 不做背景技术调研、不解释某项技术“是什么”（那是另一个职责，不属于你；如果需求里
  已经给出足够背景，直接基于它设计即可）。
- 不评审别人给出的方案是否正确，不给“通过/不通过”的结论。
- 不假设会有其他角色帮你验证方案的正确性或补全背景知识；把你的方案当作会被直接使用
  的最终产出，必须自洽、可执行。

输出格式（必须严格遵守，使用以下四个小节标题）：
1. 方案概述：1-3 句话概括设计思路。
2. 模块/文件结构：列出建议的文件或模块划分及各自职责。
3. 关键实现：核心数据结构、函数签名或伪代码。
4. 权衡与风险：说明为什么这样设计，以及已知的局限或风险；如果没有，写“无”。

只依据你被给定的需求作答；如果需求信息不足以给出可靠方案，明确说明缺口，不要臆造需求。"""


class CodingAgentState(TypedDict):
    """Explicit input/output contract for CodingAgent.

    Input: ``requirement``. Output: ``design_proposal``. Nothing else is exchanged.
    """

    requirement: str
    design_proposal: str


def initial_state(requirement: str) -> CodingAgentState:
    if not requirement.strip():
        raise ValueError("requirement must not be empty")
    return {"requirement": requirement, "design_proposal": ""}


def make_design_node(text_llm_call: TextLLMCall):
    """The single Node of this Agent's graph: design a proposal, once."""

    def design(state: CodingAgentState) -> CodingAgentState:
        proposal = text_llm_call(INSTRUCTIONS, state["requirement"])
        return {"design_proposal": proposal}  # type: ignore[typeddict-item]

    return design


def build_graph(text_llm_call: TextLLMCall):
    """Compile CodingAgent as a one-node LangGraph graph: START -> design -> END."""
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(CodingAgentState)
    graph_builder.add_node("design", make_design_node(text_llm_call))
    graph_builder.add_edge(START, "design")
    graph_builder.add_edge("design", END)
    return graph_builder.compile()


def run_coding_agent(requirement: str, text_llm_call: TextLLMCall) -> str:
    """Convenience entry point: requirement in, design_proposal out."""
    graph = build_graph(text_llm_call)
    result = graph.invoke(initial_state(requirement))
    return result["design_proposal"]


def main() -> None:
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    from config import ConfigurationError, load_settings

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    requirement = " ".join(sys.argv[1:]).strip() or (
        "设计一个函数，用于将一批文本文档去重并按相似度聚类，返回聚类结果。"
    )

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(settings)

    proposal = run_coding_agent(requirement, text_llm_call)
    print("=" * 80)
    print(f"{AGENT_NAME} -- requirement: {requirement}")
    print("=" * 80)
    print(proposal)


if __name__ == "__main__":
    main()
