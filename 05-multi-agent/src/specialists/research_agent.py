"""ResearchAgent -- a Specialist Agent that researches technical material.

Experiment 1 (Specialist Agents) constraints, enforced by this module's design:

* Exactly one Specialist Agent, expressed as a one-node LangGraph
  ``StateGraph`` (``START -> research -> END``). No loops, no tool calls.
* No Handoff, no Supervisor/Manager, no Shared State, and no MCP: this module
  never imports ``coding_agent`` or ``review_agent``, never receives a
  reference to them, and has no notion that they exist. Its instructions do
  not mention "other agents" -- ResearchAgent's output must stand on its own.
* Explicit input/output contract: input is a single ``topic: str``; output is
  a single ``research_report: str`` written in a fixed section format (see
  ``INSTRUCTIONS``). No other fields cross the boundary.

See ``docs/01-SPECIALIST-AGENTS.md`` for the design rationale.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from src.specialists.llm import TextLLMCall

AGENT_NAME = "ResearchAgent"

INSTRUCTIONS = """你是 ResearchAgent，一名只负责研究技术资料的专家。

职责范围（只做这些）：
- 针对用户给出的技术主题，研究并总结相关的事实、原理、权衡取舍、最佳实践与常见陷阱。
- 明确区分“比较确定的事实”和“不确定、需要进一步验证的内容”。

明确不做的事情：
- 不设计代码方案、不写实现代码、不给出文件结构或伪代码。
- 不评审任何人的研究结果或代码方案，不给“通过/不通过”的结论。
- 不假设会有其他角色补充、纠正或校验你的结论；把你的研究结果当作会被直接使用的最终产出，
  必须自洽、完整、可独立理解。

输出格式（必须严格遵守，使用以下三个小节标题）：
1. 摘要：1-3 句话概括结论。
2. 关键要点：分条列出研究发现。
3. 风险与不确定项：列出证据不足、需要进一步验证的地方；如果没有，写“无”。

只依据你被给定的主题作答；如果信息不足以形成可靠结论，明确说明缺口，不要编造事实。"""


class ResearchAgentState(TypedDict):
    """Explicit input/output contract for ResearchAgent.

    Input: ``topic``. Output: ``research_report``. Nothing else is exchanged.
    """

    topic: str
    research_report: str


def initial_state(topic: str) -> ResearchAgentState:
    if not topic.strip():
        raise ValueError("topic must not be empty")
    return {"topic": topic, "research_report": ""}


def make_research_node(text_llm_call: TextLLMCall):
    """The single Node of this Agent's graph: research the topic, once."""

    def research(state: ResearchAgentState) -> ResearchAgentState:
        report = text_llm_call(INSTRUCTIONS, state["topic"])
        return {"research_report": report}  # type: ignore[typeddict-item]

    return research


def build_graph(text_llm_call: TextLLMCall):
    """Compile ResearchAgent as a one-node LangGraph graph: START -> research -> END."""
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ResearchAgentState)
    graph_builder.add_node("research", make_research_node(text_llm_call))
    graph_builder.add_edge(START, "research")
    graph_builder.add_edge("research", END)
    return graph_builder.compile()


def run_research_agent(topic: str, text_llm_call: TextLLMCall) -> str:
    """Convenience entry point: topic in, research_report out."""
    graph = build_graph(text_llm_call)
    result = graph.invoke(initial_state(topic))
    return result["research_report"]


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    topic = " ".join(sys.argv[1:]).strip() or (
        "LangGraph 中 StateGraph 与普通函数式流水线相比，核心优势是什么？"
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit(f"Set OPENAI_API_KEY and OPENAI_MODEL before running {AGENT_NAME}.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)

    report = run_research_agent(topic, text_llm_call)
    print("=" * 80)
    print(f"{AGENT_NAME} -- topic: {topic}")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    main()
