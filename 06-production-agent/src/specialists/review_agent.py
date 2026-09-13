"""ReviewAgent -- a Specialist Agent that checks research results and code designs.

Experiment 1 (Specialist Agents) constraints, enforced by this module's design:

* Exactly one Specialist Agent, expressed as a one-node LangGraph
  ``StateGraph`` (``START -> review -> END``). No loops, no tool calls.
* No Handoff, no Supervisor/Manager, no Shared State, and no MCP: this module
  never imports ``research_agent`` or ``coding_agent``, never receives a
  reference to them, and has no notion that they exist. ReviewAgent does not
  call either agent to "get" something to review -- it only reviews whatever
  text content it is directly given as input.
* Explicit input/output contract: input is ``artifact_type`` (one of
  ``"research_report"`` or ``"code_design"``) plus ``content: str``; output is
  a single ``review_result: str`` written in a fixed section format (see
  ``INSTRUCTIONS``). No other fields cross the boundary.

See ``docs/01-SPECIALIST-AGENTS.md`` for the design rationale.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from src.specialists.llm import TextLLMCall

AGENT_NAME = "ReviewAgent"

ArtifactType = Literal["research_report", "code_design"]

INSTRUCTIONS = """你是 ReviewAgent，一名只负责审查他人产出的专家。你会收到两种材料之一：
- research_report：一份技术研究结果。
- code_design：一份代码设计方案。

职责范围（只做这些）：
- 检查给定材料本身的正确性、完整性、内部一致性。
- 指出材料中未说明、含糊或有风险的地方。
- 给出明确的审查结论：“通过”或“需要修改”，并说明理由。

明确不做的事情：
- 不重新做研究、不补充你自己的研究结论去替代材料中缺失的内容。
- 不重新设计代码方案、不写替代实现；只能针对给定方案提出修改建议。
- 不假设你可以联系产出该材料的角色去追问细节；只能依据你被给定的材料本身作出判断，
  信息不足时在审查结果中明确指出“缺口”，而不是去猜测或编造材料应该说什么。

输出格式（必须严格遵守，使用以下四个小节标题）：
1. 结论：通过 / 需要修改（二选一，且必须在这一行给出）。
2. 发现的问题：分条列出问题；如果没有，写“无”。
3. 缺口/需要补充的信息：列出材料未覆盖但审查该材料所必需的信息；如果没有，写“无”。
4. 改进建议：针对“发现的问题”给出具体、可执行的建议；如果没有问题，写“无”。

只依据你被给定的 artifact_type 和 content 作答，不要评审你未被给定的内容。"""


class ReviewAgentState(TypedDict):
    """Explicit input/output contract for ReviewAgent.

    Input: ``artifact_type`` and ``content``. Output: ``review_result``.
    Nothing else is exchanged.
    """

    artifact_type: ArtifactType
    content: str
    review_result: str


def initial_state(artifact_type: ArtifactType, content: str) -> ReviewAgentState:
    if artifact_type not in ("research_report", "code_design"):
        raise ValueError(f"artifact_type must be 'research_report' or 'code_design', got {artifact_type!r}")
    if not content.strip():
        raise ValueError("content must not be empty")
    return {"artifact_type": artifact_type, "content": content, "review_result": ""}


_ARTIFACT_LABELS: dict[ArtifactType, str] = {
    "research_report": "research_report（技术研究结果）",
    "code_design": "code_design（代码设计方案）",
}


def _format_user_prompt(state: ReviewAgentState) -> str:
    label = _ARTIFACT_LABELS[state["artifact_type"]]
    return f"材料类型：{label}\n\n材料内容：\n{state['content']}"


def make_review_node(text_llm_call: TextLLMCall):
    """The single Node of this Agent's graph: review the given artifact, once."""

    def review(state: ReviewAgentState) -> ReviewAgentState:
        result = text_llm_call(INSTRUCTIONS, _format_user_prompt(state))
        return {"review_result": result}  # type: ignore[typeddict-item]

    return review


def build_graph(text_llm_call: TextLLMCall):
    """Compile ReviewAgent as a one-node LangGraph graph: START -> review -> END."""
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ReviewAgentState)
    graph_builder.add_node("review", make_review_node(text_llm_call))
    graph_builder.add_edge(START, "review")
    graph_builder.add_edge("review", END)
    return graph_builder.compile()


def run_review_agent(artifact_type: ArtifactType, content: str, text_llm_call: TextLLMCall) -> str:
    """Convenience entry point: (artifact_type, content) in, review_result out."""
    graph = build_graph(text_llm_call)
    result = graph.invoke(initial_state(artifact_type, content))
    return result["review_result"]


def main() -> None:
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    from config import ConfigurationError, load_settings

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    args = sys.argv[1:]
    if args and args[0] in ("research_report", "code_design"):
        artifact_type: ArtifactType = args[0]  # type: ignore[assignment]
        content = " ".join(args[1:]).strip()
    else:
        artifact_type = "code_design"
        content = " ".join(args).strip()

    if not content:
        content = (
            "方案概述：用一个全局字典缓存所有请求结果。\n"
            "模块/文件结构：仅一个 cache.py。\n"
            "关键实现：cache = {}；get(key) 直接读写这个全局字典。\n"
            "权衡与风险：无。"
        )

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(settings)

    review = run_review_agent(artifact_type, content, text_llm_call)
    print("=" * 80)
    print(f"{AGENT_NAME} -- artifact_type: {artifact_type}")
    print("=" * 80)
    print(review)


if __name__ == "__main__":
    main()
