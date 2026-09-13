"""Synthesizer: integrates every Research Worker's findings into one report.

Ownership: owns ONLY ``synthesis``. It reads the full, accumulated
``worker_results`` (across every iteration so far -- a retry never discards
earlier findings, since ``worker_results`` is reducer-merged) and produces a
single coherent synthesis every time it runs, whether this is the first pass
or the third retry after Reviewer feedback.
"""

from __future__ import annotations

from src.multi_agent_research.state import MultiAgentResearchState, WorkerResult
from src.specialists.llm import TextLLMCall

SYNTHESIZER_INSTRUCTIONS = """你是汇总者（Synthesizer），会收到多个独立研究者对同一个
研究问题下不同方面的研究结果（其中可能有个别方面因超时或失败而没有结论，也可能包含
之前几轮研究和最新一轮补充研究的结果）。

请把这些结果整合成一份连贯的研究综述：
1. 先给出总体概述。
2. 按方面分节陈述关键发现，标注证据依据（如果研究结果里有提及）。
3. 如果有方面缺失或失败，在综述中明确指出这是"未能完成的研究项"，不要假装它存在结论，
   也不要替它编造内容。
4. 不要在这一步下最终结论或做质量判断——质量检查是审阅者（Reviewer）的职责，
   你只负责如实、完整地整合已有的研究发现。"""


def _format_results(results: list[WorkerResult]) -> str:
    sections = []
    for result in results:
        if result["status"] == "completed":
            sections.append(f"### {result['aspect']}\n{result['findings']}")
        else:
            sections.append(f"### {result['aspect']}\n[{result['status'].upper()}] {result['error']}")
    return "\n\n".join(sections) if sections else "(no research results were produced)"


def make_synthesizer_node(text_llm_call: TextLLMCall):
    """Node: Synthesizer. Owns ONLY ``synthesis``."""

    def synthesizer(state: MultiAgentResearchState) -> MultiAgentResearchState:
        user_prompt = f"研究问题：{state['question']}\n\n" + _format_results(state["worker_results"])
        result = text_llm_call(SYNTHESIZER_INSTRUCTIONS, user_prompt)

        update: MultiAgentResearchState = {  # type: ignore[typeddict-item]
            "synthesis": result,
            "trace": [
                f"Synthesizer: synthesized {len(state['worker_results'])} accumulated worker result(s)"
            ],
        }
        return update

    return synthesizer
