"""Synthesizer: integrates every Researcher's findings into one report."""

from __future__ import annotations

from src.agents.common import llm_usage_delta
from src.agents.llm import TextLLMCall
from src.graph.state import ProductionResearchState, WorkerResult
from src.observability.tracing import SpanKind, Tracer

SYNTHESIZER_INSTRUCTIONS = """你是汇总者（Synthesizer），会收到多个独立研究者对同一个
研究问题下不同方面的研究结果（其中可能有个别方面因超时、失败或被安全策略拦截而没有
结论，也可能包含之前几轮研究和最新一轮补充研究的结果）。

请把这些结果整合成一份连贯的研究综述：
1. 先给出总体概述。
2. 按方面分节陈述关键发现，标注证据依据（如果研究结果里有提及）。
3. 如果有方面缺失、失败或被拦截，在综述中明确指出这是"未能完成的研究项"，不要假装
   它存在结论，也不要替它编造内容。
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


def make_synthesizer_node(text_llm_call: TextLLMCall, *, tracer: Tracer):
    """Node: Synthesizer. Owns ONLY ``synthesis``, ``tokens_used``, ``cost_usd``."""

    def synthesizer(state: ProductionResearchState) -> ProductionResearchState:
        with tracer.span(SpanKind.AGENT, "synthesizer"):
            user_prompt = f"研究问题：{state['question']}\n\n" + _format_results(state["worker_results"])
            with tracer.span(SpanKind.LLM, "synthesizer_llm", agent_name="synthesizer") as llm_span:
                result = text_llm_call(SYNTHESIZER_INSTRUCTIONS, user_prompt)
                tokens, cost = llm_usage_delta(SYNTHESIZER_INSTRUCTIONS + user_prompt, result)
                llm_span.attributes.update(prompt_tokens=tokens, completion_tokens=0, cost_usd=cost)

            update: ProductionResearchState = {  # type: ignore[typeddict-item]
                "synthesis": result,
                "tokens_used": tokens,
                "cost_usd": cost,
                "trace": [f"Synthesizer: synthesized {len(state['worker_results'])} accumulated worker result(s)"],
            }
            return update

    return synthesizer


__all__ = ["make_synthesizer_node"]
