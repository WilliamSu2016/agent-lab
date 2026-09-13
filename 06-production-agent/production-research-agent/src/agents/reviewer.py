"""Reviewer: checks the Synthesizer's output before it becomes the final
answer, and decides whether another research loop is needed."""

from __future__ import annotations

import json

from src.agents.common import llm_usage_delta
from src.agents.llm import TextLLMCall
from src.graph.state import ProductionResearchState, ReviewVerdict
from src.observability.tracing import SpanKind, Tracer

REVIEWER_INSTRUCTIONS = """你是审阅者（Reviewer），负责检查研究综述（synthesis）是否
达到可以作为最终答案交付的质量。你必须逐一检查以下五个维度：

1. completeness（完整性）：是否覆盖了问题所需的主要方面，有没有明显遗漏？
2. factual_consistency（事实一致性）：综述内部有没有自相矛盾，或者明显不合常理的说法？
3. evidence_quality（证据质量）：结论是否有依据支撑，还是空泛断言？
4. logical_consistency（逻辑一致性）：论证过程是否连贯，结论是否从论据合理推出？
5. missing_aspects（遗漏的重要方面）：如果有完整性问题，具体列出还缺少哪些方面。

只输出一个 JSON 对象，且只有这一个 JSON 对象，不要有任何其他文字或代码块标记，格式为：

{
  "approved": true 或 false,
  "completeness": "对完整性的简短说明",
  "factual_consistency": "对事实一致性的简短说明",
  "evidence_quality": "对证据质量的简短说明",
  "logical_consistency": "对逻辑一致性的简短说明",
  "missing_aspects": ["缺失方面一", "缺失方面二"],
  "feedback": "给 Planner/Research 的具体、可执行的改进建议；approved 为 true 时可以为空字符串"
}

只有当以上五个维度都没有明显问题时才把 approved 设为 true。"""


def _parse_verdict(raw: str) -> ReviewVerdict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "approved" in parsed:
            missing = parsed.get("missing_aspects", [])
            if not isinstance(missing, list):
                missing = [str(missing)] if missing else []
            return {
                "approved": bool(parsed["approved"]),
                "completeness": str(parsed.get("completeness", "")),
                "factual_consistency": str(parsed.get("factual_consistency", "")),
                "evidence_quality": str(parsed.get("evidence_quality", "")),
                "logical_consistency": str(parsed.get("logical_consistency", "")),
                "missing_aspects": [str(item) for item in missing],
                "feedback": str(parsed.get("feedback", "")),
            }
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "approved": False,
        "completeness": "",
        "factual_consistency": "",
        "evidence_quality": "",
        "logical_consistency": "",
        "missing_aspects": [],
        "feedback": raw.strip() or "Reviewer output was not parseable JSON.",
    }


def make_reviewer_node(text_llm_call: TextLLMCall, *, tracer: Tracer):
    """Node: Reviewer. Owns ONLY ``review``, ``tokens_used``, ``cost_usd``."""

    def reviewer(state: ProductionResearchState) -> ProductionResearchState:
        with tracer.span(SpanKind.AGENT, "reviewer"):
            user_prompt = f"研究问题：{state['question']}\n\n研究综述：\n{state['synthesis']}"
            with tracer.span(SpanKind.LLM, "reviewer_llm", agent_name="reviewer") as llm_span:
                raw = text_llm_call(REVIEWER_INSTRUCTIONS, user_prompt)
                tokens, cost = llm_usage_delta(REVIEWER_INSTRUCTIONS + user_prompt, raw)
                llm_span.attributes.update(prompt_tokens=tokens, completion_tokens=0, cost_usd=cost)
            verdict = _parse_verdict(raw)

            update: ProductionResearchState = {  # type: ignore[typeddict-item]
                "review": verdict,
                "tokens_used": tokens,
                "cost_usd": cost,
                "trace": [
                    f"Reviewer: iteration {state['iteration']} approved={verdict['approved']!r} "
                    f"missing_aspects={verdict['missing_aspects']!r}"
                ],
            }
            return update

    return reviewer


__all__ = ["make_reviewer_node"]
