"""Small shared helpers used by every agent node.

Approximate token/cost accounting: none of this project's ``TextLLMCall``
implementations return real token-usage metadata (the OpenAI response's
``usage`` field is discarded by ``src/agents/llm.py``'s thin wrapper), so
every agent node estimates tokens with a simple, clearly-labeled
heuristic (``len(text) // 4``, the same rough ratio commonly used for
English/mixed-language text) rather than pretending to have exact
provider-reported counts. A real deployment should thread the real
``usage.prompt_tokens``/``usage.completion_tokens`` fields through
instead -- see ``docs/observability.md``'s note on this.
"""

from __future__ import annotations

from src.observability.metrics import estimate_cost_usd


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def llm_usage_delta(prompt: str, completion: str) -> tuple[int, float]:
    """Returns ``(tokens_used_delta, cost_usd_delta)`` for one LLM call,
    given its prompt and completion text."""
    prompt_tokens = approx_tokens(prompt)
    completion_tokens = approx_tokens(completion)
    cost = estimate_cost_usd(prompt_tokens, completion_tokens)
    return prompt_tokens + completion_tokens, cost


__all__ = ["approx_tokens", "llm_usage_delta"]
