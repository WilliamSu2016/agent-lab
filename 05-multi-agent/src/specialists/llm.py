"""Shared, minimal LLM call seam used by every Specialist Agent.

This is plain infrastructure (an HTTP client wrapper), not an Agent, not a
tool, and not a communication channel between agents. It exists only so each
Specialist Agent module can accept an injectable ``TextLLMCall`` for testing
(see ``tests/``) instead of hard-wiring the OpenAI client.

No agent in ``src/specialists/`` imports another agent's module, and no agent
calls another agent through this module. Each agent is wired independently
in its own ``main()``.
"""

from __future__ import annotations

from typing import Callable

# A TextLLMCall takes (system_prompt, user_prompt) and returns the model's
# plain-text reply. Every Specialist Agent in this package uses this exact
# shape: one instructions block in, one text block out. No tool calling, no
# multi-turn state, no shared memory.
TextLLMCall = Callable[[str, str], str]


def build_openai_text_llm_call(api_key: str, model_name: str, base_url: str) -> TextLLMCall:
    """Build a ``TextLLMCall`` backed by the plain ``openai`` client.

    Uses the base Chat Completions client directly -- not the ``openai-agents``
    SDK, not LangChain's chat model wrappers -- so each Specialist Agent stays
    a single, plain LLM call wrapped in a one-node LangGraph graph.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    return call
