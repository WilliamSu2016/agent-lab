"""Agent definition built on the OpenAI Agents SDK.

The OpenAI Agents SDK owns the entire Agent Runtime: the LLM call loop, tool
dispatch, tool-call detection, tool-result routing, continuation, and
termination logic. Nothing here reimplements any of that -- this module only
declares *what* the agent is (instructions, model, tools), not *how* it runs.

* The Agent's ``instructions`` are migrated (see ``src/prompts.py``).
* The ``web_search`` business tool (see ``src/tools.py``) is wrapped with
  ``@function_tool`` so the SDK can discover it, generate its JSON schema,
  dispatch calls to it, and feed the result back into the Agent loop.
* No MCP, no hosted web search tool, no multi-agent, no handoffs, no
  Sessions, no guardrails, no RAG.
* The flow is: User -> Agent -> Runner -> Final Answer, with the Agent/LLM
  (not this code) deciding whether ``web_search`` gets called.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.models.interface import Model
from openai import AsyncOpenAI

from src.prompts import SYSTEM_PROMPT
from src.tools import web_search as _web_search

AGENT_NAME = "Research assistant"


@function_tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for current, factual information.

    Args:
        query: A focused web search query.
    """
    return _web_search(query)


def build_model(api_key: str, model_name: str, base_url: str) -> OpenAIChatCompletionsModel:
    """Build a Chat Completions model pointed at the configured OpenAI-compatible endpoint.

    The current project targets a private OpenAI-compatible gateway (see
    ``OPENAI_BASE_URL``) that only speaks the Chat Completions protocol, not the
    Responses API the SDK uses by default. ``OpenAIChatCompletionsModel`` is the
    SDK's own Chat Completions-compatible model adapter.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)


def build_agent(model: Model | str) -> Agent:
    """Create the SDK Agent: migrated instructions plus the migrated ``web_search`` tool.

    Whether ``web_search`` actually gets called for a given question is decided
    by the LLM (via the Agent's instructions), not by this function or any
    hand-written dispatch logic.
    """
    return Agent(
        name=AGENT_NAME,
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[web_search],
    )


def run_agent(question: str, agent: Agent, *, max_turns: int = 5) -> str:
    """Run the agent synchronously and return its final text answer.

    The SDK's built-in tracing stays enabled (``tracing_disabled`` defaults to
    ``False``), so every run is recorded as a trace containing agent-run,
    LLM-generation, and tool-call/tool-result spans. Where that trace data is
    *exported to* is controlled separately -- see ``src/tracing.py``, which
    routes it to a local file instead of OpenAI's public tracing backend
    (relevant since this project targets a private, OpenAI-compatible gateway).
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    result = Runner.run_sync(
        agent,
        question,
        max_turns=max_turns,
        run_config=RunConfig(workflow_name=AGENT_NAME),
    )
    if not isinstance(result.final_output, str):
        raise TypeError("Expected a plain-text final output from the agent.")
    return result.final_output
