"""A complete LangGraph Research Agent.

User gives a research question (e.g. "Compare Python, TypeScript, and Go for
building AI agents."), and the graph:

1. Analyzes the question and drafts a research plan          -- ``planner``
2. Decides whether to search the web, and issues the search   -- ``agent`` (+ ``tools``)
3. Judges from the search results whether more research is
   needed (loops back to ``agent``, or moves on)              -- ``should_continue``
4. Synthesizes all research into a final answer                -- ``finalize``

Graph:

    START
      |
      v
    planner
      |
      v
    agent  <---------------------.
      |                          |
      v                          |
    should_continue               |
      |-- "tools" --> tools ------'
      `-- "finalize" --> finalize --> END

Design constraints:

* ``StateGraph`` + explicit ``ResearchAgentState`` + Nodes + Edges +
  Conditional Edges (per the assignment).
* The Agent Loop (``agent <-> tools``) is expressed entirely as graph edges,
  not as a hand-written ``while`` loop.
* Every tool call result is written into ``ResearchAgentState`` (the
  ``tool_results`` field), not just left buried inside opaque message blobs.
* A hard ``max_steps`` cap prevents infinite looping.
* A real LLM is used (via the plain ``openai`` client) -- but NOT the
  ``openai-agents`` SDK, NOT CrewAI, NOT the LangChain Agent abstraction
  (``AgentExecutor``/``create_react_agent``), NOT MCP, NOT LangGraph
  Memory/Checkpointer, and NOT multi-agent (there is exactly one agent loop,
  no handoffs, no sub-agents).

See ``docs/05-RESEARCH-AGENT.md`` for the full walkthrough, including a
Mermaid diagram of the compiled graph.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

MAX_STEPS = 6


# ---------------------------------------------------------------------------
# LLM call seams -- plain callables, never an Agent/Runner/AgentExecutor.
#
# Two shapes are used:
#   * TextLLMCall:  (system_prompt, user_prompt) -> plain text.
#     Used by ``planner`` (draft a plan) and ``finalize`` (write the answer).
#   * AgentLLMCall: (message_history) -> a tool-call request OR "done".
#     Used by ``agent`` to decide whether more research is needed.
# ---------------------------------------------------------------------------


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]


class AgentDecision(TypedDict, total=False):
    """Either a tool call, or nothing (meaning: research is sufficient)."""

    tool_call: ToolCall
    reasoning: str


TextLLMCall = Callable[[str, str], str]
AgentLLMCall = Callable[[list[dict[str, Any]]], AgentDecision]
SearchWebCall = Callable[[str], dict[str, Any]]

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current, factual information relevant to the research question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def build_openai_text_llm_call(api_key: str, model_name: str, base_url: str) -> TextLLMCall:
    """Plain-text LLM call using the base ``openai`` client (no Agents SDK)."""
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


def build_openai_agent_llm_call(api_key: str, model_name: str, base_url: str) -> AgentLLMCall:
    """Tool-calling LLM call using the base ``openai`` client (no Agents SDK,
    no LangChain Agent abstraction -- just one Chat Completions request that
    may or may not come back with a function/tool call)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def call(messages: list[dict[str, Any]]) -> AgentDecision:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=[SEARCH_TOOL_SCHEMA],
        )
        choice = response.choices[0].message
        if choice.tool_calls:
            tool_call = choice.tool_calls[0]
            args = json.loads(tool_call.function.arguments or "{}")
            return {"tool_call": {"name": tool_call.function.name, "args": args}}
        return {"reasoning": choice.content or ""}

    return call


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ToolResult(TypedDict):
    query: str
    result: dict[str, Any]


class ResearchAgentState(TypedDict):
    """Everything the Research Agent needs, carried across every Node."""

    question: str
    plan: str
    messages: list[dict[str, Any]]
    pending_tool_call: ToolCall | None
    tool_results: list[ToolResult]
    steps: int
    final_answer: str


def initial_state(question: str) -> ResearchAgentState:
    return {
        "question": question,
        "plan": "",
        "messages": [],
        "pending_tool_call": None,
        "tool_results": [],
        "steps": 0,
        "final_answer": "",
    }


def _print_state(label: str, state: ResearchAgentState) -> None:
    print(f"[{label}] {state}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_INSTRUCTIONS = """You are the Planner step of a research agent. Given the user's
research question, write a short research plan (3-6 bullet points) describing what should
be investigated and in what order. Do not answer the question yourself yet."""

AGENT_SYSTEM_PROMPT_TEMPLATE = """You are the research agent's decision step. You are
investigating this question:

{question}

Research plan:
{plan}

You may call the `search_web` tool as many times as needed to gather current, factual
evidence. Once you believe you have gathered enough evidence to answer the question
well, stop calling tools -- simply respond with a short note on why the research is
sufficient (do not write the final answer yourself, another step does that)."""

FINALIZE_INSTRUCTIONS = """You are the Finalize step of a research agent. You are given
the original question, the research plan, and all the web search results gathered during
research. Summarize the research findings and then give a clear final answer to the
question, with an explicit recommendation where relevant."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_planner_node(text_llm_call: TextLLMCall):
    """Node 1: analyze the question and draft a research plan.

    Owns ``plan`` and seeds ``messages`` for the agent loop that follows.
    """

    def planner(state: ResearchAgentState) -> ResearchAgentState:
        _print_state("planner: before", state)

        plan = text_llm_call(PLANNER_INSTRUCTIONS, state["question"])
        seeded_messages = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT_TEMPLATE.format(question=state["question"], plan=plan),
            },
            {"role": "user", "content": state["question"]},
        ]

        update: ResearchAgentState = {"plan": plan, "messages": seeded_messages}  # type: ignore[typeddict-item]
        _print_state("planner: after", {**state, **update})
        return update

    return planner


def make_agent_node(agent_llm_call: AgentLLMCall, *, max_steps: int = MAX_STEPS):
    """Node 2: decide whether to keep researching (tool call) or stop.

    Owns ``messages`` (appends its own turn), ``pending_tool_call``, and
    ``steps``. The actual "should we keep going?" judgment is delegated to
    the LLM's decision here; ``should_continue`` only reads the resulting
    ``pending_tool_call`` field to route the graph.
    """

    def agent(state: ResearchAgentState) -> ResearchAgentState:
        _print_state("agent: before", state)

        steps = state["steps"] + 1

        if steps > max_steps:
            # Requirement 4: hard cap so the loop can never run forever.
            update: ResearchAgentState = {
                "steps": steps,
                "pending_tool_call": None,
                "messages": state["messages"]
                + [{"role": "assistant", "content": f"Reached the maximum of {max_steps} step(s); stopping research."}],
            }  # type: ignore[typeddict-item]
            _print_state("agent: after (max steps reached)", {**state, **update})
            return update

        decision = agent_llm_call(state["messages"])

        if "tool_call" in decision:
            tool_call = decision["tool_call"]
            assistant_message = {"role": "assistant", "content": None, "tool_call": tool_call}
            update = {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": tool_call,
                "steps": steps,
            }  # type: ignore[typeddict-item]
        else:
            reasoning = decision.get("reasoning", "Research considered sufficient.")
            assistant_message = {"role": "assistant", "content": reasoning}
            update = {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": None,
                "steps": steps,
            }  # type: ignore[typeddict-item]

        _print_state("agent: after", {**state, **update})
        return update

    return agent


def make_tools_node(search_web_call: SearchWebCall = _web_search):
    """Node 3: execute the pending tool call.

    Owns ``messages`` (appends the tool result) and ``tool_results``
    (Requirement 3: every tool result is written into State, in a dedicated,
    structured field -- not just buried inside a message blob). Clears
    ``pending_tool_call``. Any tool exception is caught and turned into a
    safe error result instead of crashing the graph.
    """

    def tools(state: ResearchAgentState) -> ResearchAgentState:
        _print_state("tools: before", state)

        tool_call = state["pending_tool_call"]
        assert tool_call is not None, "tools node reached without a pending_tool_call"
        query = tool_call["args"].get("query", "")

        try:
            if tool_call["name"] != "search_web":
                raise ValueError(f"Unknown tool: {tool_call['name']!r}")
            result = search_web_call(query)
        except Exception as exc:  # noqa: BLE001 -- any tool failure must be recoverable
            result = {"error": str(exc)}

        tool_message = {"role": "tool", "name": tool_call["name"], "content": result}
        tool_result: ToolResult = {"query": query, "result": result}

        update: ResearchAgentState = {
            "messages": state["messages"] + [tool_message],
            "tool_results": state["tool_results"] + [tool_result],
            "pending_tool_call": None,
        }  # type: ignore[typeddict-item]
        _print_state("tools: after", {**state, **update})
        return update

    return tools


def _format_tool_results(tool_results: list[ToolResult]) -> str:
    if not tool_results:
        return "(no web searches were performed)"
    sections = []
    for item in tool_results:
        sections.append(f"Query: {item['query']}\nResult: {json.dumps(item['result'])}")
    return "\n\n".join(sections)


def make_finalize_node(text_llm_call: TextLLMCall):
    """Node 4: summarize research_results and write the final answer.

    Owns ``final_answer`` only.
    """

    def finalize(state: ResearchAgentState) -> ResearchAgentState:
        _print_state("finalize: before", state)

        user_prompt = (
            f"Original question: {state['question']}\n\n"
            f"Research plan:\n{state['plan']}\n\n"
            f"Search results gathered:\n{_format_tool_results(state['tool_results'])}"
        )
        final_answer = text_llm_call(FINALIZE_INSTRUCTIONS, user_prompt)

        update: ResearchAgentState = {"final_answer": final_answer}  # type: ignore[typeddict-item]
        _print_state("finalize: after", {**state, **update})
        return update

    return finalize


# ---------------------------------------------------------------------------
# Routing function (NOT a Node)
# ---------------------------------------------------------------------------


def should_continue(state: ResearchAgentState) -> Literal["tools", "finalize"]:
    """Decide, from State alone, whether more research (a tool call) is
    pending, or whether it's time to move on to ``finalize``."""
    if state["pending_tool_call"] is not None:
        return "tools"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    text_llm_call: TextLLMCall,
    agent_llm_call: AgentLLMCall,
    search_web_call: SearchWebCall = _web_search,
    *,
    max_steps: int = MAX_STEPS,
):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(ResearchAgentState)

    graph_builder.add_node("planner", make_planner_node(text_llm_call))
    graph_builder.add_node("agent", make_agent_node(agent_llm_call, max_steps=max_steps))
    graph_builder.add_node("tools", make_tools_node(search_web_call))
    graph_builder.add_node("finalize", make_finalize_node(text_llm_call))

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph_builder.add_edge("tools", "agent")  # Agent Loop closes through the graph.
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()


def print_mermaid_diagram(graph) -> str:
    """Return (and print) the compiled graph's Mermaid diagram source."""
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)
    return mermaid


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    question = " ".join(sys.argv[1:]).strip() or (
        "Compare Python, TypeScript, and Go for building AI agents, and give a recommendation."
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the research agent.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    text_llm_call = build_openai_text_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    agent_llm_call = build_openai_agent_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    graph = build_graph(text_llm_call, agent_llm_call)

    print("=" * 80)
    print("GRAPH (Mermaid)")
    print("=" * 80)
    print_mermaid_diagram(graph)

    result = graph.invoke(
        initial_state(question),
        config={"recursion_limit": MAX_STEPS * 4},
    )

    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)
    print(result)

    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
