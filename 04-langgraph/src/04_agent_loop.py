"""A hand-written Agent Loop, rebuilt as a LangGraph ``StateGraph`` with a
Conditional Edge.

The pattern being replaced is the classic hand-written loop:

    while True:
        response = LLM(...)
        if tool_call:
            execute_tool()
            continue
        return final_answer

As a LangGraph:

    START
      |
      v
    agent
      |
      v
    should_continue (routing function)
      |-- "tools" --> tools --> agent   (tool result goes back to agent)
      `-- "end"   --> END

Design constraints for this experiment:

* ``StateGraph`` is used; ``agent`` and ``tools`` are Nodes.
* ``should_continue`` is a plain routing function (not a Node) wired in with
  ``add_conditional_edges``.
* The Tool result always flows back into ``agent`` via a normal Edge
  (``tools -> agent``) -- the loop closes through the graph, not through a
  Python ``while``.
* ``agent`` reaching a final answer is the only way to reach ``END``.
* A hard maximum step count prevents infinite looping even if the (fake or
  real) model never stops requesting tools.
* Tool errors are caught inside the ``tools`` Node and turned into a safe
  tool-result message fed back to ``agent`` -- they never crash the graph.
* NO ``openai-agents`` SDK and NO LangChain Agent abstraction: ``agent`` is a
  plain function that calls an injectable ``LLMCall``, and the "loop" is
  expressed purely as graph edges.

See ``docs/04-AGENT-LOOP.md`` for the conceptual explanation.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

MAX_STEPS = 5


# ---------------------------------------------------------------------------
# LLM call seam (same idea as experiment #3): a plain callable, not an Agent.
#
# Given the running message history, it returns EITHER a tool call request
# OR a final answer -- never both.
# ---------------------------------------------------------------------------


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]


class LLMDecision(TypedDict, total=False):
    tool_call: ToolCall
    final_answer: str


LLMCall = Callable[[list[dict[str, Any]]], LLMDecision]
SearchWebCall = Callable[[str], dict[str, Any]]


def build_openai_llm_call(api_key: str, model_name: str, base_url: str) -> LLMCall:
    """Build a real ``LLMCall`` using the plain ``openai`` Chat Completions API
    with function-calling -- no ``agents.Agent``/``Runner`` and no LangChain
    Agent abstraction are involved."""
    import json

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    search_tool_schema = {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current, factual information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    def call(messages: list[dict[str, Any]]) -> LLMDecision:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=[search_tool_schema],
        )
        choice = response.choices[0].message
        if choice.tool_calls:
            tool_call = choice.tool_calls[0]
            args = json.loads(tool_call.function.arguments or "{}")
            return {"tool_call": {"name": tool_call.function.name, "args": args}}
        return {"final_answer": choice.content or ""}

    return call


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Everything the loop needs, carried across every agent <-> tools hop."""

    question: str
    messages: list[dict[str, Any]]
    pending_tool_call: ToolCall | None
    final_answer: str
    steps: int


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "messages": [{"role": "user", "content": question}],
        "pending_tool_call": None,
        "final_answer": "",
        "steps": 0,
    }


def _print_state(label: str, state: AgentState) -> None:
    print(f"[{label}] {state}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_agent_node(llm_call: LLMCall, *, max_steps: int = MAX_STEPS):
    """Node factory for ``agent``: decide either a tool call or a final answer.

    Owns ``messages`` (appends the assistant turn), ``pending_tool_call``,
    ``final_answer``, and ``steps``. Never touches tool execution itself.
    """

    def agent(state: AgentState) -> AgentState:
        _print_state("agent: before", state)

        steps = state["steps"] + 1

        if steps > max_steps:
            # Requirement 8: hard cap so a misbehaving/looping model can never
            # keep this graph running forever.
            update: AgentState = {
                "steps": steps,
                "pending_tool_call": None,
                "final_answer": (
                    f"Stopped after reaching the maximum of {max_steps} step(s) "
                    "without a final answer."
                ),
            }  # type: ignore[typeddict-item]
            _print_state("agent: after (max steps reached)", {**state, **update})
            return update

        decision = llm_call(state["messages"])

        if "tool_call" in decision:
            tool_call = decision["tool_call"]
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_call": tool_call,
            }
            update = {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": tool_call,
                "final_answer": "",
                "steps": steps,
            }  # type: ignore[typeddict-item]
        else:
            final_answer = decision.get("final_answer", "")
            assistant_message = {"role": "assistant", "content": final_answer}
            update = {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": None,
                "final_answer": final_answer,
                "steps": steps,
            }  # type: ignore[typeddict-item]

        _print_state("agent: after", {**state, **update})
        return update

    return agent


def make_tools_node(search_web_call: SearchWebCall = _web_search):
    """Node factory for ``tools``: execute the pending tool call.

    Owns ``messages`` (appends the tool result) and clears
    ``pending_tool_call``. Requirement 9: any exception raised by the tool is
    caught here and turned into a safe error message fed back to ``agent`` --
    it never propagates out of the graph.
    """

    def tools(state: AgentState) -> AgentState:
        _print_state("tools: before", state)

        tool_call = state["pending_tool_call"]
        assert tool_call is not None, "tools node reached without a pending_tool_call"

        try:
            if tool_call["name"] != "search_web":
                raise ValueError(f"Unknown tool: {tool_call['name']!r}")
            result = search_web_call(tool_call["args"].get("query", ""))
            tool_message = {"role": "tool", "name": tool_call["name"], "content": result}
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any tool failure
            # must become a safe, recoverable message, not a crash.
            tool_message = {
                "role": "tool",
                "name": tool_call["name"],
                "content": {"error": str(exc)},
            }

        update: AgentState = {
            "messages": state["messages"] + [tool_message],
            "pending_tool_call": None,
        }  # type: ignore[typeddict-item]
        _print_state("tools: after", {**state, **update})
        return update

    return tools


# ---------------------------------------------------------------------------
# Routing function (NOT a Node): decides where to go after ``agent``.
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Requirement 4/5: a plain function of State, returning the next Edge label.

    Registered with ``add_conditional_edges`` -- LangGraph calls this after
    ``agent`` runs and routes to whichever Node the returned label maps to.
    """
    if state["pending_tool_call"] is not None:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    llm_call: LLMCall,
    search_web_call: SearchWebCall = _web_search,
    *,
    max_steps: int = MAX_STEPS,
):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", make_agent_node(llm_call, max_steps=max_steps))
    graph_builder.add_node("tools", make_tools_node(search_web_call))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph_builder.add_edge("tools", "agent")  # Requirement 6: tool result goes back to agent.

    # Requirement 8 (belt and suspenders): LangGraph's own recursion guard,
    # independent of the in-state `steps` counter checked inside `agent`.
    return graph_builder.compile()


def main() -> None:
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    question = " ".join(sys.argv[1:]).strip() or "What is LangGraph?"

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the agent loop.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    llm_call = build_openai_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    graph = build_graph(llm_call)

    result = graph.invoke(
        initial_state(question),
        config={"recursion_limit": MAX_STEPS * 4},
    )

    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()
