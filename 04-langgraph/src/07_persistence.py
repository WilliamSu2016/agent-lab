"""LangGraph Persistence for the Research Agent.

Adds a **checkpointer** to the Research Agent graph (same shape as
``src/05_research_agent.py``: ``planner -> agent <-> tools -> finalize``) so
that its ``State`` is durably saved after every super-step, keyed by a
**thread id**.

Graph (unchanged from experiment 5):

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

What's new in this experiment:

* ``build_graph(..., checkpointer=...)`` is compiled with a checkpointer
  (``langgraph.checkpoint.memory.InMemorySaver`` -- an in-memory
  implementation of LangGraph's current, recommended checkpointer
  interface; no Redis/Postgres/external DB).
* Every ``graph.invoke(...)`` call is made with
  ``config={"configurable": {"thread_id": "..."}}``. The thread id is the
  checkpointer's partition key: each distinct thread id gets its own,
  completely isolated sequence of saved State snapshots ("checkpoints").
* ``graph.get_state(config)`` reads back the latest checkpoint for a thread
  without re-running anything.
* Compiling with ``interrupt_before=["tools"]`` pauses the graph right
  before the ``tools`` Node runs. The paused State is a checkpoint like any
  other -- resuming is just ``graph.invoke(None, config)`` again: LangGraph
  loads the last checkpoint for that thread id and continues the graph from
  there, it does not restart from ``START``.

Design constraints (same family as experiment 5): a real LLM is used via
plain ``openai`` calls (never the ``openai-agents`` SDK, never a LangChain
Agent abstraction), and the only "memory" mechanism at play is LangGraph's
own checkpointer -- no separate conversation-memory layer is introduced.

See ``docs/07-PERSISTENCE.md`` for the full walkthrough.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

MAX_STEPS = 6


# ---------------------------------------------------------------------------
# LLM call seams -- identical shape to src/05_research_agent.py.
# ---------------------------------------------------------------------------


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]


class AgentDecision(TypedDict, total=False):
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
# State -- exactly what gets checkpointed after every super-step.
# ---------------------------------------------------------------------------


class ToolResult(TypedDict):
    query: str
    result: dict[str, Any]


class ResearchAgentState(TypedDict):
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


# ---------------------------------------------------------------------------
# Prompts (unchanged from experiment 5)
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
# Nodes (same responsibilities as experiment 5 -- Persistence changes how the
# graph is *compiled and invoked*, not what each Node does)
# ---------------------------------------------------------------------------


def make_planner_node(text_llm_call: TextLLMCall):
    def planner(state: ResearchAgentState) -> ResearchAgentState:
        plan = text_llm_call(PLANNER_INSTRUCTIONS, state["question"])
        seeded_messages = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT_TEMPLATE.format(question=state["question"], plan=plan),
            },
            {"role": "user", "content": state["question"]},
        ]
        return {"plan": plan, "messages": seeded_messages}  # type: ignore[typeddict-item]

    return planner


def make_agent_node(agent_llm_call: AgentLLMCall, *, max_steps: int = MAX_STEPS):
    def agent(state: ResearchAgentState) -> ResearchAgentState:
        steps = state["steps"] + 1

        if steps > max_steps:
            return {
                "steps": steps,
                "pending_tool_call": None,
                "messages": state["messages"]
                + [{"role": "assistant", "content": f"Reached the maximum of {max_steps} step(s); stopping research."}],
            }  # type: ignore[typeddict-item]

        decision = agent_llm_call(state["messages"])

        if "tool_call" in decision:
            tool_call = decision["tool_call"]
            assistant_message = {"role": "assistant", "content": None, "tool_call": tool_call}
            return {
                "messages": state["messages"] + [assistant_message],
                "pending_tool_call": tool_call,
                "steps": steps,
            }  # type: ignore[typeddict-item]

        reasoning = decision.get("reasoning", "Research considered sufficient.")
        assistant_message = {"role": "assistant", "content": reasoning}
        return {
            "messages": state["messages"] + [assistant_message],
            "pending_tool_call": None,
            "steps": steps,
        }  # type: ignore[typeddict-item]

    return agent


def make_tools_node(search_web_call: SearchWebCall = _web_search):
    def tools(state: ResearchAgentState) -> ResearchAgentState:
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

        return {
            "messages": state["messages"] + [tool_message],
            "tool_results": state["tool_results"] + [tool_result],
            "pending_tool_call": None,
        }  # type: ignore[typeddict-item]

    return tools


def _format_tool_results(tool_results: list[ToolResult]) -> str:
    if not tool_results:
        return "(no web searches were performed)"
    sections = []
    for item in tool_results:
        sections.append(f"Query: {item['query']}\nResult: {json.dumps(item['result'])}")
    return "\n\n".join(sections)


def make_finalize_node(text_llm_call: TextLLMCall):
    def finalize(state: ResearchAgentState) -> ResearchAgentState:
        user_prompt = (
            f"Original question: {state['question']}\n\n"
            f"Research plan:\n{state['plan']}\n\n"
            f"Search results gathered:\n{_format_tool_results(state['tool_results'])}"
        )
        final_answer = text_llm_call(FINALIZE_INSTRUCTIONS, user_prompt)
        return {"final_answer": final_answer}  # type: ignore[typeddict-item]

    return finalize


def should_continue(state: ResearchAgentState) -> Literal["tools", "finalize"]:
    if state["pending_tool_call"] is not None:
        return "tools"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph assembly -- the only part that changes for Persistence: ``compile()``
# now takes a ``checkpointer`` (and, optionally, ``interrupt_before``).
# ---------------------------------------------------------------------------


def build_graph(
    text_llm_call: TextLLMCall,
    agent_llm_call: AgentLLMCall,
    search_web_call: SearchWebCall = _web_search,
    *,
    max_steps: int = MAX_STEPS,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
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
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def build_memory_checkpointer():
    """LangGraph's current, recommended in-memory checkpointer.

    ``InMemorySaver`` stores every checkpoint in a plain Python dict, keyed
    by thread id (and, within a thread, by checkpoint id). It requires no
    external service (no Redis, no Postgres, no file system) -- state is
    lost when the process exits, which is exactly the right trade-off for
    this experiment (Persistence *mechanism* first; a durable backend is a
    drop-in ``checkpointer=`` swap later, e.g. ``PostgresSaver``).
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def thread_config(thread_id: str) -> dict[str, Any]:
    """Build the ``config`` LangGraph needs to route calls to one thread's
    checkpoint history. ``thread_id`` is the partition key for persistence:
    every ``invoke``/``get_state`` call using the same ``thread_id`` reads
    and writes the same, isolated sequence of checkpoints."""
    return {"configurable": {"thread_id": thread_id}}


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the research agent.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    text_llm_call = build_openai_text_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    agent_llm_call = build_openai_agent_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)

    checkpointer = build_memory_checkpointer()
    graph = build_graph(text_llm_call, agent_llm_call, checkpointer=checkpointer)

    # Thread A: one independent research task.
    config_a = thread_config("thread-a")
    result_a = graph.invoke(
        initial_state("Compare Python and Go for building AI agents."),
        config=config_a,
    )
    print("=" * 80)
    print("THREAD A -- final answer")
    print("=" * 80)
    print(result_a["final_answer"])

    # Thread B: a second, independent research task -- its own checkpoint history.
    config_b = thread_config("thread-b")
    result_b = graph.invoke(
        initial_state("Compare React and Vue for building web frontends."),
        config=config_b,
    )
    print("=" * 80)
    print("THREAD B -- final answer")
    print("=" * 80)
    print(result_b["final_answer"])

    # Prove isolation: reading Thread A's saved state back still shows
    # Thread A's own question, untouched by Thread B's run.
    state_a = graph.get_state(config_a)
    print("=" * 80)
    print("THREAD A -- state re-read from the checkpointer")
    print("=" * 80)
    print(state_a.values["question"])


if __name__ == "__main__":
    main()
