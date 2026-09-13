"""Human-in-the-loop for the LangGraph Research Agent.

Builds on ``src/07_persistence.py``: the same
``planner -> agent <-> tools -> generator`` research pipeline, plus a
high-risk ``publish_report`` tool that must NEVER be called automatically.

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
      `-- "generate" --> generator  <---------------.
                              |                      |
                              v                      |
                        human_review                 |
                              |                      |
                    route_after_review                |
                       |-- "approve" --> publish --> END
                       |-- "reject"  --> END
                       `-- "request_changes" --------'

``generator`` drafts a Research Report from everything gathered so far.
Before the report can ever reach ``publish_report`` (the high-risk tool),
the graph MUST pass through ``human_review``, where it calls LangGraph's
``interrupt()`` -- execution genuinely pauses, mid-graph, and the current
State is checkpointed. A human (or, in tests, a scripted stand-in) then
resumes the graph with one of three decisions:

* ``approve``          -> ``publish`` runs the high-risk tool -> END
* ``reject``            -> END, the report is discarded, nothing is published
* ``request_changes``   -> back to ``generator`` with the human's feedback,
                            which drafts a revision and routes through
                            ``human_review`` again

This requires all three pieces working together:

* **Graph interrupt** -- ``langgraph.types.interrupt(payload)`` inside the
  ``human_review`` Node.
* **Persistence** -- a checkpointer is mandatory: without one, LangGraph
  raises immediately, because a paused graph has nowhere to save its State.
* **Resume** -- ``graph.invoke(Command(resume=decision), config)`` reads the
  checkpoint for that ``thread_id`` and continues exactly where it paused.

See ``docs/08-HUMAN-IN-THE-LOOP.md`` for the full walkthrough.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

from typing_extensions import TypedDict

from src.tools import web_search as _web_search

MAX_STEPS = 6


# ---------------------------------------------------------------------------
# LLM / tool call seams -- same shape as experiments 5 and 7.
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
PublishReportCall = Callable[[str], dict[str, Any]]

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


def fake_publish_report(report: str) -> dict[str, Any]:
    """A stand-in for a real "publish to the world" side effect (e.g. posting
    to a CMS, sending an external email, writing to a public S3 bucket).
    Deliberately trivial -- the point of this experiment is that it must
    never be reachable without going through ``human_review`` first."""
    return {"status": "published", "length": len(report)}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ToolResult(TypedDict):
    query: str
    result: dict[str, Any]


ReviewDecision = Literal["approve", "reject", "request_changes"]


class HumanReviewState(TypedDict):
    question: str
    plan: str
    messages: list[dict[str, Any]]
    pending_tool_call: ToolCall | None
    tool_results: list[ToolResult]
    steps: int
    report: str
    review_feedback: list[str]
    review_decision: ReviewDecision | None
    published: bool
    publish_result: dict[str, Any] | None
    final_answer: str


def initial_state(question: str) -> HumanReviewState:
    return {
        "question": question,
        "plan": "",
        "messages": [],
        "pending_tool_call": None,
        "tool_results": [],
        "steps": 0,
        "report": "",
        "review_feedback": [],
        "review_decision": None,
        "published": False,
        "publish_result": None,
        "final_answer": "",
    }


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
evidence. Once you believe you have gathered enough evidence, stop calling tools --
simply respond with a short note on why the research is sufficient (do not write the
final report yourself, another step does that)."""

GENERATOR_INSTRUCTIONS = """You are the Generator. Draft a Research Report answering the
original question, based on the plan and the web search results gathered. If you are
given previous human reviewer feedback, revise the report to explicitly address every
point raised rather than starting over."""


def _format_tool_results(tool_results: list[ToolResult]) -> str:
    if not tool_results:
        return "(no web searches were performed)"
    return "\n\n".join(f"Query: {item['query']}\nResult: {json.dumps(item['result'])}" for item in tool_results)


def _build_generator_prompt(state: HumanReviewState) -> str:
    base = (
        f"Original question: {state['question']}\n\n"
        f"Research plan:\n{state['plan']}\n\n"
        f"Search results gathered:\n{_format_tool_results(state['tool_results'])}"
    )
    if not state["review_feedback"]:
        return base
    feedback_text = "\n".join(f"- {item}" for item in state["review_feedback"])
    return (
        f"{base}\n\nPrevious draft:\n{state['report']}\n\n"
        f"Human reviewer feedback to address:\n{feedback_text}"
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def make_planner_node(text_llm_call: TextLLMCall):
    def planner(state: HumanReviewState) -> HumanReviewState:
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
    def agent(state: HumanReviewState) -> HumanReviewState:
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
    def tools(state: HumanReviewState) -> HumanReviewState:
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


def should_continue(state: HumanReviewState) -> Literal["tools", "generate"]:
    if state["pending_tool_call"] is not None:
        return "tools"
    return "generate"


def make_generator_node(text_llm_call: TextLLMCall):
    """Drafts (or revises) the Research Report. Owns ``report`` only -- it
    never decides whether the report gets published; that decision belongs
    entirely to ``human_review``."""

    def generator(state: HumanReviewState) -> HumanReviewState:
        report = text_llm_call(GENERATOR_INSTRUCTIONS, _build_generator_prompt(state))
        return {"report": report}  # type: ignore[typeddict-item]

    return generator


def make_human_review_node():
    """The mandatory checkpoint before the high-risk ``publish_report`` tool
    can ever run.

    Calls ``interrupt(...)`` with the report awaiting review. This *pauses
    graph execution right here* -- LangGraph raises a ``GraphInterrupt``
    internally, which unwinds the current ``invoke()`` call back to the
    caller, but NOT before the checkpointer has saved the State as it stood
    at this point (this Node has not returned an update yet). Resuming later
    with ``Command(resume=<decision>)`` re-enters this exact Node from the
    top; ``interrupt()`` then returns the human's decision instead of
    pausing again.
    """

    def human_review(state: HumanReviewState) -> HumanReviewState:
        from langgraph.types import interrupt

        decision_input = interrupt(
            {
                "kind": "publish_report_approval",
                "question": state["question"],
                "report": state["report"],
                "review_round": len(state["review_feedback"]),
            }
        )

        decision: ReviewDecision = decision_input["decision"]
        if decision not in ("approve", "reject", "request_changes"):
            raise ValueError(f"Unknown review decision: {decision!r}")

        update: HumanReviewState = {"review_decision": decision}  # type: ignore[typeddict-item]
        if decision == "request_changes":
            comments = decision_input.get("comments", "")
            update["review_feedback"] = state["review_feedback"] + [comments]  # type: ignore[typeddict-item]
        return update

    return human_review


def route_after_review(state: HumanReviewState) -> Literal["publish", "end", "generate"]:
    decision = state["review_decision"]
    if decision == "approve":
        return "publish"
    if decision == "reject":
        return "end"
    if decision == "request_changes":
        return "generate"
    raise ValueError(f"human_review must set review_decision before routing, got {decision!r}")


def make_publish_node(publish_report_call: PublishReportCall = fake_publish_report):
    """The high-risk tool itself. Only reachable via the ``"approve"`` branch
    of ``route_after_review`` -- there is no edge from anywhere else in the
    graph into this Node."""

    def publish(state: HumanReviewState) -> HumanReviewState:
        result = publish_report_call(state["report"])
        return {
            "published": True,
            "publish_result": result,
            "final_answer": state["report"],
        }  # type: ignore[typeddict-item]

    return publish


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(
    text_llm_call: TextLLMCall,
    agent_llm_call: AgentLLMCall,
    search_web_call: SearchWebCall = _web_search,
    publish_report_call: PublishReportCall = fake_publish_report,
    *,
    max_steps: int = MAX_STEPS,
    checkpointer=None,
):
    from langgraph.graph import END, START, StateGraph

    graph_builder = StateGraph(HumanReviewState)

    graph_builder.add_node("planner", make_planner_node(text_llm_call))
    graph_builder.add_node("agent", make_agent_node(agent_llm_call, max_steps=max_steps))
    graph_builder.add_node("tools", make_tools_node(search_web_call))
    graph_builder.add_node("generator", make_generator_node(text_llm_call))
    graph_builder.add_node("human_review", make_human_review_node())
    graph_builder.add_node("publish", make_publish_node(publish_report_call))

    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "generate": "generator"},
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("generator", "human_review")
    graph_builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"publish": "publish", "end": END, "generate": "generator"},
    )
    graph_builder.add_edge("publish", END)

    return graph_builder.compile(checkpointer=checkpointer)


def build_memory_checkpointer():
    """Same in-memory checkpointer as experiment 7 -- ``interrupt()``
    requires one; without it, ``invoke`` raises immediately (see
    ``docs/08-HUMAN-IN-THE-LOOP.md`` Q1)."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def main() -> None:
    import os

    from dotenv import load_dotenv
    from langgraph.types import Command

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

    config = thread_config("cli-run")
    result = graph.invoke(
        initial_state("Compare Python, TypeScript, and Go for building AI agents."),
        config=config,
    )

    state = graph.get_state(config)
    if state.next == ("human_review",):
        print("=" * 80)
        print("PAUSED FOR HUMAN REVIEW -- draft report:")
        print("=" * 80)
        print(result["report"])

        raw = input("\nApprove, reject, or request_changes? [approve/reject/request_changes]: ").strip()
        comments = ""
        if raw == "request_changes":
            comments = input("Feedback for the generator: ").strip()
        result = graph.invoke(Command(resume={"decision": raw, "comments": comments}), config=config)

    print("=" * 80)
    print("FINAL STATE")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()
