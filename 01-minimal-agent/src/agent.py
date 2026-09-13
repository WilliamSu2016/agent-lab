from __future__ import annotations

import json
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

from src.llm_client import LLMError
from src.prompts import SYSTEM_PROMPT
from src.tools import WEB_SEARCH_TOOL


class LLMClient(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


Tool = Callable[[str], dict[str, Any]]


def run_agent(
    question: str,
    llm: LLMClient,
    web_search: Tool,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Run one bounded research loop and return its answer and execution state."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    state: dict[str, Any] = {
        "run_id": str(uuid4()),
        "status": "running",
        "question": question,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "turns": 0,
        "tool_calls": [],
        "errors": [],
        "final_answer": None,
    }
    print(f"[agent] run={state['run_id']} started")

    for turn in range(1, max_turns + 1):
        state["turns"] = turn
        print(f"[agent] turn {turn}/{max_turns}: asking LLM")
        response = _call_llm(llm, state["messages"], state["errors"])
        if response is None:
            state["status"] = "failed"
            state["final_answer"] = "Unable to get a response from the LLM API."
            print("[agent] stopping: LLM API failed")
            return state

        tool_calls = response.get("tool_calls") or []
        content = response.get("content")
        if not tool_calls:
            if not isinstance(content, str) or not content.strip():
                state["status"] = "failed"
                state["errors"].append("LLM returned neither a tool call nor a final answer.")
                state["final_answer"] = "The LLM returned an invalid response."
                print("[agent] stopping: invalid LLM response")
                return state

            state["messages"].append({"role": "assistant", "content": content})
            state["status"] = "completed"
            state["final_answer"] = content
            print("[agent] stopping: received final answer")
            return state

        state["messages"].append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )
        for tool_call in tool_calls:
            tool_result = _execute_web_search(tool_call, web_search)
            state["tool_calls"].append(tool_result)
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
            print(f"[agent] tool result: {tool_result['status']}")

    state["status"] = "limit_reached"
    state["final_answer"] = (
        "Research stopped because the maximum number of LLM turns was reached."
    )
    print("[agent] stopping: maximum turns reached")
    return state


def _call_llm(
    llm: LLMClient, messages: list[dict[str, Any]], errors: list[str]
) -> dict[str, Any] | None:
    for attempt in range(1, 3):
        try:
            return llm.complete(messages, [WEB_SEARCH_TOOL])
        except LLMError as error:
            errors.append(str(error))
            print(f"[agent] LLM API error (attempt {attempt}/2): {error}")
            if attempt == 1:
                time.sleep(1)
    return None


def _execute_web_search(tool_call: dict[str, Any], web_search: Tool) -> dict[str, Any]:
    call_id = tool_call.get("id")
    function = tool_call.get("function", {})
    name = function.get("name")

    if not isinstance(call_id, str) or not call_id:
        return _tool_error("", "INVALID_TOOL_CALL", "Tool call is missing an id.")
    if name != "web_search":
        return _tool_error(call_id, "UNKNOWN_TOOL", f"Tool '{name}' is unavailable.")

    try:
        arguments = json.loads(function.get("arguments", "{}"))
    except (TypeError, json.JSONDecodeError):
        return _tool_error(call_id, "INVALID_ARGUMENTS", "Tool arguments must be valid JSON.")

    if not isinstance(arguments, dict):
        return _tool_error(
            call_id, "INVALID_ARGUMENTS", "Tool arguments must be a JSON object."
        )

    query = arguments.get("query")
    if set(arguments) != {"query"} or not isinstance(query, str) or not query.strip():
        return _tool_error(
            call_id, "INVALID_ARGUMENTS", "web_search requires exactly one non-empty query."
        )

    print(f"[agent] executing web_search: {query}")
    try:
        result = web_search(query)
    except Exception as error:
        print(f"[agent] web_search failed: {error}")
        return _tool_error(call_id, "TOOL_EXECUTION_FAILED", str(error))

    return {
        "tool_call_id": call_id,
        "name": "web_search",
        "status": "success",
        "content": result,
    }


def _tool_error(tool_call_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call_id,
        "name": "web_search",
        "status": "error",
        "error": {"code": code, "message": message},
    }
