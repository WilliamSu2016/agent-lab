"""Minimal, deterministic Evaluation Harness for the research agent.

This is intentionally NOT an agent framework. It is a small, data-driven test
runner:

- Test cases live in ``tests/evaluations/cases.json`` (plain data, no code).
- Each case scripts a fake LLM and a fake ``web_search`` deterministically, so
  the real ``run_agent`` loop (``src/agent.py``) is exercised exactly as in
  production, without any real network or LLM API calls.
- Assertions are plain Python ``assert`` statements / comparisons -- no
  LLM-as-a-judge, no fuzzy scoring.

Usage:
    python -m tests.evaluations.run_evaluations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent import run_agent  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.json"


class CaseWebSearchError(Exception):
    """Raised by the scripted web_search stand-in to simulate a tool failure."""


class ScriptedLLM:
    """A fake LLM client that replays a fixed sequence of responses.

    Each entry in ``turns`` describes one LLM response (``assistant``), with
    an optional list of ``tool_calls``. If the script runs out of scripted
    turns before the agent stops (e.g. the max-turns test case), the last
    scripted turn is repeated so the loop keeps requesting a tool and the
    agent is forced to hit its turn limit.
    """

    def __init__(self, turns: list[dict[str, Any]]):
        self._turns = turns
        self._index = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        turn = self._turns[min(self._index, len(self._turns) - 1)]
        self._index += 1
        assistant = turn["assistant"]
        tool_calls = [
            {
                "id": tc["id"],
                "function": {
                    "name": tc.get("name", "web_search"),
                    "arguments": json.dumps({"query": tc["query"]}),
                },
            }
            for tc in assistant.get("tool_calls", [])
        ]
        return {"content": assistant.get("content"), "tool_calls": tool_calls}


def build_scripted_web_search(case: dict[str, Any]):
    """Build a deterministic web_search stand-in from a case's scripted tool outcomes.

    Outcomes are keyed by query text, collected from every scripted turn in
    the case. A query mapped to an "error" outcome raises an exception (to
    simulate a tool failure); a query mapped to a "result" outcome returns
    that value directly (to simulate success, including empty results).
    """
    outcomes: dict[str, dict[str, Any]] = {}
    for turn in case["turns"]:
        for tool_call in turn["assistant"].get("tool_calls", []):
            outcomes[tool_call["query"]] = tool_call

    def _web_search(query: str) -> dict[str, Any]:
        outcome = outcomes.get(query)
        if outcome is None:
            raise CaseWebSearchError(f"Unscripted query encountered: {query!r}")
        if "error" in outcome:
            raise CaseWebSearchError(outcome["error"])
        return outcome["result"]

    return _web_search


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one case through the real agent loop and check deterministic properties.

    Returns a report dict with ``passed`` (bool) and ``checks`` (list of
    per-assertion pass/fail detail), plus the raw agent ``state`` for
    debugging failures.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    llm = ScriptedLLM(case["turns"])
    web_search = build_scripted_web_search(case)
    max_turns = case["max_turns"]
    expected = case["expected_tool_usage"]
    expected_behavior = case["expected_behavior"]

    # --- 1. Program runs successfully (no unhandled exception). ---
    error: Exception | None = None
    state: dict[str, Any] | None = None
    try:
        state = run_agent(case["input"], llm, web_search, max_turns=max_turns)
    except Exception as exc:  # pragma: no cover - would indicate a real bug
        error = exc

    check("agent_ran_without_exception", error is None, "" if error is None else repr(error))
    if state is None:
        return {"id": case["id"], "passed": False, "checks": checks, "state": None}

    # --- 2. Never exceeds max_turns. ---
    check(
        "turns_within_limit",
        state["turns"] <= max_turns,
        f"turns={state['turns']} max_turns={max_turns}",
    )

    # --- 3. Stops (reaches a terminal status; the loop does not hang). ---
    terminal_statuses = {"completed", "limit_reached", "failed"}
    check(
        "agent_stopped_with_terminal_status",
        state["status"] in terminal_statuses,
        f"status={state['status']!r}",
    )

    # --- 4. Correct high-level behavior / final-answer expectations. ---
    if expected_behavior == "limit_reached":
        check(
            "status_matches_expected_behavior",
            state["status"] == "limit_reached",
            f"status={state['status']!r}",
        )
        check(
            "turns_equal_max_turns_on_limit",
            state["turns"] == max_turns,
            f"turns={state['turns']} max_turns={max_turns}",
        )
        check(
            "final_answer_present_on_limit",
            isinstance(state["final_answer"], str) and bool(state["final_answer"].strip()),
        )
    else:
        check(
            "status_matches_expected_behavior",
            state["status"] == "completed",
            f"status={state['status']!r}",
        )
        check(
            "final_answer_produced",
            isinstance(state["final_answer"], str) and bool(state["final_answer"].strip()),
            f"final_answer={state['final_answer']!r}",
        )

    # --- 5. Correct tool usage (count, tool names, error handling). ---
    tool_calls = state["tool_calls"]
    call_count = len(tool_calls)
    min_calls = expected.get("min_calls", 0)
    max_calls = expected.get("max_calls", min_calls)
    check(
        "tool_call_count_within_expected_range",
        min_calls <= call_count <= max_calls,
        f"calls={call_count} expected=[{min_calls},{max_calls}]",
    )

    expected_tools = set(expected.get("tools", []))
    used_tools = {tc["name"] for tc in tool_calls}
    check(
        "tool_names_match_expected",
        used_tools <= expected_tools if expected_tools else not used_tools,
        f"used={sorted(used_tools)} expected_subset_of={sorted(expected_tools)}",
    )

    expect_tool_error = expected.get("expect_tool_error", False)
    has_tool_error = any(tc["status"] == "error" for tc in tool_calls)
    check(
        "tool_error_handling_matches_expectation",
        has_tool_error == expect_tool_error,
        f"has_tool_error={has_tool_error} expected={expect_tool_error}",
    )

    if expect_tool_error and "expected_error_code" in expected:
        error_codes = {
            tc["error"]["code"] for tc in tool_calls if tc["status"] == "error"
        }
        check(
            "tool_error_code_matches_expected",
            expected["expected_error_code"] in error_codes,
            f"error_codes={sorted(error_codes)} expected={expected['expected_error_code']!r}",
        )

    if not expected.get("allow_empty_results", False):
        for tc in tool_calls:
            if tc["status"] == "success":
                results = tc["content"].get("results")
                check(
                    f"tool_result_non_empty_{tc['tool_call_id']}",
                    bool(results),
                    f"results={results!r}",
                )

    passed = all(c["passed"] for c in checks)
    return {"id": case["id"], "passed": passed, "checks": checks, "state": state}


def run_all(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return [evaluate_case(case) for case in load_cases(path)]


def print_report(results: list[dict[str, Any]]) -> bool:
    all_passed = True
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        if not result["passed"]:
            all_passed = False
            for c in result["checks"]:
                if not c["passed"]:
                    print(f"    - {c['name']}: {c['detail']}")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{total} cases passed")
    return all_passed


def main() -> int:
    results = run_all()
    ok = print_report(results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
