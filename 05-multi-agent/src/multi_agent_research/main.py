"""CLI entry point for the Multi-Agent Research System.

Usage (from ``05-multi-agent``, with ``.env`` providing ``OPENAI_API_KEY``
and ``OPENAI_MODEL``):

    .\\.venv\\Scripts\\python.exe -m src.multi_agent_research.main "<question>"
"""

from __future__ import annotations

import os
import sys

from src.multi_agent_research.graph import (
    build_graph,
    print_mermaid_diagram,
    run_multi_agent_research,
)
from src.multi_agent_research.state import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_WORKER_TIMEOUT_SECONDS,
)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    question = " ".join(sys.argv[1:]).strip() or (
        "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the pipeline.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    from langgraph.checkpoint.memory import InMemorySaver

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(api_key=api_key, model_name=model_name, base_url=base_url)
    checkpointer = InMemorySaver()

    graph = build_graph(text_llm_call, text_llm_call, text_llm_call, text_llm_call, checkpointer=checkpointer)

    print("=" * 80)
    print("GRAPH (Mermaid)")
    print("=" * 80)
    print_mermaid_diagram(graph)

    outcome = run_multi_agent_research(
        question,
        text_llm_call,
        text_llm_call,
        text_llm_call,
        text_llm_call,
        max_workers=DEFAULT_MAX_WORKERS,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        per_worker_timeout_seconds=DEFAULT_WORKER_TIMEOUT_SECONDS,
        checkpointer=checkpointer,
    )

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print(f"\nFINAL TASKS (iteration {outcome.iteration}):")
    for task in outcome.tasks:
        print(f"  - [{task['task_id']}] {task['aspect']}  ({task['reason']})")

    print("\nWORKER RESULTS (accumulated across all iterations):")
    for result in outcome.worker_results:
        print(f"  - [{result['status']}] {result['aspect']}")

    print(f"\nSYNTHESIS:\n{outcome.synthesis}")
    print(f"\nREVIEW:\n{outcome.review}")
    print(f"\nFINAL_ANSWER:\n{outcome.final_answer}")
    print(f"\nTOTAL ITERATIONS: {outcome.iteration}")

    print("\n" + "=" * 80)
    print("TRACE (from final state)")
    print("=" * 80)
    for line in outcome.trace:
        print(f"  - {line}")

    print("\n" + "=" * 80)
    print("STATE HISTORY (from checkpointer persistence)")
    print("=" * 80)
    config = {"configurable": {"thread_id": outcome.thread_id}}
    history = list(graph.get_state_history(config))
    for snapshot in reversed(history):
        writes = list((snapshot.metadata.get("writes") or {}).keys()) if snapshot.metadata else []
        step = snapshot.metadata.get("step") if snapshot.metadata else "?"
        print(f"  step={step} writes={writes}")


if __name__ == "__main__":
    main()
