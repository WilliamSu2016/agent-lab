"""CLI entry point for the Multi-Agent Research System.

Usage (from the project root, with ``.env`` providing ``OPENAI_API_KEY`` and
``OPENAI_MODEL`` -- see ``.env.example`` and ``docs/01-CONFIGURATION.md``):

    .\\.venv\\Scripts\\python.exe -m src.multi_agent_research.main "<question>"

Configuration & Secrets: this module never reads ``os.environ`` directly and
never hard-codes an API key, model name, timeout, or limit. All of that comes
from ``config.load_settings()``, the single place secrets are read from the
environment; a missing/invalid required setting (including the API key)
fails immediately with a clear ``ConfigurationError`` instead of an obscure
failure deep inside an HTTP client.
"""

from __future__ import annotations

import sys

from src.multi_agent_research.graph import (
    MultiAgentResearchTimeoutError,
    build_graph,
    print_mermaid_diagram,
    run_multi_agent_research,
)


def main() -> None:
    from dotenv import load_dotenv

    # Loads a local .env into the process environment, if present. This is
    # the *only* place a .env file is read; config.load_settings() itself
    # never touches the filesystem, only os.environ.
    load_dotenv()

    from config import ConfigurationError, load_settings

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    question = " ".join(sys.argv[1:]).strip() or (
        "2026 年 AI Agent 开发生态有哪些值得 Solo Developer 关注的机会？"
    )

    from langgraph.checkpoint.memory import InMemorySaver

    from src.specialists.llm import build_openai_text_llm_call

    text_llm_call = build_openai_text_llm_call(settings)
    checkpointer = InMemorySaver()

    graph = build_graph(text_llm_call, text_llm_call, text_llm_call, text_llm_call, checkpointer=checkpointer)

    print("=" * 80)
    print(f"ENVIRONMENT: {settings.environment}  MODEL: {settings.model_name}")
    print("=" * 80)
    print("GRAPH (Mermaid)")
    print("=" * 80)
    print_mermaid_diagram(graph)

    try:
        outcome = run_multi_agent_research(
            question,
            text_llm_call,
            text_llm_call,
            text_llm_call,
            text_llm_call,
            max_workers=settings.limits.max_workers,
            max_iterations=settings.limits.max_iterations,
            per_worker_timeout_seconds=settings.limits.worker_timeout_seconds,
            checkpointer=checkpointer,
            run_timeout_seconds=settings.limits.run_timeout_seconds,
        )
    except MultiAgentResearchTimeoutError as exc:
        raise SystemExit(str(exc)) from exc

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
