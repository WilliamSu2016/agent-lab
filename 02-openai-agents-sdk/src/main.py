"""CLI entry point: User -> Agent -> Runner -> Final Answer.

Reads the question from argv, loads ``.env``, requires
``OPENAI_API_KEY``/``OPENAI_MODEL``, and prints the final answer. All Agent
Runtime behavior (the LLM loop, tool dispatch, tool-result routing,
continuation/termination) is owned by the OpenAI Agents SDK's ``Runner`` --
this file only wires up the application entry point.
"""

import os
import sys

from dotenv import load_dotenv

from src.agent import build_agent, build_model, run_agent
from src.tracing import enable_local_tracing


def main() -> None:
    load_dotenv()
    enable_local_tracing()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python -m src.main "Your research question"')

    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL")
    if not api_key or not model_name:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the agent.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url)
    agent = build_agent(model)

    final_answer = run_agent(question, agent, max_turns=5)
    print(f"\n{final_answer}")


if __name__ == "__main__":
    main()
