import os
import sys

from dotenv import load_dotenv
from src.agent import run_agent
from src.llm_client import OpenAICompatibleClient
from src.tools import web_search


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python -m src.main "Your research question"')

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL before running the agent.")

    llm = OpenAICompatibleClient(
        api_key=api_key,
        model=model,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    state = run_agent(question, llm, web_search, max_turns=5)
    print(f"\n[{state['status']}] {state['final_answer']}")


if __name__ == "__main__":
    main()
