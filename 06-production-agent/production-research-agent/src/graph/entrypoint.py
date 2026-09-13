"""Module-level graph entrypoint for the LangGraph CLI's local development
server (``langgraph dev``, configured by ``langgraph.json``).

The CLI needs a plain module-level ``graph`` variable (it cannot call a
factory function that takes constructor arguments). ``langgraph dev``
supplies its own development-mode checkpointing for whatever graph it
loads, so this module intentionally does not open a competing SQLite
connection -- compare with the production wiring in ``src/api/main.py``,
which always binds the real ``src.graph.checkpointer.sqlite_checkpointer``.

If ``OPENAI_API_KEY``/``OPENAI_MODEL`` are not yet configured (e.g. a
fresh checkout, before ``.env`` is filled in), this module still imports
successfully -- the graph is built with a stub LLM call that returns an
explanatory placeholder instead of raising ``ConfigurationError`` at
import time, so ``langgraph dev`` can still start and show the graph
structure; a real run naturally still requires real credentials.
"""

from __future__ import annotations

from dotenv import load_dotenv

from src.config import ConfigurationError, load_settings
from src.graph.graph import build_graph

load_dotenv()

try:
    settings = load_settings()
    from src.agents.llm import build_openai_text_llm_call

    _llm_call = build_openai_text_llm_call(settings)
except ConfigurationError:

    def _llm_call(system_prompt: str, user_prompt: str) -> str:  # type: ignore[misc]
        return (
            "[dev-mode placeholder] OPENAI_API_KEY/OPENAI_MODEL are not configured; "
            "this graph was built without a real LLM. Set them in .env to run for real."
        )


graph = build_graph(_llm_call, _llm_call, _llm_call, _llm_call)

__all__ = ["graph"]
