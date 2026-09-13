"""Unit tests: the optional ``memory_store`` cache-hit short-circuit in
``src.graph.graph.run_production_research`` (Memory experiment: an exact
repeat of the same tenant's question is served from cache at zero
additional graph execution, never a substitute for real retrieval/RAG --
see ``src.memory.store``'s own module docstring)."""

from __future__ import annotations

import json

from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import run_production_research
from src.memory.store import SessionMemoryStore

IDENTITY = {"user_id": "u1", "tenant_id": "t1", "roles": ["researcher"]}


def _planner_llm(system: str, user: str) -> str:
    return json.dumps(["Research A"])


def _researcher_llm(system: str, user: str) -> str:
    return "Finding for A."


def _synth_llm(system: str, user: str) -> str:
    return "Synthesized report."


def _review_llm(system: str, user: str) -> str:
    return '{"approved": true, "completeness": "ok", "factual_consistency": "ok", "evidence_quality": "ok", "logical_consistency": "ok", "missing_aspects": [], "feedback": ""}'


def test_second_identical_question_is_served_from_memory_without_rerunning_the_graph(tmp_path):
    call_count = {"n": 0}

    def counting_researcher_llm(system: str, user: str) -> str:
        call_count["n"] += 1
        return "Finding for A."

    memory_store = SessionMemoryStore()

    with sqlite_checkpointer(tmp_path / "ckpt.sqlite3") as checkpointer:
        first = run_production_research(
            "What is LangGraph?", IDENTITY, _planner_llm, counting_researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer, memory_store=memory_store,
        )
        assert call_count["n"] == 1
        assert "Synthesized report" in first.final_answer

        second = run_production_research(
            "What is LangGraph?", IDENTITY, _planner_llm, counting_researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer, memory_store=memory_store,
        )
        # The researcher LLM was never invoked a second time -- the cached
        # answer was served directly.
        assert call_count["n"] == 1
        assert second.final_answer == first.final_answer
        assert second.tokens_used == 0
        assert second.cost_usd == 0.0


def test_memory_cache_is_isolated_per_tenant(tmp_path):
    memory_store = SessionMemoryStore()
    other_identity = {"user_id": "u2", "tenant_id": "t2", "roles": ["researcher"]}

    with sqlite_checkpointer(tmp_path / "ckpt.sqlite3") as checkpointer:
        run_production_research(
            "What is LangGraph?", IDENTITY, _planner_llm, _researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer, memory_store=memory_store,
        )
        # A different tenant asking the exact same question must NOT see
        # tenant t1's cached answer -- tenant isolation applies to memory
        # too, not just tool calls.
        assert memory_store.recall("t2", "What is LangGraph?") is None
        second = run_production_research(
            "What is LangGraph?", other_identity, _planner_llm, _researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer, memory_store=memory_store,
        )
        assert second.tokens_used > 0  # a real run happened, not a cache hit


def test_without_a_memory_store_every_call_runs_the_full_graph(tmp_path):
    call_count = {"n": 0}

    def counting_researcher_llm(system: str, user: str) -> str:
        call_count["n"] += 1
        return "Finding for A."

    with sqlite_checkpointer(tmp_path / "ckpt.sqlite3") as checkpointer:
        run_production_research(
            "What is LangGraph?", IDENTITY, _planner_llm, counting_researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer,
        )
        run_production_research(
            "What is LangGraph?", IDENTITY, _planner_llm, counting_researcher_llm, _synth_llm, _review_llm,
            checkpointer=checkpointer,
        )
        assert call_count["n"] == 2
