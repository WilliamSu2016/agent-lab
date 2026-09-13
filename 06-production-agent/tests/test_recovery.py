"""Tests for Durable Execution: crash -> checkpoint -> restart -> resume.

Scenario required by the experiment:

    Research A succeeds
    Research B succeeds
    Research C crashes

    After recovery:
        Research A is NOT re-executed
        Research B is NOT re-executed
        Research C resumes from the right place

Every test uses a real on-disk SQLite checkpointer file (never
``:memory:`` -- see ``src/durable/checkpointer.py``'s docstring for why
that would not actually exercise crash/restart semantics) so that
"restart" genuinely means "a brand-new graph object opens the same
checkpoint file", not "reuse the same in-process saver".
"""

from __future__ import annotations

import sqlite3

import pytest

from src.durable.checkpointer import sqlite_checkpointer
from src.durable.graph import build_durable_graph, initial_state
from src.durable.recovery import (
    CrashInjector,
    WorkerCrash,
    get_execution_history,
    get_pending_tasks,
    resume,
    run_or_crash,
)
from src.reliability.idempotency import InMemoryIdempotencyStore


class CountingSideEffect:
    """A side-effect stand-in that records both how many times it was
    *attempted* (``calls``) and -- since it is always wrapped in
    ``reliability.idempotency.idempotent`` by ``graph.py`` -- relies on the
    idempotency store to prevent a replayed node from actually incrementing
    ``calls`` a second time for the same task_id."""

    def __init__(self):
        self.calls: dict[str, int] = {"A": 0, "B": 0, "C": 0}

    def for_task(self, task_id: str):
        def _effect(*, task_id: str, content: str) -> dict[str, str]:
            self.calls[task_id] += 1
            return {"task_id": task_id, "findings": content}

        return _effect

    def as_side_effects(self) -> dict[str, object]:
        return {task_id: self.for_task(task_id) for task_id in ("A", "B", "C")}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "durable_checkpoints.sqlite3"


class TestNoCrashBaseline:
    def test_full_run_completes_and_calls_each_side_effect_exactly_once(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            outcome = run_or_crash(graph, initial_state("why is the sky blue"), thread_id="baseline")

        assert outcome.status == "completed"
        assert effects.calls == {"A": 1, "B": 1, "C": 1}
        assert "Task A" in outcome.final_report
        assert "Task C" in outcome.final_report


class TestCrashDuringResearchC:
    """The exact scenario from the requirement: A succeeds, B succeeds, C
    crashes -- exercised twice, once where the crash happens *before* C's
    side effect ever runs, and once where it happens *after* (idempotency
    is only actually load-bearing in the second case)."""

    def test_crash_before_c_side_effect_then_resume(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="before")

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(
                store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=checkpointer
            )
            outcome = run_or_crash(graph, initial_state("q"), thread_id="crash-before")

            assert outcome.status == "crashed"
            assert isinstance(outcome.crash, WorkerCrash)
            # A and B already ran for real; C never got the chance to.
            assert effects.calls == {"A": 1, "B": 1, "C": 0}
            # The checkpoint knows exactly where execution is stuck.
            assert get_pending_tasks(graph, "crash-before") == ("research_c",)

        # "Restart": a brand-new graph object, opening the same on-disk
        # checkpoint file -- not the same Python object the crash happened on.
        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(
                store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=checkpointer2
            )
            resumed = resume(graph2, thread_id="crash-before")

            assert resumed.status == "completed"
            # A and B were NOT re-executed; only C ran (for the first time).
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
            assert resumed.results["A"] == resumed.results["A"]  # sanity: present
            assert set(resumed.results.keys()) == {"A", "B", "C"}
            assert get_pending_tasks(graph2, "crash-before") == ()

    def test_crash_after_c_side_effect_then_resume_does_not_duplicate_side_effect(self, db_path):
        """Crash happens *after* C's real side effect already executed but
        before the graph committed the result. On resume, LangGraph must
        replay node C from scratch (it never got a checkpoint), but the
        idempotency store must prevent the real side effect from firing
        twice -- this is requirement 7 ("不重复执行已经成功且具有副作用的操作")
        composed with checkpoint-based recovery."""
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="after")

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(
                store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=checkpointer
            )
            outcome = run_or_crash(graph, initial_state("q"), thread_id="crash-after")

            assert outcome.status == "crashed"
            # C's side effect DID run once already, even though the crash
            # happened before the graph could record it.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
            assert get_pending_tasks(graph, "crash-after") == ("research_c",)

        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(
                store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=checkpointer2
            )
            resumed = resume(graph2, thread_id="crash-after")

            assert resumed.status == "completed"
            # The node replayed, but the real side effect was NOT invoked
            # again -- the idempotency store returned the cached result.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
            assert set(resumed.results.keys()) == {"A", "B", "C"}

    def test_a_and_b_checkpoints_are_committed_before_c_crashes(self, db_path):
        """Directly verifies the checkpoint-level claim (not just the side
        effect call counts): by the time C crashes, the persisted state
        already contains A's and B's results, and the run knows it is
        waiting on research_c specifically."""
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="before")

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(
                store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=checkpointer
            )
            outcome = run_or_crash(graph, initial_state("q"), thread_id="verify-checkpoint")
            assert outcome.status == "crashed"

            history = get_execution_history(graph, "verify-checkpoint")
            # Newest checkpoint first: waiting on research_c, with A and B
            # already recorded in the persisted state.
            latest = history[0]
            assert latest.next_tasks == ("research_c",)
            assert set(latest.results_so_far.keys()) == {"A", "B"}


class TestResumeIsANoOpOnAnAlreadyCompletedThread:
    def test_resuming_a_finished_thread_does_not_re_execute_anything(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            outcome = run_or_crash(graph, initial_state("q"), thread_id="already-done")
            assert outcome.status == "completed"
            assert get_pending_tasks(graph, "already-done") == ()

            # Calling resume() again (e.g. a duplicate "restart" signal
            # arriving after the run had already finished) must be a no-op.
            resumed_again = resume(graph, thread_id="already-done")
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
            assert resumed_again.final_report == outcome.final_report


class TestExecutionHistory:
    def test_history_shows_every_completed_step_in_order(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            run_or_crash(graph, initial_state("q"), thread_id="history-thread")

            history = get_execution_history(graph, "history-thread")
            # Oldest-to-newest view of which task set had already completed.
            results_progression = [set(entry.results_so_far.keys()) for entry in reversed(history)]
            assert results_progression == [
                set(),
                set(),
                {"A"},
                {"A", "B"},
                {"A", "B", "C"},
                {"A", "B", "C"},
            ]

    def test_history_survives_across_a_new_connection_to_the_same_file(self, db_path):
        """Execution history is itself durable -- a fresh connection to the
        same checkpoint file can read the full history of a completed (or
        crashed) run without needing the original graph/process."""
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            run_or_crash(graph, initial_state("q"), thread_id="durable-history")

        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer2)
            history = get_execution_history(graph2, "durable-history")
            assert len(history) == 6
            assert history[0].next_tasks == ()


class TestInMemorySaverIsUnsuitableForCrashRecovery:
    """Negative control demonstrating *why* the requirement bans
    ``InMemorySaver`` for production: even without any real process
    restart, simply constructing a brand-new saver instance (the closest
    an in-process test can get to simulating "a different process")
    already loses everything, whereas the SQLite-backed checkpointer does
    not."""

    def test_a_new_in_memory_saver_instance_has_no_knowledge_of_a_prior_thread(self):
        from langgraph.checkpoint.memory import InMemorySaver

        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="before")

        saver = InMemorySaver()
        graph = build_durable_graph(
            store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=saver
        )
        outcome = run_or_crash(graph, initial_state("q"), thread_id="memory-thread")
        assert outcome.status == "crashed"
        assert get_pending_tasks(graph, "memory-thread") == ("research_c",)

        # "Restart": a brand-new InMemorySaver, as a real process restart
        # would produce (a fresh, empty heap) -- unlike the SQLite file,
        # there is nothing on "disk" to reopen.
        fresh_saver = InMemorySaver()
        graph2 = build_durable_graph(
            store, side_effects=effects.as_side_effects(), crash_injector=crash_injector, checkpointer=fresh_saver
        )
        # The new saver has no record of "memory-thread" at all: nothing to
        # resume from -- get_state returns an empty snapshot, not the
        # crashed run's progress.
        assert get_pending_tasks(graph2, "memory-thread") == ()
        snapshot = graph2.get_state({"configurable": {"thread_id": "memory-thread"}})
        assert snapshot.values == {}


def test_sqlite_checkpointer_creates_parent_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "checkpoints.sqlite3"
    assert not nested_path.parent.exists()
    with sqlite_checkpointer(nested_path) as checkpointer:
        assert nested_path.exists()
        # Sanity: it is a real SQLite file with the checkpoints table LangGraph expects.
        conn = sqlite3.connect(str(nested_path))
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "checkpoints" in tables
        del checkpointer
