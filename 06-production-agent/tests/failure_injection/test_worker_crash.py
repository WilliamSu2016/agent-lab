"""Failure #8 Worker crash, #9 Checkpoint failure.

Expected behaviour (per the experiment's own example):

    Worker crash
        checkpoint -> restart -> resume: already-completed, already
        durably-checkpointed nodes are never re-executed; only the node
        that had not yet completed runs again.

    Checkpoint failure
        a failure while *writing* a checkpoint must propagate (never be
        silently swallowed), and every checkpoint durably committed
        *before* the failure must remain intact and resumable. A subtler,
        honest finding this file also demonstrates: because a node's side
        effect can run *before* its own checkpoint write succeeds, a
        checkpoint-write failure can cause that node to be replayed on
        resume even though its side effect already happened once -- the
        only thing preventing the side effect from firing a *second* time
        in that case is the idempotency layer (``src.reliability.idempotency``),
        not checkpointing by itself.
"""

from __future__ import annotations

import pytest

from src.durable.checkpointer import sqlite_checkpointer
from src.durable.graph import build_durable_graph, initial_state
from src.durable.recovery import (
    CrashInjector,
    WorkerCrash,
    get_pending_tasks,
    resume,
    run_or_crash,
)
from src.reliability.idempotency import InMemoryIdempotencyStore


class CountingSideEffect:
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
    return tmp_path / "chaos_checkpoints.sqlite3"


class TestWorkerCrash:
    """Failure #8, using the exact scenario the experiment specifies:
    Research A succeeds, Research B succeeds, Research C crashes; after
    recovery A and B are not re-executed and C resumes correctly."""

    def test_research_a_and_b_are_not_reexecuted_after_a_simulated_crash_on_c(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="before")

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(
                store,
                side_effects=effects.as_side_effects(),
                crash_injector=crash_injector,
                checkpointer=checkpointer,
            )
            outcome = run_or_crash(graph, initial_state("chaos question"), thread_id="chaos-crash")

            assert outcome.status == "crashed"
            assert isinstance(outcome.crash, WorkerCrash)
            assert effects.calls == {"A": 1, "B": 1, "C": 0}
            assert get_pending_tasks(graph, "chaos-crash") == ("research_c",)

        # Restart: a brand-new process would open a fresh graph object
        # against the same on-disk checkpoint file.
        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(
                store,
                side_effects=effects.as_side_effects(),
                crash_injector=crash_injector,
                checkpointer=checkpointer2,
            )
            resumed = resume(graph2, thread_id="chaos-crash")

            assert resumed.status == "completed"
            # A and B are NOT re-executed; only C, which had never
            # completed, runs (for the first time).
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
            assert set(resumed.results.keys()) == {"A", "B", "C"}
            assert get_pending_tasks(graph2, "chaos-crash") == ()

    def test_crash_after_c_side_effect_already_ran_does_not_duplicate_it_on_resume(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        crash_injector = CrashInjector()
        crash_injector.arm("C", when="after")

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(
                store,
                side_effects=effects.as_side_effects(),
                crash_injector=crash_injector,
                checkpointer=checkpointer,
            )
            outcome = run_or_crash(graph, initial_state("chaos question"), thread_id="chaos-crash-after")
            assert outcome.status == "crashed"
            # C's side effect already ran once even though the crash
            # happened before the graph could commit that fact.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}

        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(
                store,
                side_effects=effects.as_side_effects(),
                crash_injector=crash_injector,
                checkpointer=checkpointer2,
            )
            resumed = resume(graph2, thread_id="chaos-crash-after")
            assert resumed.status == "completed"
            # The node replayed, but the idempotency store -- not the
            # checkpointer -- prevented the real side effect from firing twice.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}


def _persistent_failure_from_step(checkpointer, first_failing_step: int):
    """Wrap ``checkpointer.put`` so that every write from ``first_failing_step``
    onward raises -- modeling sustained storage unavailability (a downed
    database, a full disk) rather than one single flaky write. A *single*
    one-off failed write was observed (via direct experimentation against
    this project's real LangGraph version) to not reliably block later
    writes from independently succeeding -- LangGraph's checkpoint
    persistence is not a simple "one write must finish before the next
    step's write is attempted" chain, so a single-shot failure does not
    deterministically produce a "stuck exactly here" state. A *sustained*
    failure from a point onward does: nothing from that point on can ever
    be durably committed, which is the realistic "storage is down" failure
    mode this test needs.
    """
    original_put = checkpointer.put

    def failing_put(config, checkpoint, metadata, new_versions):
        step = metadata.get("step") if metadata else None
        if step is not None and step >= first_failing_step:
            raise RuntimeError("simulated checkpoint storage failure")
        return original_put(config, checkpoint, metadata, new_versions)

    checkpointer.put = failing_put
    return original_put


class TestCheckpointFailure:
    """Failure #9: the checkpoint *store itself* fails to durably persist
    completed steps (e.g. a database write error, a full disk, a network
    partition to the checkpoint backend) -- sustained from some point
    onward, modeling storage being genuinely down rather than one flaky
    write."""

    def test_checkpoint_write_failure_propagates_uncaught(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            # A checkpoint-store failure is never swallowed anywhere in the
            # graph -- it must escape `graph.invoke()` exactly like a worker
            # crash would, so a supervising process can detect it.
            with pytest.raises(RuntimeError, match="simulated checkpoint storage failure"):
                graph.invoke(initial_state("chaos question"), {"configurable": {"thread_id": "chaos-ckpt-fail"}})

            # By the time the failing writes are attempted, every research
            # side effect has already executed in-process (LangGraph runs a
            # superstep's node before persisting its checkpoint) -- this is
            # the honest finding this test exists to pin down: a checkpoint
            # write failure does not "undo" work that already happened.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}

    def test_checkpoints_written_before_the_failure_remain_durable_on_disk(self, db_path):
        """Ground-truth durability check: inspect the SQLite file directly
        (bypassing LangGraph's own state-reconstruction APIs, whose exact
        ``next``-node bookkeeping under fault injection is an internal
        implementation detail this experiment does not assert on) to prove
        that every checkpoint committed *before* storage "went down" is
        still there, and nothing from the failure point onward is."""
        import sqlite3

        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        thread_id = "chaos-ckpt-fail-2"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            with pytest.raises(RuntimeError):
                graph.invoke(initial_state("chaos question"), {"configurable": {"thread_id": thread_id}})

        conn = sqlite3.connect(str(db_path))
        try:
            steps = sorted(
                row[0]
                for row in conn.execute(
                    "SELECT json_extract(metadata, '$.step') FROM checkpoints WHERE thread_id = ?",
                    (thread_id,),
                )
            )
        finally:
            conn.close()

        # Only the checkpoints strictly before the injected failure point
        # (start=-1, task-dispatch=0, research_a-completed=1) made it to
        # disk; nothing from step 2 onward did.
        assert steps == [-1, 0, 1]

    def test_storage_recovering_then_resuming_completes_without_duplicating_side_effects(self, db_path):
        effects = CountingSideEffect()
        store = InMemoryIdempotencyStore()
        thread_id = "chaos-ckpt-fail-3"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            with pytest.raises(RuntimeError):
                graph.invoke(initial_state("chaos question"), {"configurable": {"thread_id": thread_id}})
            assert effects.calls == {"A": 1, "B": 1, "C": 1}

        # "Storage recovered": a brand-new, healthy checkpointer/graph
        # object over the same on-disk file -- the same "restart" pattern
        # used for the plain worker-crash scenario above.
        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_durable_graph(store, side_effects=effects.as_side_effects(), checkpointer=checkpointer2)
            resumed = resume(graph2, thread_id=thread_id)

            assert resumed.status == "completed"
            assert set(resumed.results.keys()) == {"A", "B", "C"}
            # Only the last durably-committed checkpoint (research_a done)
            # is what LangGraph resumes from, so research_b/c and finalize
            # are replayed from scratch -- but the idempotency store (not
            # the checkpointer) guarantees the *real* side effects for B
            # and C, which already ran once above, are not invoked again.
            assert effects.calls == {"A": 1, "B": 1, "C": 1}
