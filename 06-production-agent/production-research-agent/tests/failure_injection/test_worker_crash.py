"""Failure #8 Worker crash, #9 Checkpoint failure.

Expected behaviour (the exact scenario this experiment specifies):

    Research A succeeds
    Research B succeeds
    Research C crashes
        checkpoint -> restart -> resume: A and B are NOT re-executed;
        C resumes from the appropriate point.

Real, uncaught worker crashes (an OS-level process kill, ``os._exit``, a
segfault) never surface as a Python ``Exception`` a ``try/except
Exception`` clause could catch -- they simply stop the process. This
suite models that faithfully with a ``BaseException`` subclass
(``WorkerCrash``), deliberately *not* a subclass of ``Exception``: like a
real ``SystemExit``/``KeyboardInterrupt``, it is not swallowed by
``src.agents.researcher``'s "never raise" ``except Exception`` fallback
(that fallback exists to keep one *recoverable* researcher failure --
timeout/guardrail/tool error -- from crashing the rest of the fan-out; it
must NOT also hide a genuine process crash from the supervising process).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.graph import recovery
from src.graph.checkpointer import sqlite_checkpointer
from src.graph.graph import build_graph
from src.graph.state import initial_state
from src.observability.tracing import ExecutionContext, bind_execution_context


class WorkerCrash(BaseException):
    """Simulates a real out-of-process worker crash (OS kill/segfault):
    intentionally a ``BaseException``, not an ``Exception``, so it is not
    caught by any layer's ``except Exception`` recoverable-failure
    handling -- exactly like a real ``SystemExit``/interpreter kill."""


IDENTITY = {"user_id": "u1", "tenant_id": "t1", "roles": ["researcher"]}


def _planner_llm(system: str, user: str) -> str:
    return json.dumps(["Research A", "Research B", "Research C"])


def _synth_llm(system: str, user: str) -> str:
    return "Synthesized report combining A, B, C."


def _review_llm(system: str, user: str) -> str:
    return "APPROVE\nComplete and consistent."


def _researcher_llm_that_crashes_on_c(call_count: dict):
    def _call(system: str, user: str) -> str:
        key = "C" if "Research C" in user else ("A" if "Research A" in user else "B")
        call_count[key] = call_count.get(key, 0) + 1
        if key == "C":
            raise WorkerCrash("Simulated worker crash while researching 'Research C'.")
        return f"Finding for {key}: {user[:60]}"

    return _call


def _researcher_llm_fixed(call_count: dict):
    def _call(system: str, user: str) -> str:
        key = "C" if "Research C" in user else ("A" if "Research A" in user else "B")
        call_count[key] = call_count.get(key, 0) + 1
        return f"Finding for {key}: {user[:60]}"

    return _call


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "chaos_checkpoints.sqlite3"


class TestWorkerCrash:
    """Failure #8, using the exact scenario the experiment specifies:
    Research A succeeds, Research B succeeds, Research C crashes; after
    recovery A and B are not re-executed and C resumes correctly."""

    def test_research_a_and_b_are_not_reexecuted_after_a_simulated_crash_on_c(self, db_path):
        call_count: dict = {}
        thread_id = "chaos-crash"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(
                _planner_llm,
                _researcher_llm_that_crashes_on_c(call_count),
                _synth_llm,
                _review_llm,
                checkpointer=checkpointer,
            )
            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=3, max_iterations=1)
            outcome = recovery.run_or_crash(graph, state, thread_id)

            assert outcome.status == "crashed"
            assert isinstance(outcome.crash, WorkerCrash)
            # A and B already ran to completion; C crashed.
            assert call_count == {"A": 1, "B": 1, "C": 1}
            assert recovery.get_pending_tasks(graph, thread_id) == ("researcher",)

        # Restart: a brand-new process would open a fresh graph object
        # against the same on-disk checkpoint file -- never the same
        # in-process graph the crash happened on.
        with sqlite_checkpointer(db_path) as checkpointer2:
            # The underlying cause of the crash is now "fixed" (e.g. the
            # bad deploy that caused the OOM/segfault was rolled back) --
            # modeled here with a researcher LLM call that no longer
            # raises for Research C.
            graph2 = build_graph(
                _planner_llm,
                _researcher_llm_fixed(call_count),
                _synth_llm,
                _review_llm,
                checkpointer=checkpointer2,
            )
            resumed = recovery.resume(graph2, thread_id)

            assert resumed.status == "completed"
            # A and B are NOT re-executed (still exactly 1 call each);
            # only C -- which had never successfully completed -- runs
            # again.
            assert call_count == {"A": 1, "B": 1, "C": 2}
            assert recovery.get_pending_tasks(graph2, thread_id) == ()
            assert "Synthesized report" in resumed.final_answer

    def test_execution_history_is_inspectable_after_a_crashed_run(self, db_path):
        """Requirement 10: "支持查看 execution history" -- even a crashed,
        not-yet-resumed run's history/pending tasks are inspectable
        without needing the original (crashed) graph/process object."""
        call_count: dict = {}
        thread_id = "chaos-history"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(
                _planner_llm, _researcher_llm_that_crashes_on_c(call_count), _synth_llm, _review_llm, checkpointer=checkpointer
            )
            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=3, max_iterations=1)
            recovery.run_or_crash(graph, state, thread_id)

        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_graph(
                _planner_llm, _researcher_llm_that_crashes_on_c(call_count), _synth_llm, _review_llm, checkpointer=checkpointer2
            )
            history = recovery.get_execution_history(graph2, thread_id)
            assert len(history) >= 2
            # Newest-first: the most recent checkpoint recorded is still
            # waiting on the researcher fan-out to finish.
            assert recovery.get_pending_tasks(graph2, thread_id) == ("researcher",)


def _invoke_direct(graph, state, thread_id: str):
    """Direct ``graph.invoke`` (bypassing ``recovery.run_or_crash``'s own
    exception handling) for tests that need the raw exception to escape,
    with the execution context bound the same way ``recovery.py`` does it
    internally."""
    context = ExecutionContext(**state["context"])
    with bind_execution_context(context):
        return graph.invoke(state, {"configurable": {"thread_id": thread_id}})


def _persistent_failure_from_step(checkpointer, first_failing_step: int):
    """Wrap ``checkpointer.put`` so that every write from
    ``first_failing_step`` onward raises -- modeling sustained storage
    unavailability (a downed database, a full disk) rather than one flaky
    write."""
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
    completed steps."""

    def test_checkpoint_write_failure_propagates_uncaught(self, db_path):
        call_count: dict = {}
        thread_id = "chaos-ckpt-fail"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(_planner_llm, _researcher_llm_fixed(call_count), _synth_llm, _review_llm, checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=3, max_iterations=1)
            # A checkpoint-store failure is never swallowed anywhere in
            # the graph -- it must escape `graph.invoke()` exactly like a
            # worker crash would, so a supervising process can detect it.
            with pytest.raises(RuntimeError, match="simulated checkpoint storage failure"):
                _invoke_direct(graph, state, thread_id)

    def test_checkpoints_written_before_the_failure_remain_durable_on_disk(self, db_path):
        """Ground-truth durability check: inspect the SQLite file
        directly to prove every checkpoint committed *before* storage
        "went down" is still there."""
        call_count: dict = {}
        thread_id = "chaos-ckpt-fail-2"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(_planner_llm, _researcher_llm_fixed(call_count), _synth_llm, _review_llm, checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=3, max_iterations=1)
            with pytest.raises(RuntimeError):
                _invoke_direct(graph, state, thread_id)

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

        # Steps -1 and 0/1 (start, supervisor_entry) made it to disk;
        # nothing from step 2 onward did.
        assert -1 in steps
        assert all(step < 2 for step in steps)

    def test_storage_recovering_then_resuming_completes(self, db_path):
        call_count: dict = {}
        thread_id = "chaos-ckpt-fail-3"

        with sqlite_checkpointer(db_path) as checkpointer:
            graph = build_graph(_planner_llm, _researcher_llm_fixed(call_count), _synth_llm, _review_llm, checkpointer=checkpointer)
            _persistent_failure_from_step(checkpointer, first_failing_step=2)

            state = initial_state("Research topic", IDENTITY, mode="balanced", max_workers=3, max_iterations=1)
            with pytest.raises(RuntimeError):
                _invoke_direct(graph, state, thread_id)

        # "Storage recovered": a brand-new, healthy checkpointer/graph
        # object over the same on-disk file.
        with sqlite_checkpointer(db_path) as checkpointer2:
            graph2 = build_graph(_planner_llm, _researcher_llm_fixed(call_count), _synth_llm, _review_llm, checkpointer=checkpointer2)
            resumed = recovery.resume(graph2, thread_id)
            assert resumed.status == "completed"

