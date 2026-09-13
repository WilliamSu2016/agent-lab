"""Production-capable checkpointer factory.

Requirement: "使用 production-capable checkpointer；不再使用 InMemorySaver 作为
生产方案。"

Why ``langgraph.checkpoint.memory.InMemorySaver`` is a *demo-only* choice
--------------------------------------------------------------------------
``InMemorySaver`` keeps every checkpoint in a plain Python ``dict`` that
lives only inside the current process's heap:

* A worker crash (the exact scenario this experiment simulates) kills the
  process -- and with it, every checkpoint. There is nothing left to
  "restart and resume" from; the crash is indistinguishable from total data
  loss.
* It cannot be shared across processes/replicas. A crashed worker being
  replaced by a *different* worker process (the realistic production
  scenario -- a supervisor/orchestrator restarts a fresh container, it does
  not resurrect the dead one) has no access to the dead process's memory.
* It has no eviction/retention story and is not safe for concurrent access
  from multiple threads/processes.

This module instead builds a **file-backed SQLite checkpointer**
(``langgraph-checkpoint-sqlite``, wrapping the same durability primitives
as ``langgraph-checkpoint-postgres``): checkpoints are written to a
``.sqlite3`` file on disk, so a brand new Python process (a real restarted
worker, or -- as in this experiment's tests -- a freshly constructed graph
object standing in for one) can open the *same* file and continue exactly
where the crashed process left off. See ``docs/03-DURABLE-EXECUTION.md``
for the full Checkpoint vs Memory vs Database vs Execution History
comparison this design is based on.

For a real multi-replica production deployment (several worker processes on
different machines, not just several process restarts on one machine),
swap ``SqliteSaver`` for ``langgraph.checkpoint.postgres.PostgresSaver`` (or
any other ``BaseCheckpointSaver`` backed by a shared database) -- the rest
of this package (``graph.py``, ``recovery.py``) only depends on the
``BaseCheckpointSaver`` interface, never on SQLite specifically, so that
swap requires no changes outside this one factory function.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_CHECKPOINT_DB_PATH = Path("traces") / "durable_checkpoints.sqlite3"


@contextmanager
def sqlite_checkpointer(db_path: str | Path = DEFAULT_CHECKPOINT_DB_PATH) -> Iterator[SqliteSaver]:
    """Open (creating if needed) a file-backed SQLite checkpointer.

    Yields a ``SqliteSaver`` bound to ``db_path``. Because the backing
    store is a file rather than process memory, closing this context
    manager and opening a *new* one against the same ``db_path`` -- from
    the same process or, in a real deployment, a completely different one
    -- sees every checkpoint written before the connection closed. That is
    exactly the "restart" half of "crash -> checkpoint -> restart ->
    resume": see ``src/durable/recovery.py::resume``.

    ``db_path`` defaults to a real on-disk file
    (``traces/durable_checkpoints.sqlite3``) so a plain run of this
    experiment behaves like a real deployment would. Tests use a temporary
    file per test (never ``:memory:`` -- an in-memory SQLite connection has
    exactly the same "dies with the process" problem as ``InMemorySaver``
    and would not actually exercise crash/restart semantics).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: LangGraph's Pregel runtime may touch the
    # connection from a different thread than the one that opened it
    # (e.g. when a node itself is invoked in a worker thread); SqliteSaver
    # serializes access internally, so this is safe.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()
