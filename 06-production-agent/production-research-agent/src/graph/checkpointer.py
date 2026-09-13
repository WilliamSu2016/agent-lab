"""Production-capable checkpointer factory.

Same rationale as the Durable Execution experiment's
``src/durable/checkpointer.py`` (see ``docs/reliability.md``): a
file-backed SQLite checkpointer survives a worker crash / process
restart, unlike ``langgraph.checkpoint.memory.InMemorySaver`` -- this
project **never** uses ``InMemorySaver`` as a production checkpointer.

For a genuinely multi-replica deployment (several worker processes on
different machines, not just process restarts on one machine), swap
``SqliteSaver`` for ``langgraph.checkpoint.postgres.PostgresSaver`` (any
``BaseCheckpointSaver`` backed by a shared database) -- nothing else in
``src/graph``/``src/agents`` depends on SQLite specifically.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_CHECKPOINT_DB_PATH = Path("traces") / "production_checkpoints.sqlite3"


@contextmanager
def sqlite_checkpointer(db_path: str | Path = DEFAULT_CHECKPOINT_DB_PATH) -> Iterator[SqliteSaver]:
    """Open (creating if needed) a file-backed SQLite checkpointer bound to
    ``db_path``. Reopening against the same path -- from the same process
    or, in a real deployment, a freshly restarted one -- resumes exactly
    where the previous connection left off."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()


def open_sqlite_checkpointer(db_path: str | Path = DEFAULT_CHECKPOINT_DB_PATH) -> tuple[SqliteSaver, sqlite3.Connection]:
    """Non-contextmanager variant for callers (e.g. ``src/api/main.py``)
    that must keep the connection open for the whole process lifetime and
    close it explicitly on shutdown."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver, conn


__all__ = ["DEFAULT_CHECKPOINT_DB_PATH", "sqlite_checkpointer", "open_sqlite_checkpointer"]
