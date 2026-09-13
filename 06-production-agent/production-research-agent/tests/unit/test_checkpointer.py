"""Unit tests: the production checkpointer factory
(``src.graph.checkpointer``) -- confirms it is genuinely file-backed and
NOT ``InMemorySaver``, and that it round-trips checkpoints across a
freshly re-opened connection (the same guarantee ``recovery.py``'s
"restart" step relies on)."""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.checkpointer import sqlite_checkpointer


def test_sqlite_checkpointer_is_not_the_in_memory_saver(tmp_path):
    with sqlite_checkpointer(tmp_path / "ckpt.sqlite3") as checkpointer:
        assert isinstance(checkpointer, SqliteSaver)
        assert not type(checkpointer).__name__ == "InMemorySaver"


def test_sqlite_checkpointer_creates_a_real_file_on_disk(tmp_path):
    db_path = tmp_path / "nested" / "ckpt.sqlite3"
    assert not db_path.exists()
    with sqlite_checkpointer(db_path):
        pass
    assert db_path.exists()


def test_reopening_the_same_path_resumes_from_the_same_checkpoints(tmp_path):
    db_path = tmp_path / "ckpt.sqlite3"
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}

    with sqlite_checkpointer(db_path) as checkpointer:
        checkpointer.put(
            config,
            {
                "v": 1,
                "id": "chk-1",
                "ts": "2024-01-01T00:00:00+00:00",
                "channel_values": {"x": 1},
                "channel_versions": {},
                "versions_seen": {},
            },
            {"step": 0, "source": "input", "writes": {}, "parents": {}},
            {},
        )

    # A brand-new checkpointer object over the SAME file -- simulating a
    # restarted process -- must see what the previous one wrote.
    with sqlite_checkpointer(db_path) as checkpointer2:
        tup = checkpointer2.get_tuple(config)
        assert tup is not None
        assert tup.checkpoint["id"] == "chk-1"
