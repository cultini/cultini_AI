"""Per-chat conversation memory (SQLite).

Each ``/chat`` turn is persisted so a chat_id keeps a context window across
requests. History is reloaded on every call and fed back into retrieval and
synthesis so follow-up questions resolve. Mirrors the ``app/feedback.py``
pattern: same on-disk SQLite file, a fresh connection per call
(``check_same_thread=False``) because FastAPI serves requests across threads.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL,
    role       TEXT NOT NULL,         -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    route      TEXT,                  -- 'cultural' | 'direct' | NULL (user turns)
    timestamp  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_chat ON conversation_turns(chat_id, id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def append_turn(chat_id: str, role: str, content: str, route: str | None = None) -> int:
    """Record one turn (user or assistant). Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversation_turns (chat_id, role, content, route, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, route, datetime.now(timezone.utc).isoformat()),
        )
        return int(cur.lastrowid)


def get_history(chat_id: str, limit: int) -> list[dict]:
    """Return the last ``limit`` turns for ``chat_id``, oldest-first.

    Each item is ``{role, content, route}``.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, route FROM conversation_turns "
            "WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


if __name__ == "__main__":
    init_db()
    print("conversation_turns table initialized at", config.SQLITE_PATH)
