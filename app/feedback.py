"""Human-in-the-loop feedback store (SQLite).

Every answer can be voted on and corrected. Corrections from native-speaker
experts are the raw material for re-curating the corpus (and flipping fiches
from ``a_verifier`` to ``documentee``). ``get_stats()`` powers the HITL
dashboard endpoint.

A fresh connection is opened per call (``check_same_thread=False``) because
FastAPI serves requests across threads.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question      TEXT NOT NULL,
    ai_response   TEXT NOT NULL,
    vote          TEXT,            -- 'up' | 'down' | NULL
    reason        TEXT,
    correction    TEXT,
    expert_status TEXT NOT NULL DEFAULT 'pending',  -- pending | verified | rejected
    timestamp     TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(_SCHEMA)


def insert_feedback(
    question: str,
    ai_response: str,
    vote: str | None = None,
    reason: str | None = None,
    correction: str | None = None,
) -> int:
    """Record a vote and/or a correction. Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (question, ai_response, vote, reason, correction, "
            "expert_status, timestamp) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (question, ai_response, vote, reason, correction,
             datetime.now(timezone.utc).isoformat()),
        )
        return int(cur.lastrowid)


def get_stats() -> dict:
    """Aggregate counts for the HITL dashboard."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
        up = conn.execute("SELECT COUNT(*) AS n FROM feedback WHERE vote='up'").fetchone()["n"]
        down = conn.execute("SELECT COUNT(*) AS n FROM feedback WHERE vote='down'").fetchone()["n"]
        corrections = conn.execute(
            "SELECT COUNT(*) AS n FROM feedback WHERE correction IS NOT NULL AND correction != ''"
        ).fetchone()["n"]
        by_status = {
            row["expert_status"]: row["n"]
            for row in conn.execute(
                "SELECT expert_status, COUNT(*) AS n FROM feedback GROUP BY expert_status"
            ).fetchall()
        }
        recent = [
            dict(row)
            for row in conn.execute(
                "SELECT id, question, vote, reason, correction, expert_status, timestamp "
                "FROM feedback ORDER BY id DESC LIMIT 10"
            ).fetchall()
        ]
    return {
        "total_feedback": total,
        "votes": {"up": up, "down": down},
        "corrections_submitted": corrections,
        "by_expert_status": by_status,
        "recent": recent,
    }


if __name__ == "__main__":
    init_db()
    print("feedback.db initialized at", config.SQLITE_PATH)
