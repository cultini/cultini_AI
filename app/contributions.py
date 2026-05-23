"""Contribution moderation queue (SQLite).

This is the "file de modération" in the contribution flow: every submission that
clears (or is caught by) the auto-filter is persisted here with its flags and a
status, awaiting expert validation. Approving a row promotes it to a documented
fiche (see ``app.ingest.promote_contribution``) and re-indexes Qdrant.

Mirrors ``app/feedback.py`` / ``app/conversations.py``: same on-disk SQLite file
(``config.SQLITE_PATH``), a fresh connection per call (``check_same_thread=False``)
because FastAPI serves requests across threads.

Statuses: ``pending`` → ``approved`` | ``rejected`` (expert) ; ``auto_rejected``
is set by the auto-filter before any human sees it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contributions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    titre            TEXT NOT NULL,
    categorie        TEXT NOT NULL,
    region           TEXT NOT NULL,
    contenu          TEXT NOT NULL,
    source           TEXT NOT NULL,
    contributor_name TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected|auto_rejected
    flags            TEXT,                              -- JSON: auto-filter verdict
    fiche_id         TEXT,                              -- set once promoted to a fiche
    timestamp        TEXT NOT NULL,
    moderated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_contrib_status ON contributions(status, id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["flags"] = json.loads(d["flags"]) if d.get("flags") else None
    return d


def insert_contribution(
    *,
    titre: str,
    categorie: str,
    region: str,
    contenu: str,
    source: str,
    contributor_name: str | None,
    status: str,
    flags: dict,
) -> int:
    """Persist one screened submission. Returns the new row id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO contributions (titre, categorie, region, contenu, source, "
            "contributor_name, status, flags, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                titre, categorie, region, contenu, source, contributor_name,
                status, json.dumps(flags, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return int(cur.lastrowid)


def get_contribution(contrib_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM contributions WHERE id = ?", (contrib_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_contributions(status: str | None = None, limit: int = 100) -> list[dict]:
    """List submissions, newest first, optionally filtered by ``status``."""
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM contributions WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contributions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def set_status(contrib_id: int, status: str, fiche_id: str | None = None) -> None:
    """Record an expert decision (and the resulting fiche id when approved)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE contributions SET status = ?, fiche_id = ?, moderated_at = ? "
            "WHERE id = ?",
            (status, fiche_id, datetime.now(timezone.utc).isoformat(), contrib_id),
        )


def get_stats() -> dict:
    """Counts per status for a moderation dashboard."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM contributions").fetchone()["n"]
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM contributions GROUP BY status"
            ).fetchall()
        }
    return {"total": total, "by_status": by_status}


if __name__ == "__main__":
    init_db()
    print("contributions table initialized at", config.SQLITE_PATH)
