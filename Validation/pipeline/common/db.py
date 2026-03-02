"""SQLite persistence for experiment runs and per-question results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from .config import Config

# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment      TEXT    NOT NULL,          -- e.g. "baseline", "phase1", "phase2", "ceiling"
    corpus_mode     TEXT    NOT NULL,          -- "none" | "gold_ref" | "fetched"
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    config_json     TEXT    NOT NULL,          -- serialised Config snapshot
    notes           TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS question_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id),
    question_id         TEXT    NOT NULL,
    source_idx          TEXT    NOT NULL,
    complexity          INTEGER NOT NULL,
    answer_type         TEXT    NOT NULL,
    generated_answer    TEXT,
    contexts_json       TEXT,                  -- JSON list of retrieved chunks
    faithfulness        REAL,
    answer_relevancy    REAL,
    context_precision   REAL,
    context_recall      REAL,
    answer_correctness  REAL,
    trajectory_steps    INTEGER DEFAULT 0,     -- agent hops (phase2 only)
    trajectory_success  INTEGER DEFAULT 1,     -- 1=converged, 0=gave up
    latency_s           REAL,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS run_summary (
    summary_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL UNIQUE REFERENCES runs(run_id),
    num_questions       INTEGER NOT NULL,
    avg_faithfulness    REAL,
    avg_answer_relevancy REAL,
    avg_context_precision REAL,
    avg_context_recall  REAL,
    avg_answer_correctness REAL,
    avg_latency_s       REAL,
    breakdown_json      TEXT                   -- per-complexity averages
);
"""


class RunDB:
    """Thin wrapper around the runs SQLite database."""

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._db_path = config.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection helpers ──────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── runs ────────────────────────────────────────────────────────────────

    def start_run(
        self,
        experiment: str,
        corpus_mode: str,
        config_snapshot: dict[str, Any],
        notes: str = "",
    ) -> int:
        """Insert a new run row and return its run_id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO runs (experiment, corpus_mode, started_at, config_json, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    experiment,
                    corpus_mode,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(config_snapshot, default=str),
                    notes,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def finish_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )

    # ── question results ────────────────────────────────────────────────────

    def insert_result(self, run_id: int, **kwargs: Any) -> int:
        """Insert a single question result. kwargs must match column names."""
        cols = list(kwargs.keys())
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        col_str = ", ".join(["run_id"] + cols)
        values = [run_id] + [
            json.dumps(v) if isinstance(v, (list, dict)) else v
            for v in kwargs.values()
        ]
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO question_results ({col_str}) VALUES ({placeholders})",
                values,
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_results(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM question_results WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── run summary ─────────────────────────────────────────────────────────

    def save_summary(self, run_id: int, summary: dict[str, Any]) -> None:
        cols = list(summary.keys())
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        col_str = ", ".join(["run_id"] + cols)
        values = [run_id] + [
            json.dumps(v) if isinstance(v, (list, dict)) else v
            for v in summary.values()
        ]
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO run_summary ({col_str}) VALUES ({placeholders})",
                values,
            )

    def get_summary(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM run_summary WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
