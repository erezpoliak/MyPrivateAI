"""SQLite persistence for the app (documents, chats, messages, sources)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from ..pipeline.common.config import Config

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class AppDB:
    """Thin SQLite wrapper for the four app tables.

    SQLite is opened in WAL mode with foreign keys enforced.
    Every public method opens, uses, and closes its own connection so the
    class is safe to share across threads (each thread serialises through
    the context manager).
    """

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._db_path = config.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── connection ─────────────────────────────────────────────────────────

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
        schema = _SCHEMA_PATH.read_text()
        with self._connect() as conn:
            conn.executescript(schema)

    # ── documents ──────────────────────────────────────────────────────────

    def create_document(
        self,
        id: str,
        filename: str,
        title: str,
        page_count: int = 0,
        chunk_count: int = 0,
        status: str = "ingesting",
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents (id, filename, title, page_count, chunk_count, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (id, filename, title, page_count, chunk_count, status),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (id,)
            ).fetchone()
            return dict(row)

    def update_document_status(
        self,
        id: str,
        status: str,
        chunk_count: int | None = None,
        page_count: int | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE documents
                   SET status = ?,
                       chunk_count = COALESCE(?, chunk_count),
                       page_count = COALESCE(?, page_count),
                       error = ?
                   WHERE id = ?""",
                (status, chunk_count, page_count, error, id),
            )

    def get_document(self, id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (id,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (id,))

    # ── chats ──────────────────────────────────────────────────────────────

    def create_chat(self, id: str, title: str = "New chat") -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats (id, title) VALUES (?, ?)",
                (id, title),
            )
            row = conn.execute(
                "SELECT * FROM chats WHERE id = ?", (id,)
            ).fetchone()
            return dict(row)

    def get_chat(self, id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chats WHERE id = ?", (id,)
            ).fetchone()
            return dict(row) if row else None

    def list_chats(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_chat(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (id,))

    # ── messages ───────────────────────────────────────────────────────────

    def create_message(
        self, id: str, chat_id: str, role: str, content: str
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
                (id, chat_id, role, content),
            )
            # Bump chat.updated_at so list_chats() sorts correctly
            conn.execute(
                "UPDATE chats SET updated_at = (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) WHERE id = ?",
                (chat_id,),
            )
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?", (id,)
            ).fetchone()
            return dict(row)

    def get_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── message_sources ────────────────────────────────────────────────────

    def create_message_sources(self, sources: list[dict[str, Any]]) -> None:
        """Insert one row per cited source chunk.

        Each dict must have: message_id, chunk_index, doc_id, title,
        page_start, page_end, snippet.
        """
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO message_sources
                       (message_id, chunk_index, doc_id, title, page_start, page_end, snippet)
                   VALUES
                       (:message_id, :chunk_index, :doc_id, :title, :page_start, :page_end, :snippet)""",
                sources,
            )

    def get_message_sources(self, message_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM message_sources
                   WHERE message_id = ?
                   ORDER BY chunk_index ASC""",
                (message_id,),
            ).fetchall()
            return [dict(r) for r in rows]
