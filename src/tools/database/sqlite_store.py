from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """
    현재 파일 위치 기준으로 프로젝트 루트 경로 추정
    src/tools/database/sqlite_store.py -> 프로젝트 루트
    """
    return Path(__file__).resolve().parents[3]


def _default_db_path() -> Path:
    root = _project_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "agent_trace.db"


class TraceStore:
    """
    SQLite 기반 trace/session/user_memory 저장소
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path) if db_path else str(_default_db_path())
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _to_json(value: Any) -> str | None:
        if value is None:
            return None
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            logger.exception("JSON serialization failed. fallback to string.")
            return json.dumps({"raw": str(value)}, ensure_ascii=False)

    def _init_db(self) -> None:
        schema_sql = """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            task_type TEXT,
            initial_input TEXT,
            final_output_summary TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            ended_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user_id
            ON sessions (user_id);

        CREATE INDEX IF NOT EXISTS idx_sessions_created_at
            ON sessions (created_at);

        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_no INTEGER NOT NULL,
            parent_step_no INTEGER,
            event_type TEXT NOT NULL,
            event_name TEXT NOT NULL,
            input_payload TEXT,
            output_payload TEXT,
            error_message TEXT,
            model_name TEXT,
            prompt_version TEXT,
            tool_version TEXT,
            data_version TEXT,
            latency_ms INTEGER,
            hitl_flag INTEGER NOT NULL DEFAULT 0,
            hitl_decision TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_trace_events_session_id
            ON trace_events (session_id);

        CREATE INDEX IF NOT EXISTS idx_trace_events_session_step
            ON trace_events (session_id, step_no);

        CREATE INDEX IF NOT EXISTS idx_trace_events_event_type
            ON trace_events (event_type);

        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            source_session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, memory_key)
        );

        CREATE INDEX IF NOT EXISTS idx_user_memory_user_id
            ON user_memory (user_id);

        CREATE INDEX IF NOT EXISTS idx_user_memory_type
            ON user_memory (memory_type);
        """

        with self._connect() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    # =========================================================
    # sessions
    # =========================================================

    def create_session(
        self,
        session_id: str,
        user_id: str | None = None,
        task_type: str | None = None,
        initial_input: str | None = None,
    ) -> None:
        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, user_id, task_type, initial_input,
                    final_output_summary, status, created_at, updated_at, ended_at
                )
                VALUES (?, ?, ?, ?, NULL, 'active', ?, ?, NULL)
                """,
                (session_id, user_id, task_type, initial_input, now, now),
            )
            conn.commit()

    def update_session_summary(
        self,
        session_id: str,
        final_output_summary: str | None = None,
        status: str | None = None,
    ) -> None:
        now = self._now_iso()

        fields: list[str] = []
        values: list[Any] = []

        if final_output_summary is not None:
            fields.append("final_output_summary = ?")
            values.append(final_output_summary)

        if status is not None:
            fields.append("status = ?")
            values.append(status)

        fields.append("updated_at = ?")
        values.append(now)

        if status in {"completed", "failed"}:
            fields.append("ended_at = ?")
            values.append(now)

        values.append(session_id)

        query = f"""
        UPDATE sessions
        SET {", ".join(fields)}
        WHERE session_id = ?
        """

        with self._connect() as conn:
            conn.execute(query, values)
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    # =========================================================
    # trace events
    # =========================================================

    def get_next_step_no(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(step_no), 0) AS max_step
                FROM trace_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return int(row["max_step"]) + 1 if row else 1

    def log_event(
        self,
        session_id: str,
        event_type: str,
        event_name: str,
        input_payload: Any = None,
        output_payload: Any = None,
        error_message: str | None = None,
        parent_step_no: int | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        tool_version: str | None = None,
        data_version: str | None = None,
        latency_ms: int | None = None,
        hitl_flag: bool = False,
        hitl_decision: str | None = None,
        step_no: int | None = None,
    ) -> int:
        now = self._now_iso()

        self.create_session(session_id=session_id)

        if step_no is None:
            step_no = self.get_next_step_no(session_id)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trace_events (
                    session_id,
                    step_no,
                    parent_step_no,
                    event_type,
                    event_name,
                    input_payload,
                    output_payload,
                    error_message,
                    model_name,
                    prompt_version,
                    tool_version,
                    data_version,
                    latency_ms,
                    hitl_flag,
                    hitl_decision,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    step_no,
                    parent_step_no,
                    event_type,
                    event_name,
                    self._to_json(input_payload),
                    self._to_json(output_payload),
                    error_message,
                    model_name,
                    prompt_version,
                    tool_version,
                    data_version,
                    latency_ms,
                    1 if hitl_flag else 0,
                    hitl_decision,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_trace_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM trace_events
                WHERE session_id = ?
                ORDER BY step_no ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================
    # user memory
    # =========================================================

    def upsert_memory(
        self,
        user_id: str,
        memory_key: str,
        memory_value: Any,
        memory_type: str,
        confidence: float = 0.5,
        source_session_id: str | None = None,
    ) -> None:
        now = self._now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory (
                    user_id, memory_key, memory_value, memory_type,
                    confidence, source_session_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key)
                DO UPDATE SET
                    memory_value = excluded.memory_value,
                    memory_type = excluded.memory_type,
                    confidence = excluded.confidence,
                    source_session_id = excluded.source_session_id,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    memory_key,
                    self._to_json(memory_value),
                    memory_type,
                    confidence,
                    source_session_id,
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_user_memory(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM user_memory
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]