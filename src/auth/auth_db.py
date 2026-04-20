"""
src/auth/auth_db.py
users 테이블 CRUD + sessions 테이블 user_id 연동

테이블:
  users    — 사용자 계정 (id, email, password, name, role, is_active)
  sessions — 기존 테이블에 user_id 컬럼 추가
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ROOT_DIR

AUTH_DB_PATH = ROOT_DIR / "data" / "auth.db"


def _get_conn() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """테이블 생성 + 초기 admin 계정 생성"""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    UNIQUE NOT NULL,
                password    TEXT    NOT NULL,
                name        TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'user',
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT    NOT NULL,
                last_login  TEXT
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id),
                session_id  TEXT    NOT NULL,
                role        TEXT,
                purpose     TEXT,
                hitl_level  INTEGER DEFAULT 2,
                status      TEXT    DEFAULT 'running',
                summary     TEXT,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT
            );
        """)

    # 초기 admin 계정 생성 (없을 때만)
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@elens.com")
    admin_pw    = os.environ.get("ADMIN_PASSWORD", "elens2024!")
    admin_name  = os.environ.get("ADMIN_NAME", "관리자")

    if not get_user_by_email(admin_email):
        create_user(
            email=admin_email,
            password=admin_pw,
            name=admin_name,
            role="admin",
        )


# ── 비밀번호 해싱 ────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.environ.get("PASSWORD_SALT", "elens_salt_2024")
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


# ── 사용자 CRUD ──────────────────────────────────────────────────────

def create_user(
    email: str,
    password: str,
    name: str,
    role: str = "user",
) -> dict:
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (email, password, name, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (email, _hash_password(password), name, role, now),
        )
    return get_user_by_email(email)


def get_user_by_email(email: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_user(
    user_id: int,
    name: str | None = None,
    is_active: bool | None = None,
    password: str | None = None,
) -> None:
    fields, values = [], []
    if name is not None:
        fields.append("name = ?");       values.append(name)
    if is_active is not None:
        fields.append("is_active = ?");  values.append(int(is_active))
    if password is not None:
        fields.append("password = ?");   values.append(_hash_password(password))
    if not fields:
        return
    values.append(user_id)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE id = ?",
            values,
        )


def delete_user(user_id: int) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def update_last_login(user_id: int) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (_now(), user_id),
        )


# ── 세션 CRUD ────────────────────────────────────────────────────────

def save_session(
    user_id: int,
    session_id: str,
    role: str = "",
    purpose: str = "",
    hitl_level: int = 2,
) -> None:
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_sessions "
            "(user_id, session_id, role, purpose, hitl_level, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (user_id, session_id, role, purpose, hitl_level, now, now),
        )


def update_session(
    session_id: str,
    status: str | None = None,
    summary: str | None = None,
) -> None:
    fields, values = ["updated_at = ?"], [_now()]
    if status:
        fields.append("status = ?");  values.append(status)
    if summary:
        fields.append("summary = ?"); values.append(summary[:500])
    values.append(session_id)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE user_sessions SET {', '.join(fields)} WHERE session_id = ?",
            values,
        )


def get_sessions_by_user(user_id: int) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM user_sessions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_sessions() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT s.*, u.name as user_name, u.email as user_email
            FROM user_sessions s
            LEFT JOIN users u ON s.user_id = u.id
            ORDER BY s.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")