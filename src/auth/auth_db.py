"""
src/auth/auth_db.py
users 테이블 CRUD + sessions 테이블 user_id 연동

테이블:
    users    — 사용자 계정 (id, login_id, email, password, name, role, is_active)
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
                login_id    TEXT    UNIQUE,
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

            CREATE TABLE IF NOT EXISTS signup_requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                login_id    TEXT,
                email       TEXT    UNIQUE NOT NULL,
                password_hash TEXT,
                message     TEXT,
                status      TEXT    NOT NULL DEFAULT 'pending',
                created_at  TEXT    NOT NULL,
                reviewed_at TEXT
            );
        """)

        _ensure_column(conn, "users", "login_id", "TEXT")
        _ensure_column(conn, "signup_requests", "login_id", "TEXT")
        _ensure_column(conn, "signup_requests", "password_hash", "TEXT")

        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_login_id ON users(login_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_signup_requests_login_id ON signup_requests(login_id)"
        )

        _backfill_users_login_id(conn)
        _backfill_signup_requests(conn)

    # 초기 admin 계정 생성 (없을 때만)
    admin_login_id = _normalize_login_id(os.environ.get("ADMIN_LOGIN_ID", "admin"))
    admin_email    = os.environ.get("ADMIN_EMAIL", f"{admin_login_id}@elens.local")
    admin_pw       = os.environ.get("ADMIN_PASSWORD", "elens2024!")
    admin_name     = os.environ.get("ADMIN_NAME", "관리자")

    if not get_user_by_login_id(admin_login_id):
        create_user(
            login_id=admin_login_id,
            password=admin_pw,
            name=admin_name,
            role="admin",
            email=admin_email,
        )


# ── 비밀번호 해싱 ────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.environ.get("PASSWORD_SALT", "elens_salt_2024")
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed


def _normalize_login_id(login_id: str) -> str:
    return (login_id or "").strip().lower()


def _make_placeholder_email(login_id: str) -> str:
    return f"{_normalize_login_id(login_id)}@local.invalid"


# ── 사용자 CRUD ──────────────────────────────────────────────────────

def create_user(
    login_id: str,
    password: str,
    name: str,
    role: str = "user",
    email: str | None = None,
    password_hashed: bool = False,
) -> dict:
    login_id = _normalize_login_id(login_id)
    stored_pw = password if password_hashed else _hash_password(password)
    stored_email = (email or _make_placeholder_email(login_id)).strip().lower()
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (login_id, email, password, name, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?)",
            (login_id, stored_email, stored_pw, name, role, now),
        )
    return get_user_by_login_id(login_id)


def get_user_by_login_id(login_id: str) -> Optional[dict]:
    login_id = _normalize_login_id(login_id)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE login_id = ?", (login_id,)
        ).fetchone()
    return dict(row) if row else None


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
            SELECT s.*, u.name as user_name, u.login_id as user_login_id
            FROM user_sessions s
            LEFT JOIN users u ON s.user_id = u.id
            ORDER BY s.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


# ── 가입 요청 CRUD ──────────────────────────────────────────────────

def create_signup_request(
    name: str,
    login_id: str,
    password: str,
    message: str = "",
) -> bool:
    """가입 요청 생성 — 이미 있으면 False"""
    norm_login_id = _normalize_login_id(login_id)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO signup_requests (name, login_id, email, password_hash, message, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (
                    name,
                    norm_login_id,
                    _make_placeholder_email(norm_login_id),
                    _hash_password(password),
                    message,
                    _now(),
                ),
            )
        return True
    except Exception:
        return False


def get_signup_requests(status: str = "pending") -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signup_requests WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_signup_requests() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signup_requests ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def approve_signup_request(request_id: int) -> Optional[dict]:
    """가입 요청 승인 → 요청 비밀번호로 사용자 계정 생성"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM signup_requests WHERE id = ?", (request_id,)
        ).fetchone()
    if not row:
        return None
    req = dict(row)

    if not req.get("password_hash"):
        return None

    # 계정 생성
    user = create_user(
        login_id=req["login_id"],
        password=req["password_hash"],
        name=req["name"],
        role="user",
        email=req.get("email") or _make_placeholder_email(req["login_id"]),
        password_hashed=True,
    )

    # 요청 상태 업데이트
    with _get_conn() as conn:
        conn.execute(
            "UPDATE signup_requests SET status = 'approved', reviewed_at = ? WHERE id = ?",
            (_now(), request_id),
        )
    return user


def reject_signup_request(request_id: int) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE signup_requests SET status = 'rejected', reviewed_at = ? WHERE id = ?",
            (_now(), request_id),
        )


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_def: str,
) -> None:
    cols = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in cols:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
        )


def _backfill_users_login_id(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, email, login_id FROM users ORDER BY id").fetchall()
    used: set[str] = set()

    for row in rows:
        current = _normalize_login_id(row["login_id"] or "")
        if current:
            used.add(current)

    for row in rows:
        current = _normalize_login_id(row["login_id"] or "")
        if current:
            continue

        email = (row["email"] or "").strip().lower()
        base = email.split("@")[0] if "@" in email else f"user{row['id']}"
        base = _normalize_login_id(base) or f"user{row['id']}"

        candidate = base
        idx = 1
        while candidate in used:
            idx += 1
            candidate = f"{base}{idx}"

        used.add(candidate)
        conn.execute(
            "UPDATE users SET login_id = ? WHERE id = ?",
            (candidate, row["id"]),
        )


def _backfill_signup_requests(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, login_id, email, password_hash FROM signup_requests"
    ).fetchall()

    for row in rows:
        updates: list[str] = []
        values: list[object] = []

        login_id = _normalize_login_id(row["login_id"] or "")
        if not login_id:
            email = (row["email"] or "").strip().lower()
            base = email.split("@")[0] if "@" in email else f"request{row['id']}"
            login_id = _normalize_login_id(base) or f"request{row['id']}"
            updates.append("login_id = ?")
            values.append(login_id)

        if not (row["email"] or "").strip():
            updates.append("email = ?")
            values.append(_make_placeholder_email(login_id))

        if updates:
            values.append(row["id"])
            conn.execute(
                f"UPDATE signup_requests SET {', '.join(updates)} WHERE id = ?",
                values,
            )