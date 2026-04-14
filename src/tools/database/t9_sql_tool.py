"""
T-09 SQL Tool (production-ready v2)

자연어 질문 → SQL 생성 → 안전성 검증 → 실행 → 결과 반환
- SELECT 전용
- DDL / DML 차단
- 다중 문장 차단
- 실패 시 수정 SQL 1회 재시도
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import text
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.database.t10_schema_rag import search_schema
from tools.output.t20_trace_logger import log_tool_call

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None

SUPPORTED_DIALECTS = {"mysql", "sqlite"}
DEFAULT_LIMIT = 100

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|GRANT|REVOKE|EXEC|EXECUTE|CALL)\b",
    re.IGNORECASE,
)

CODE_BLOCK_PATTERN = re.compile(r"```(?:sql)?|```", re.IGNORECASE)


# ============================================================
# LLM
# ============================================================

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _safe_llm_invoke(messages: list[Any]) -> str:
    llm = _get_llm()
    response = llm.invoke(messages)

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ============================================================
# SQL safety helpers
# ============================================================

def _strip_code_block(text_value: str) -> str:
    return CODE_BLOCK_PATTERN.sub("", text_value).strip()


def _normalize_sql(sql: str) -> str:
    sql = _strip_code_block(sql)
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def _has_multiple_statements(sql: str) -> bool:
    """
    세미콜론 기반 다중 문장 차단.
    끝 세미콜론은 normalize에서 제거했으므로,
    내부 세미콜론이 있으면 다중 문장으로 간주.
    """
    return ";" in sql


def _starts_with_select_or_cte(sql: str) -> bool:
    stripped = sql.strip().lstrip("(").strip().lower()
    return stripped.startswith("select") or stripped.startswith("with")


def _is_select_only(sql: str) -> tuple[bool, str | None]:
    """
    SELECT / WITH ... SELECT만 허용.
    DDL/DML 및 다중 문장 차단.
    """
    normalized = _normalize_sql(sql)

    if not normalized:
        return False, "빈 SQL입니다."

    if _has_multiple_statements(normalized):
        return False, "다중 SQL 문장은 허용되지 않습니다."

    if not _starts_with_select_or_cte(normalized):
        return False, "SELECT 또는 WITH-SELECT 쿼리만 허용됩니다."

    if FORBIDDEN_SQL_PATTERN.search(normalized):
        return False, "DDL/DML/관리 구문이 포함되어 있습니다."

    return True, None


def _ensure_limit(sql: str, dialect: str, limit: int = DEFAULT_LIMIT) -> str:
    """
    LIMIT이 없으면 기본 LIMIT 추가.
    """
    normalized = _normalize_sql(sql)

    # 이미 LIMIT이 있으면 그대로 사용
    if re.search(r"\blimit\s+\d+\b", normalized, re.IGNORECASE):
        return normalized

    # sqlite/mysql 모두 LIMIT 지원
    return f"{normalized}\nLIMIT {limit}"


# ============================================================
# SQL generation
# ============================================================

def _generate_sql(question: str, schema_context: str, dialect: str = "mysql") -> str:
    """
    LLM으로 SQL 생성
    """
    system = (
        f"너는 {dialect} SQL 전문가다.\n"
        "사용자 질문에 답하는 SELECT SQL만 작성해라.\n"
        "반드시 단일 SQL 문장만 반환해라.\n"
        f"기본적으로 LIMIT {DEFAULT_LIMIT} 이하를 사용해라.\n"
        "DDL, DML, 주석, 설명, 마크다운 금지.\n\n"
        f"스키마:\n{schema_context}"
    )

    raw = _safe_llm_invoke([
        SystemMessage(content=system),
        HumanMessage(content=question),
    ])

    return _normalize_sql(raw)


def _fix_sql(original_sql: str, error_msg: str, schema_context: str, dialect: str) -> str:
    """
    에러 메시지를 참고해서 SQL 수정
    """
    system = (
        f"아래 SQL에서 에러가 발생했다. 이를 수정한 {dialect} SQL만 반환해라.\n"
        "반드시 SELECT SQL 한 문장만 반환해라.\n"
        f"기본적으로 LIMIT {DEFAULT_LIMIT} 이하를 사용해라.\n"
        "설명, 마크다운, 주석 금지.\n\n"
        f"스키마:\n{schema_context}"
    )

    msg = f"SQL: {original_sql}\n에러: {error_msg}"

    raw = _safe_llm_invoke([
        SystemMessage(content=system),
        HumanMessage(content=msg),
    ])

    return _normalize_sql(raw)


# ============================================================
# Main
# ============================================================

def run_sql(
    session_id: str,
    question: str,
    db_url: str,
    dialect: str = "mysql",
    row_limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """
    자연어 질문을 SQL로 변환하고 실행

    Args:
        session_id: 현재 세션 ID
        question:   자연어 질문
        db_url:     SQLAlchemy DB URL
                    예) "mysql+pymysql://user:pw@host/db"
                        "sqlite:///path/to/db.sqlite"
        dialect:    "mysql" | "sqlite"
        row_limit:  결과 최대 행 수

    Returns:
        {
            "sql": str,
            "result": list[dict],
            "row_count": int,
            "error": str | None,
        }
    """
    start_ts = time.perf_counter()

    if dialect not in SUPPORTED_DIALECTS:
        error = f"지원하지 않는 dialect입니다: {dialect}"
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=None,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": "", "result": [], "row_count": 0, "error": error}

    # 1) 스키마 검색
    try:
        schema_info = search_schema(question)
        schema_context = schema_info.get("context", "") if isinstance(schema_info, dict) else ""
    except Exception as exc:
        error = f"스키마 검색 실패: {exc}"
        logger.exception("[T-09] schema search failed: %s", exc)
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=None,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": "", "result": [], "row_count": 0, "error": error}

    if not schema_context.strip():
        error = "관련 스키마 정보를 찾지 못했습니다."
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=None,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": "", "result": [], "row_count": 0, "error": error}

    # 2) SQL 생성
    try:
        sql = _generate_sql(question, schema_context, dialect)
        sql = _ensure_limit(sql, dialect, row_limit)
    except Exception as exc:
        error = f"SQL 생성 실패: {exc}"
        logger.exception("[T-09] SQL generation failed: %s", exc)
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=None,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": "", "result": [], "row_count": 0, "error": error}

    # 3) 1차 안전성 검증
    ok, validation_error = _is_select_only(sql)
    if not ok:
        error = f"허용되지 않는 SQL입니다: {validation_error} / SQL: {sql}"
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=sql,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": sql, "result": [], "row_count": 0, "error": error}

    # 4) 실행
    exec_result = _execute_sql(db_url=db_url, sql=sql)
    if exec_result["error"] is None:
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=sql,
            result=exec_result["result"],
            error=None,
            dialect=dialect,
            row_count=exec_result["row_count"],
            latency_ms=_elapsed_ms(start_ts),
        )
        return exec_result

    # 5) 실패 시 수정 SQL 재생성
    first_error = exec_result["error"]
    try:
        fix_sql = _fix_sql(sql, first_error, schema_context, dialect)
        fix_sql = _ensure_limit(fix_sql, dialect, row_limit)
    except Exception as exc:
        error = f"SQL 수정 실패: {exc}"
        logger.exception("[T-09] SQL fix generation failed: %s", exc)
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=sql,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": sql, "result": [], "row_count": 0, "error": error}

    # 6) 수정 SQL 재검증
    ok, validation_error = _is_select_only(fix_sql)
    if not ok:
        error = f"수정된 SQL도 허용되지 않습니다: {validation_error} / SQL: {fix_sql}"
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=fix_sql,
            result=[],
            error=error,
            dialect=dialect,
            latency_ms=_elapsed_ms(start_ts),
        )
        return {"sql": fix_sql, "result": [], "row_count": 0, "error": error}

    # 7) 수정 SQL 실행
    retry_result = _execute_sql(db_url=db_url, sql=fix_sql)
    if retry_result["error"] is None:
        _log_sql_event(
            session_id=session_id,
            question=question,
            sql=fix_sql,
            result=retry_result["result"],
            error=None,
            dialect=dialect,
            row_count=retry_result["row_count"],
            latency_ms=_elapsed_ms(start_ts),
        )
        return retry_result

    # 8) 최종 실패
    final_error = retry_result["error"]
    _log_sql_event(
        session_id=session_id,
        question=question,
        sql=fix_sql,
        result=[],
        error=final_error,
        dialect=dialect,
        latency_ms=_elapsed_ms(start_ts),
    )
    return {"sql": fix_sql, "result": [], "row_count": 0, "error": final_error}


# ============================================================
# Execution / logging helpers
# ============================================================

def _execute_sql(db_url: str, sql: str) -> dict[str, Any]:
    """
    SQL 실행
    """
    try:
        engine = sqlalchemy.create_engine(db_url)

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        result = df.to_dict(orient="records")
        return {
            "sql": sql,
            "result": result,
            "row_count": len(df),
            "error": None,
        }

    except Exception as exc:
        logger.exception("[T-09] SQL execution failed: %s", exc)
        return {
            "sql": sql,
            "result": [],
            "row_count": 0,
            "error": str(exc),
        }


def _elapsed_ms(start_ts: float) -> int:
    return int((time.perf_counter() - start_ts) * 1000)


def _log_sql_event(
    session_id: str,
    question: str,
    sql: str | None,
    result: list[dict[str, Any]],
    error: str | None,
    dialect: str,
    row_count: int | None = None,
    latency_ms: int | None = None,
) -> None:
    """
    trace logger용 공통 래퍼
    """
    try:
        payload = {
            "question": question,
            "dialect": dialect,
        }
        if sql is not None:
            payload["sql"] = sql

        output = {
            "row_count": row_count if row_count is not None else len(result),
            "result_preview": result[:10],
        }

        log_tool_call(
            session_id=session_id,
            tool_name="sql_tool",
            params=payload,
            result=output if error is None else None,
            error=error,
            tool_version="t09_v2",
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.exception("[T-09] trace logging failed: %s", exc)