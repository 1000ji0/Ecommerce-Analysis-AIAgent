"""
AG-03 SQL Agent
자연어 질의 → SQL 변환·실행 → KPI 계산

역할:
- T-10: 스키마 RAG (관련 테이블·컬럼 검색)
- T-9: SQL Tool (NL→SQL 변환·실행)
- T-11: KPI Calculator (매출·CVR·ROAS 등)
- T-20: SQLite + MD 로깅
"""
from __future__ import annotations

import time
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_TARGET_COL
from state import GraphState
from tools.database.t10_schema_rag import search_schema
from tools.database.t9_sql_tool import run_sql
from tools.analytics.t11_kpi_calculator import calculate_kpi
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

import pandas as pd

_store = TraceStore()


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def sql_agent_node(state: GraphState) -> dict[str, Any]:
    """
    AG-03 메인 노드

    수행 작업:
    1. T-10: 스키마 RAG로 관련 테이블·컬럼 검색
    2. T-09: 자연어 → SQL 변환·실행
    3. T-11: KPI 계산
    """
    session_id  = state.get("session_id", "")
    user_input  = state.get("user_input", "")
    plan        = state.get("execution_plan", {})
    ag03_params = plan.get("params", {}).get("AG-03", {})

    db_url = ag03_params.get("db_url", "")
    if not db_url:
        error_msg = "AG-03: db_url이 없습니다. execution_plan.params.AG-03.db_url을 설정하세요."
        log_tool_call(session_id, "AG-03_sql_agent", {}, None, error=error_msg)
        current_results = state.get("agent_results", {})
        current_results["AG-03"] = {"error": error_msg, "sql": "", "kpi_result": {}}
        return {"agent_results": current_results}

    # ── Step 1: T-10 스키마 RAG ──────────────────────────────────────
    t0 = time.time()
    schema_info = search_schema(user_input)
    log_tool_call(
        session_id=session_id,
        tool_name="T-10_schema_rag",
        params={"question": user_input},
        result=schema_info,
        latency_ms=int((time.time() - t0) * 1000),
    )

    # ── Step 2: T-09 SQL Tool ────────────────────────────────────────
    t0 = time.time()
    sql_result = run_sql(
        session_id=session_id,
        question=user_input,
        db_url=db_url,
    )
    log_tool_call(
        session_id=session_id,
        tool_name="T-9_sql_tool",
        params={"question": user_input, "db_url": db_url},
        result={
            "sql":       sql_result.get("sql", ""),
            "row_count": sql_result.get("row_count", 0),
            "error":     sql_result.get("error"),
        },
        error=sql_result.get("error"),
        latency_ms=int((time.time() - t0) * 1000),
    )

    # SQL 실패 시 조기 반환
    if sql_result.get("error"):
        current_results = state.get("agent_results", {})
        current_results["AG-03"] = sql_result
        return {"agent_results": current_results}

    # ── Step 3: T-11 KPI Calculator ──────────────────────────────────
    kpi_result = {}
    if sql_result.get("result"):
        df = pd.DataFrame(sql_result["result"])
        t0 = time.time()
        kpi_result = calculate_kpi(
            session_id=session_id,
            df=df,
            segment_col=ag03_params.get("segment_col"),
        )
        log_tool_call(
            session_id=session_id,
            tool_name="T-11_kpi_calculator",
            params={"row_count": len(df)},
            result=kpi_result,
            latency_ms=int((time.time() - t0) * 1000),
        )

    # ── 결과 집계 ────────────────────────────────────────────────────
    result = {
        "sql":            sql_result.get("sql", ""),
        "row_count":      sql_result.get("row_count", 0),
        "query_result":   sql_result.get("result", []),
        "kpi_result":     kpi_result.get("kpi_result", {}),
        "segment_result": kpi_result.get("segment_result", {}),
    }

    log_tool_call(
        session_id=session_id,
        tool_name="AG-03_sql_agent_complete",
        params={"user_input": user_input},
        result=result,
    )

    current_results = state.get("agent_results", {})
    current_results["AG-03"] = result

    return {"agent_results": current_results}