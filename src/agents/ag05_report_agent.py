"""
AG-05 Report Generator Agent
분석 결과 통합 → 페르소나 응답 → 보고서 생성

역할:
- T-17: 페르소나 응답 생성기 (마케터/분석가 자동 판단)
- T-18: 보고서 생성기 (PDF/CSV)
- T-20: SQLite + MD 로깅 + 세션 완료 기록
"""
from __future__ import annotations

import time
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state import GraphState
from tools.output.t17_persona_responder import generate_response
from tools.output.t18_report_gen import generate_report
from tools.output.t20_trace_logger import log_tool_call, log_final_response
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def report_agent_node(state: GraphState) -> dict[str, Any]:
    """
    AG-05 메인 노드

    수행 작업:
    1. 전체 분석 결과 통합
    2. T-17: 페르소나 응답 생성
    3. T-18: PDF/CSV 보고서 생성
    4. T-20: 세션 완료 기록
    5. HITL ④ 최종 보고서 승인 트리거
    """
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})
    ag05_params   = plan.get("params", {}).get("AG-05", {})

    report_format = ag05_params.get("format", "pdf")

    # ── 분석 결과 통합 ───────────────────────────────────────────────
    ag04_result = agent_results.get("AG-04", {})
    ag03_result = agent_results.get("AG-03", {})

    combined = {
        "summary":         ag04_result.get("summary", ""),
        "insights":        ag04_result.get("insights", []),
        "actions":         ag04_result.get("actions", []),
        "viz_suggestions": ag04_result.get("viz_suggestions", []),
        "kpi_result":      ag03_result.get("kpi_result", {}),
        "feature_ranking": ag04_result.get("feature_importance", {}).get("final_ranking", {}),
        "image_paths":     ag04_result.get("image_paths", []),
    }

    # ── Step 1: T-17 페르소나 응답 ───────────────────────────────────
    t0 = time.time()
    persona_result = generate_response(
        session_id=session_id,
        user_input=user_input,
        analysis_result=combined,
    )
    log_tool_call(
        session_id=session_id,
        tool_name="T-17_persona_responder",
        params={"user_input": user_input},
        result={"persona": persona_result.get("persona")},
        latency_ms=int((time.time() - t0) * 1000),
    )

    final_response = persona_result.get("response", "")

    # 최종 응답 별도 기록 (MD + SQLite)
    log_final_response(
        session_id=session_id,
        response=final_response,
        persona=persona_result.get("persona"),
    )

    # ── Step 2: T-18 보고서 생성 ─────────────────────────────────────
    t0 = time.time()
    report_result = generate_report(
        session_id=session_id,
        analysis_result=combined,
        output_format=report_format,
    )
    log_tool_call(
        session_id=session_id,
        tool_name="T-18_report_gen",
        params={"format": report_format},
        result=report_result,
        error=report_result.get("error") if not report_result.get("success") else None,
        latency_ms=int((time.time() - t0) * 1000),
    )

    # ── Step 3: 세션 완료 기록 ───────────────────────────────────────
    _store.update_session_summary(
        session_id=session_id,
        final_output_summary=combined.get("summary", "")[:500],
        status="completed",
    )

    # ── 결과 집계 ────────────────────────────────────────────────────
    result = {
        "persona":       persona_result.get("persona"),
        "report_path":   report_result.get("report_path", ""),
        "report_format": report_format,
        "success":       report_result.get("success", False),
    }

    log_tool_call(
        session_id=session_id,
        tool_name="AG-05_report_agent_complete",
        params={},
        result=result,
    )

    current_results = {**agent_results, "AG-05": result}

    return {
        "agent_results":  current_results,
        "final_response": final_response,
        "hitl_required":  True,   # HITL ④ 최종 승인 트리거
    }