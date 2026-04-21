"""
AG-05 보고서 생성 에이전트

지원 포맷:
  docx — Word 문서 (python-docx)
  pdf  — PDF 문서 (reportlab)
    md   — Markdown 문서

사용자가 포맷 선택 가능:
  "보고서 만들어줘"         → 포맷 선택 프롬프트
  "word로 보고서 만들어줘"  → docx
  "pdf로 보고서 만들어줘"   → pdf
    "md로 보고서 만들어줘"    → md
"""
from __future__ import annotations
import re
import time
from typing import Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_TARGET_COL
from state import GraphState
from tools.output.t18_report_gen import generate_report
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


def report_agent_node(state: GraphState) -> dict[str, Any]:
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})
    is_full       = plan.get("is_full_pipeline", False)

    # ── 포맷 결정 ─────────────────────────────────────────────────
    report_format = _detect_format(user_input, plan)

    # ── 분석 결과 수집 ─────────────────────────────────────────────
    ag04 = agent_results.get("AG-04", {})
    ag03 = agent_results.get("AG-03", {})

    fi = ag04.get("feature_importance", {})
    if isinstance(fi, str):
        import json
        try:
            fi = json.loads(fi)
        except Exception:
            fi = {}

    analysis_result = {
        "summary":         ag04.get("summary", ag04.get("react_answer", ""))[:500],
        "insights":        ag04.get("insights", []),
        "actions":         ag04.get("actions", []),
        "viz_suggestions": ag04.get("viz_suggestions", []),
        "kpi_result":      ag03.get("kpi_result", {}),
        "feature_ranking": fi.get("final_ranking", {}),
        "image_paths":     ag04.get("image_paths", []),
    }

    # ── 보고서 생성 ────────────────────────────────────────────────
    t0     = time.time()
    result = generate_report(
        session_id=session_id,
        analysis_result=analysis_result,
        output_format=report_format,
    )
    log_tool_call(session_id, "AG-05_report_gen",
                  {"format": report_format},
                  {"success": result.get("success"),
                   "path":    result.get("report_path", "")},
                  latency_ms=int((time.time() - t0) * 1000))

    result["report_format"] = report_format
    next_agent = "hitl_final" if is_full else "respond"

    return {
        "agent_results": {**agent_results, "AG-05": result},
        "current_agent": "AG-05",
        "next_agent":    next_agent,
    }


def _detect_format(user_input: str, plan: dict) -> str:
    """
    사용자 입력 or 실행 계획에서 포맷 결정
    기본값: docx
    """
    # 실행 계획에 포맷이 명시된 경우
    plan_format = plan.get("params", {}).get("AG-05", {}).get("format", "")
    if plan_format in ("docx", "pdf", "csv", "md"):
        return plan_format

    q = user_input.lower()

    # PDF 키워드
    if any(k in q for k in ("pdf", "피디에프")):
        return "pdf"

    # Word/docx 키워드
    if any(k in q for k in ("word", "워드", "docx", "doc", "문서")):
        return "docx"

    # Markdown 키워드
    if any(k in q for k in ("md", "마크다운", "markdown")):
        return "md"

    # 기본값
    return "docx"