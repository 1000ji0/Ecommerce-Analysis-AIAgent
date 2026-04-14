"""
graph.py — 대화형 에이전트 StateGraph

흐름:
  사용자 메시지 → AG-01 (의도 파악 + Sub-Agent 결정)
    → AG-02 (FE 파이프라인, 필요 시)
    → AG-03 (SQL/KPI, 필요 시)
    → AG-04 (분석/인사이트, 필요 시)
    → AG-05 (보고서, 필요 시)
  → AG-01 (결과 정리 + 응답 생성)
  → 다음 메시지 대기

HITL: 전체 분석 흐름에서만 4개 포인트 개입
"""
from __future__ import annotations
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state import GraphState
from agents.ag01_orchestrator import orchestrator_node, orchestrator_respond_node
from agents.ag02_fe_agent import fe_agent_node
from agents.ag03_sql_agent import sql_agent_node
from agents.ag04_insight_agent import insight_agent_node
from agents.ag05_report_agent import report_agent_node
from tools.control.t16_hitl import (
    hitl_plan_approval,
    hitl_preprocessing_check,
    hitl_analysis_check,
    hitl_final_approval,
)
from tools.output.t20_trace_logger import log_hitl, log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── HITL 노드 ────────────────────────────────────────────────────────

def hitl_plan_node(state: GraphState) -> dict[str, Any]:
    """HITL ① 분석 계획 승인"""
    session_id = state.get("session_id", "")
    plan       = state.get("execution_plan", {})

    result  = hitl_plan_approval(session_id=session_id, plan=plan)
    history = state.get("hitl_history", [])
    history.append({
        "point":          "HITL-①-계획승인",
        "response":       result["response"],
        "modified_input": result["modified_input"],
        "timestamp":      _now(),
    })
    log_hitl(session_id, "HITL-①-계획승인",
             "분석 계획을 확인해주세요.", result["response"],
             decision=result["response"])
    return {"hitl_history": history, "hitl_required": False}


def hitl_preprocess_node(state: GraphState) -> dict[str, Any]:
    """HITL ② 전처리 결과 확인"""
    session_id  = state.get("session_id", "")
    ag02_result = state.get("agent_results", {}).get("AG-02", {})

    result  = hitl_preprocessing_check(session_id=session_id, result=ag02_result)
    history = state.get("hitl_history", [])
    history.append({
        "point":     "HITL-②-전처리확인",
        "response":  result["response"],
        "timestamp": _now(),
    })
    log_hitl(session_id, "HITL-②-전처리확인",
             "전처리 결과를 확인해주세요.", result["response"],
             decision=result["response"])
    return {"hitl_history": history, "hitl_required": False}


def hitl_analysis_node(state: GraphState) -> dict[str, Any]:
    """HITL ③ Feature 선정 확인"""
    session_id  = state.get("session_id", "")
    ag04_result = state.get("agent_results", {}).get("AG-04", {})

    result  = hitl_analysis_check(session_id=session_id, result=ag04_result)
    history = state.get("hitl_history", [])
    history.append({
        "point":     "HITL-③-Feature선정확인",
        "response":  result["response"],
        "timestamp": _now(),
    })
    log_hitl(session_id, "HITL-③-Feature선정확인",
             "분석 결과 및 Feature 선정을 확인해주세요.", result["response"],
             decision=result["response"])
    return {"hitl_history": history, "hitl_required": False}


def hitl_final_node(state: GraphState) -> dict[str, Any]:
    """HITL ④ 최종 보고서 승인 — 승인 시에만 LTM 저장"""
    session_id  = state.get("session_id", "")
    ag05_result = state.get("agent_results", {}).get("AG-05", {})
    report_path = ag05_result.get("report_path", "")
    summary     = state.get("agent_results", {}).get("AG-04", {}).get("summary", "")

    result  = hitl_final_approval(session_id=session_id,
                                   report_path=report_path,
                                   report_summary=summary)
    history = state.get("hitl_history", [])
    history.append({
        "point":     "HITL-④-최종승인",
        "response":  result["response"],
        "timestamp": _now(),
    })
    log_hitl(session_id, "HITL-④-최종승인",
             "최종 보고서를 확인하고 승인해주세요.", result["response"],
             decision=result["response"])

    # LTM 저장 — 최종 승인 시에만
    if result["response"] == "승인":
        _store.update_session_summary(
            session_id=session_id,
            final_output_summary=summary[:500],
            status="completed",
        )
        log_tool_call(session_id, "LTM_저장", {}, {"status": "saved"})

    return {"hitl_history": history, "hitl_required": False}


# ── 조건부 엣지 ──────────────────────────────────────────────────────

def route_orchestrator(state: GraphState) -> str:
    """
    AG-01이 결정한 next_agent 기반으로 라우팅
    next_agent: "AG-02" | "AG-03" | "AG-04" | "AG-05" | "respond"
    """
    next_agent = state.get("next_agent", "respond")

    routes = {
        "AG-02":   "fe_agent",
        "AG-03":   "sql_agent",
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }
    return routes.get(next_agent, "orchestrator_respond")


def route_after_fe(state: GraphState) -> str:
    """AG-02 완료 후 — 전체 분석이면 HITL ②, 아니면 다음 agent"""
    plan       = state.get("execution_plan", {})
    is_full    = plan.get("is_full_pipeline", False)
    next_agent = state.get("next_agent", "respond")

    if is_full:
        return "hitl_preprocess"
    routes = {
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }
    return routes.get(next_agent, "orchestrator_respond")


def route_after_hitl_plan(state: GraphState) -> str:
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")
    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "orchestrator"
    return "fe_agent"


def route_after_hitl_preprocess(state: GraphState) -> str:
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")
    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "fe_agent"
    return "insight_agent"


def route_after_hitl_analysis(state: GraphState) -> str:
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")
    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "insight_agent"
    return "report_agent"


def route_after_hitl_final(state: GraphState) -> str:
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")
    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "report_agent"
    return "orchestrator_respond"


def route_after_ag(state: GraphState) -> str:
    """AG-03, AG-04, AG-05 완료 후 라우팅"""
    plan       = state.get("execution_plan", {})
    is_full    = plan.get("is_full_pipeline", False)
    next_agent = state.get("next_agent", "respond")

    # 전체 분석 파이프라인 중 AG-04 완료 → HITL ③
    current = state.get("current_agent", "")
    if is_full and current == "AG-04":
        return "hitl_analysis"
    if is_full and current == "AG-05":
        return "hitl_final"

    routes = {
        "AG-02":   "fe_agent",
        "AG-03":   "sql_agent",
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }
    return routes.get(next_agent, "orchestrator_respond")


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(GraphState)

    # 노드 등록
    builder.add_node("orchestrator",         orchestrator_node)
    builder.add_node("orchestrator_respond", orchestrator_respond_node)
    builder.add_node("fe_agent",             fe_agent_node)
    builder.add_node("sql_agent",            sql_agent_node)
    builder.add_node("insight_agent",        insight_agent_node)
    builder.add_node("report_agent",         report_agent_node)
    builder.add_node("hitl_plan",            hitl_plan_node)
    builder.add_node("hitl_preprocess",      hitl_preprocess_node)
    builder.add_node("hitl_analysis",        hitl_analysis_node)
    builder.add_node("hitl_final",           hitl_final_node)

    # 엣지
    builder.add_edge(START, "orchestrator")

    # AG-01 → Sub-Agent 라우팅
    builder.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "fe_agent":             "fe_agent",
            "sql_agent":            "sql_agent",
            "insight_agent":        "insight_agent",
            "report_agent":         "report_agent",
            "hitl_plan":            "hitl_plan",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ① → AG-02
    builder.add_conditional_edges(
        "hitl_plan",
        route_after_hitl_plan,
        {
            "orchestrator": "orchestrator",
            "fe_agent":     "fe_agent",
        },
    )

    # AG-02 완료 후
    builder.add_conditional_edges(
        "fe_agent",
        route_after_fe,
        {
            "hitl_preprocess":      "hitl_preprocess",
            "insight_agent":        "insight_agent",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ② → AG-04
    builder.add_conditional_edges(
        "hitl_preprocess",
        route_after_hitl_preprocess,
        {
            "orchestrator":  "orchestrator",
            "fe_agent":      "fe_agent",
            "insight_agent": "insight_agent",
        },
    )

    # AG-03 완료
    builder.add_conditional_edges(
        "sql_agent",
        route_after_ag,
        {
            "orchestrator_respond": "orchestrator_respond",
            "report_agent":         "report_agent",
        },
    )

    # AG-04 완료
    builder.add_conditional_edges(
        "insight_agent",
        route_after_ag,
        {
            "hitl_analysis":        "hitl_analysis",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ③ → AG-05
    builder.add_conditional_edges(
        "hitl_analysis",
        route_after_hitl_analysis,
        {
            "orchestrator":  "orchestrator",
            "insight_agent": "insight_agent",
            "report_agent":  "report_agent",
        },
    )

    # AG-05 완료
    builder.add_conditional_edges(
        "report_agent",
        route_after_ag,
        {
            "hitl_final":           "hitl_final",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ④ → 응답
    builder.add_conditional_edges(
        "hitl_final",
        route_after_hitl_final,
        {
            "orchestrator":         "orchestrator",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # 응답 → 종료 (대화 루프는 main.py에서 관리)
    builder.add_edge("orchestrator_respond", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


graph = build_graph()