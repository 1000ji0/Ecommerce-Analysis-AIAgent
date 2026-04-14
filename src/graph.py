"""
graph.py
LangGraph StateGraph 조립

노드 구성:
  orchestrator       → AG-01: 계획 수립
  hitl_plan          → HITL ①: 계획 승인
  orchestrator_post  → AG-01: HITL 후 처리
  fe_agent           → AG-02: Feature Engineering
  sql_agent          → AG-03: SQL 조회 (선택적)
  hitl_preprocess    → HITL ②: 전처리 확인
  insight_agent      → AG-04: EDA·인사이트
  hitl_analysis      → HITL ③: 분석 결과 확인
  report_agent       → AG-05: 보고서 생성
  hitl_final         → HITL ④: 최종 승인
  end                → 종료

HITL 분기:
  승인   → 다음 노드로 진행
  수정   → 현재 단계 재처리
  재실행 → orchestrator로 돌아가서 전체 재실행
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state import GraphState
from agents.ag01_orchestrator import orchestrator_node, orchestrator_after_hitl_node
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
from tools.output.t20_trace_logger import log_hitl
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── HITL 노드 함수들 ─────────────────────────────────────────────────

def hitl_plan_node(state: GraphState) -> dict[str, Any]:
    """HITL ① 분석 계획 승인"""
    session_id = state.get("session_id", "")
    plan       = state.get("execution_plan", {})

    result = hitl_plan_approval(session_id=session_id, plan=plan)

    history = state.get("hitl_history", [])
    history.append({
        "point":          "HITL-①-계획승인",
        "response":       result["response"],
        "modified_input": result["modified_input"],
    })

    log_hitl(
        session_id=session_id,
        hitl_point="HITL-①-계획승인",
        message="분석 계획을 확인하고 승인해주세요.",
        response=result["response"],
        decision=result["response"],
    )

    return {
        "hitl_history":  history,
        "hitl_required": False,
    }


def hitl_preprocess_node(state: GraphState) -> dict[str, Any]:
    """HITL ② 전처리 결과 확인"""
    session_id   = state.get("session_id", "")
    agent_results = state.get("agent_results", {})
    ag02_result  = agent_results.get("AG-02", {})

    result = hitl_preprocessing_check(session_id=session_id, result=ag02_result)

    history = state.get("hitl_history", [])
    history.append({
        "point":          "HITL-②-전처리확인",
        "response":       result["response"],
        "modified_input": result["modified_input"],
    })

    log_hitl(
        session_id=session_id,
        hitl_point="HITL-②-전처리확인",
        message="전처리 결과를 확인해주세요.",
        response=result["response"],
        decision=result["response"],
    )

    return {
        "hitl_history":  history,
        "hitl_required": False,
    }


def hitl_analysis_node(state: GraphState) -> dict[str, Any]:
    """HITL ③ 분석 결과 확인"""
    session_id   = state.get("session_id", "")
    agent_results = state.get("agent_results", {})
    ag04_result  = agent_results.get("AG-04", {})

    result = hitl_analysis_check(session_id=session_id, result=ag04_result)

    history = state.get("hitl_history", [])
    history.append({
        "point":          "HITL-③-결과확인",
        "response":       result["response"],
        "modified_input": result["modified_input"],
    })

    log_hitl(
        session_id=session_id,
        hitl_point="HITL-③-결과확인",
        message="분석 결과를 확인해주세요.",
        response=result["response"],
        decision=result["response"],
    )

    return {
        "hitl_history":  history,
        "hitl_required": False,
    }


def hitl_final_node(state: GraphState) -> dict[str, Any]:
    """HITL ④ 최종 보고서 승인"""
    session_id   = state.get("session_id", "")
    agent_results = state.get("agent_results", {})
    report_path  = agent_results.get("AG-05", {}).get("report_path", "")
    summary      = agent_results.get("AG-04", {}).get("summary", "")

    result = hitl_final_approval(
        session_id=session_id,
        report_path=report_path,
        report_summary=summary,
    )

    history = state.get("hitl_history", [])
    history.append({
        "point":          "HITL-④-최종승인",
        "response":       result["response"],
        "modified_input": result["modified_input"],
    })

    log_hitl(
        session_id=session_id,
        hitl_point="HITL-④-최종승인",
        message="최종 보고서를 확인하고 승인해주세요.",
        response=result["response"],
        decision=result["response"],
    )

    # 최종 승인 시 세션 완료 기록
    if result["response"] == "승인":
        _store.update_session_summary(
            session_id=session_id,
            status="completed",
        )

    return {
        "hitl_history":  history,
        "hitl_required": False,
    }


# ── 조건부 엣지 함수들 ───────────────────────────────────────────────

def route_after_hitl_plan(state: GraphState) -> Literal[
    "orchestrator_post", "orchestrator"
]:
    """HITL ① 후 라우팅"""
    history = state.get("hitl_history", [])
    last    = history[-1] if history else {}
    response = last.get("response", "승인")

    if response == "재실행":
        return "orchestrator"        # 처음부터 재실행
    return "orchestrator_post"       # 승인 or 수정 → post 처리


def route_after_orchestrator_post(state: GraphState) -> Literal[
    "fe_agent", "sql_agent"
]:
    """계획에 따라 첫 번째 Sub-Agent 결정"""
    plan   = state.get("execution_plan", {})
    stages = plan.get("stages", ["AG-02"])

    if stages[0] == "AG-03":
        return "sql_agent"
    return "fe_agent"


def route_after_hitl_preprocess(state: GraphState) -> Literal[
    "insight_agent", "fe_agent", "orchestrator"
]:
    """HITL ② 후 라우팅"""
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")

    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "fe_agent"            # 전처리 재실행
    return "insight_agent"           # 승인 → 분석 단계로


def route_after_hitl_analysis(state: GraphState) -> Literal[
    "report_agent", "insight_agent", "orchestrator"
]:
    """HITL ③ 후 라우팅"""
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")

    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "insight_agent"       # 분석 재실행
    return "report_agent"            # 승인 → 보고서 생성


def route_after_hitl_final(state: GraphState) -> str:
    """HITL ④ 후 라우팅"""
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")

    if response == "재실행":
        return "orchestrator"
    if response == "수정":
        return "report_agent"        # 보고서 재생성
    return "__end__"                 # 승인 → 종료


def route_sql_or_insight(state: GraphState) -> Literal[
    "hitl_preprocess", "insight_agent"
]:
    """AG-03 완료 후 — 다음 stage 확인"""
    plan   = state.get("execution_plan", {})
    stages = plan.get("stages", [])

    # AG-02가 plan에 있으면 아직 FE 안 한 것 → hitl_preprocess로
    # AG-02 없이 AG-03만이면 바로 insight로
    if "AG-02" in stages:
        return "hitl_preprocess"
    return "insight_agent"


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_graph():
    """
    StateGraph 조립 및 컴파일

    Returns:
        compiled graph (invoke / stream 가능)
    """
    builder = StateGraph(GraphState)

    # ── 노드 등록 ────────────────────────────────────────────────────
    builder.add_node("orchestrator",      orchestrator_node)
    builder.add_node("hitl_plan",         hitl_plan_node)
    builder.add_node("orchestrator_post", orchestrator_after_hitl_node)
    builder.add_node("fe_agent",          fe_agent_node)
    builder.add_node("sql_agent",         sql_agent_node)
    builder.add_node("hitl_preprocess",   hitl_preprocess_node)
    builder.add_node("insight_agent",     insight_agent_node)
    builder.add_node("hitl_analysis",     hitl_analysis_node)
    builder.add_node("report_agent",      report_agent_node)
    builder.add_node("hitl_final",        hitl_final_node)

    # ── 엣지 연결 ────────────────────────────────────────────────────
    # 시작
    builder.add_edge(START, "orchestrator")

    # orchestrator → HITL ①
    builder.add_edge("orchestrator", "hitl_plan")

    # HITL ① → 승인/수정: orchestrator_post / 재실행: orchestrator
    builder.add_conditional_edges(
        "hitl_plan",
        route_after_hitl_plan,
        {
            "orchestrator_post": "orchestrator_post",
            "orchestrator":      "orchestrator",
        },
    )

    # orchestrator_post → fe_agent or sql_agent (계획 기반)
    builder.add_conditional_edges(
        "orchestrator_post",
        route_after_orchestrator_post,
        {
            "fe_agent":  "fe_agent",
            "sql_agent": "sql_agent",
        },
    )

    # fe_agent → HITL ②
    builder.add_edge("fe_agent", "hitl_preprocess")

    # sql_agent → HITL ② or insight_agent
    builder.add_conditional_edges(
        "sql_agent",
        route_sql_or_insight,
        {
            "hitl_preprocess": "hitl_preprocess",
            "insight_agent":   "insight_agent",
        },
    )

    # HITL ② → 승인: insight / 수정: fe_agent / 재실행: orchestrator
    builder.add_conditional_edges(
        "hitl_preprocess",
        route_after_hitl_preprocess,
        {
            "insight_agent": "insight_agent",
            "fe_agent":      "fe_agent",
            "orchestrator":  "orchestrator",
        },
    )

    # insight_agent → HITL ③
    builder.add_edge("insight_agent", "hitl_analysis")

    # HITL ③ → 승인: report / 수정: insight / 재실행: orchestrator
    builder.add_conditional_edges(
        "hitl_analysis",
        route_after_hitl_analysis,
        {
            "report_agent":  "report_agent",
            "insight_agent": "insight_agent",
            "orchestrator":  "orchestrator",
        },
    )

    # report_agent → HITL ④
    builder.add_edge("report_agent", "hitl_final")

    # HITL ④ → 승인: END / 수정: report / 재실행: orchestrator
    builder.add_conditional_edges(
        "hitl_final",
        route_after_hitl_final,
        {
            "__end__":       END,
            "report_agent":  "report_agent",
            "orchestrator":  "orchestrator",
        },
    )

    # ── 컴파일 ───────────────────────────────────────────────────────
    # MemorySaver: interrupt() 재개를 위한 체크포인터
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── 싱글톤 그래프 인스턴스 ───────────────────────────────────────────
graph = build_graph()