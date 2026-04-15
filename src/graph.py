"""
graph.py — 대화형 에이전트 StateGraph

흐름:
  사용자 메시지 → AG-01 (의도 파악 + Sub-Agent 결정)
    → AG-02 (FE 파이프라인, 필요 시)
    → AG-03 (SQL/KPI, 필요 시)
    → AG-04 (분석/인사이트, 필요 시)
    → AG-05 (보고서, 필요 시)
  → AG-01 (결과 정리 + 응답 생성)

HITL: 전체 분석 흐름에서만 4개 포인트 개입
      각 HITL 노드는 human_in_the_loop.py의 워크플로를 통해 실행
      LLM 질문 생성 → 컨텍스트 제공 → interrupt() → 재개
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state import GraphState
from agents.ag01_orchestrator import orchestrator_node, orchestrator_respond_node
from agents.ag02_fe_agent import fe_agent_node
from agents.ag03_sql_agent import sql_agent_node
from agents.ag04_insight_agent import insight_agent_node
from agents.ag05_report_agent import report_agent_node
from human_in_the_loop import hitl_graph, HITLPoint, HITLState
from tools.output.t20_trace_logger import log_hitl, log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── HITL 노드 (human_in_the_loop.py 연동) ───────────────────────────

def _run_hitl_workflow(
    state: GraphState,
    hitl_point: str,
    task: str,
    task_context: dict,
) -> dict[str, Any]:
    """
    human_in_the_loop.py의 4단계 워크플로 실행
    1) LLM 질문 생성  2) 컨텍스트 제공  3) interrupt()  4) 재개

    Returns:
        {"response": str, "user_answer": str, "modified_input": dict}
    """
    session_id = state.get("session_id", "")
    # HITL 전용 thread_id (메인 그래프와 분리)
    hitl_config = {"configurable": {"thread_id": f"{session_id}_{hitl_point}"}}

    initial = HITLState(
        session_id=session_id,
        task=task,
        task_context=task_context,
        hitl_point=hitl_point,
    )

    # hitl_graph 실행 — interrupt() 발생 시 caller가 처리
    final = hitl_graph.invoke(initial, config=hitl_config)

    if isinstance(final, dict):
        final = HITLState(**final)

    response     = "승인"
    user_answer  = ""
    modified     = {}

    if final.hitl_response:
        response    = final.hitl_response.response
        user_answer = final.hitl_response.user_answer
        modified    = final.hitl_response.modified_input

    return {
        "response":       response,
        "user_answer":    user_answer,
        "modified_input": modified,
    }


def hitl_plan_node(state: GraphState) -> dict[str, Any]:
    """HITL ① 분석 계획 승인"""
    session_id = state.get("session_id", "")
    plan       = state.get("execution_plan", {})

    result = _run_hitl_workflow(
        state=state,
        hitl_point=HITLPoint.PLAN.value,
        task="분석 계획 수립",
        task_context={
            "stages":      plan.get("stages", []),
            "params":      plan.get("params", {}),
            "description": plan.get("description", ""),
        },
    )

    history = list(state.get("hitl_history", []))
    history.append({
        "point":       HITLPoint.PLAN.value,
        "response":    result["response"],
        "user_answer": result["user_answer"],
        "timestamp":   _now(),
    })

    log_hitl(session_id, HITLPoint.PLAN.value,
             "분석 계획을 확인해주세요.",
             result["response"], decision=result["response"])

    return {"hitl_history": history, "hitl_required": False}


def hitl_preprocess_node(state: GraphState) -> dict[str, Any]:
    """HITL ② 전처리 결과 확인"""
    session_id  = state.get("session_id", "")
    ag02_result = state.get("agent_results", {}).get("AG-02", {})

    result = _run_hitl_workflow(
        state=state,
        hitl_point=HITLPoint.PREPROCESS.value,
        task="데이터 전처리 및 Feature Engineering",
        task_context={
            "output_path":  ag02_result.get("output_path", ""),
            "stages_done":  ag02_result.get("stages_done", []),
            "row_count":    ag02_result.get("row_count", "?"),
            "col_count":    ag02_result.get("col_count", "?"),
            "removed_rows": ag02_result.get("removed_rows", 0),
        },
    )

    history = list(state.get("hitl_history", []))
    history.append({
        "point":       HITLPoint.PREPROCESS.value,
        "response":    result["response"],
        "user_answer": result["user_answer"],
        "timestamp":   _now(),
    })

    log_hitl(session_id, HITLPoint.PREPROCESS.value,
             "전처리 결과를 확인해주세요.",
             result["response"], decision=result["response"])

    return {"hitl_history": history, "hitl_required": False}


def hitl_analysis_node(state: GraphState) -> dict[str, Any]:
    """HITL ③ Feature 선정 확인"""
    session_id  = state.get("session_id", "")
    ag04_result = state.get("agent_results", {}).get("AG-04", {})
    fi          = ag04_result.get("feature_importance", {})

    result = _run_hitl_workflow(
        state=state,
        hitl_point=HITLPoint.FEATURE.value,
        task="변수 중요도 분석 및 Feature 선정",
        task_context={
            "task":          fi.get("task", ""),
            "final_ranking": fi.get("final_ranking", {}),
            "explanation":   fi.get("explanation", ""),
            "insights":      ag04_result.get("insights", [])[:3],
            "summary":       ag04_result.get("summary", ""),
        },
    )

    history = list(state.get("hitl_history", []))
    history.append({
        "point":       HITLPoint.FEATURE.value,
        "response":    result["response"],
        "user_answer": result["user_answer"],
        "timestamp":   _now(),
    })

    log_hitl(session_id, HITLPoint.FEATURE.value,
             "분석 결과 및 Feature 선정을 확인해주세요.",
             result["response"], decision=result["response"])

    return {"hitl_history": history, "hitl_required": False}


def hitl_final_node(state: GraphState) -> dict[str, Any]:
    """HITL ④ 최종 보고서 승인 — 승인 시에만 LTM 저장"""
    session_id  = state.get("session_id", "")
    ag05_result = state.get("agent_results", {}).get("AG-05", {})
    ag04_result = state.get("agent_results", {}).get("AG-04", {})
    summary     = ag04_result.get("summary", "")

    result = _run_hitl_workflow(
        state=state,
        hitl_point=HITLPoint.FINAL.value,
        task="최종 보고서 생성",
        task_context={
            "report_path":    ag05_result.get("report_path", ""),
            "report_format":  ag05_result.get("report_format", "pdf"),
            "report_summary": summary[:300],
        },
    )

    history = list(state.get("hitl_history", []))
    history.append({
        "point":       HITLPoint.FINAL.value,
        "response":    result["response"],
        "user_answer": result["user_answer"],
        "timestamp":   _now(),
    })

    log_hitl(session_id, HITLPoint.FINAL.value,
             "최종 보고서를 확인하고 승인해주세요.",
             result["response"], decision=result["response"])

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
    next_agent = state.get("next_agent", "respond")
    routes = {
        "AG-02":   "fe_agent",
        "AG-03":   "sql_agent",
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "hitl_plan": "hitl_plan",
        "respond": "orchestrator_respond",
    }
    return routes.get(next_agent, "orchestrator_respond")


def route_after_hitl_plan(state: GraphState) -> str:
    history  = state.get("hitl_history", [])
    last     = history[-1] if history else {}
    response = last.get("response", "승인")
    if response in ("재실행", "수정"):
        return "orchestrator"
    return "fe_agent"


def route_after_fe(state: GraphState) -> str:
    plan    = state.get("execution_plan", {})
    is_full = plan.get("is_full_pipeline", False)
    if is_full:
        return "hitl_preprocess"
    next_agent = state.get("next_agent", "respond")
    return {
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }.get(next_agent, "orchestrator_respond")


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
    plan       = state.get("execution_plan", {})
    is_full    = plan.get("is_full_pipeline", False)
    current    = state.get("current_agent", "")
    next_agent = state.get("next_agent", "respond")

    if is_full and current == "AG-04":
        return "hitl_analysis"
    if is_full and current == "AG-05":
        return "hitl_final"

    return {
        "AG-02":   "fe_agent",
        "AG-03":   "sql_agent",
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }.get(next_agent, "orchestrator_respond")


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

    # 기본 엣지
    builder.add_edge(START, "orchestrator")

    # AG-01 → Sub-Agent
    builder.add_conditional_edges(
        "orchestrator", route_orchestrator,
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
        "hitl_plan", route_after_hitl_plan,
        {"orchestrator": "orchestrator", "fe_agent": "fe_agent"},
    )

    # AG-02 → HITL ② or next
    builder.add_conditional_edges(
        "fe_agent", route_after_fe,
        {
            "hitl_preprocess":      "hitl_preprocess",
            "insight_agent":        "insight_agent",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ② → AG-04
    builder.add_conditional_edges(
        "hitl_preprocess", route_after_hitl_preprocess,
        {
            "orchestrator":  "orchestrator",
            "fe_agent":      "fe_agent",
            "insight_agent": "insight_agent",
        },
    )

    # AG-03 완료
    builder.add_conditional_edges(
        "sql_agent", route_after_ag,
        {
            "orchestrator_respond": "orchestrator_respond",
            "report_agent":         "report_agent",
        },
    )

    # AG-04 완료 → HITL ③ or next
    builder.add_conditional_edges(
        "insight_agent", route_after_ag,
        {
            "hitl_analysis":        "hitl_analysis",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ③ → AG-05
    builder.add_conditional_edges(
        "hitl_analysis", route_after_hitl_analysis,
        {
            "orchestrator":  "orchestrator",
            "insight_agent": "insight_agent",
            "report_agent":  "report_agent",
        },
    )

    # AG-05 완료 → HITL ④ or next
    builder.add_conditional_edges(
        "report_agent", route_after_ag,
        {
            "hitl_final":           "hitl_final",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    # HITL ④ → 응답
    builder.add_conditional_edges(
        "hitl_final", route_after_hitl_final,
        {
            "orchestrator":         "orchestrator",
            "report_agent":         "report_agent",
            "orchestrator_respond": "orchestrator_respond",
        },
    )

    builder.add_edge("orchestrator_respond", END)

    return builder.compile(checkpointer=MemorySaver())


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


graph = build_graph()