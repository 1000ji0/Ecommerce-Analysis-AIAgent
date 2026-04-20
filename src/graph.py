"""
graph.py — 대화형 에이전트 StateGraph

HITL 레벨 (0~4):
  0 — 완전 수동:     모든 HITL 포인트 실행
  1 — 보조 자동화:   모든 HITL 포인트 실행
  2 — 부분 자동화:   전체 파이프라인 시에만 (기본값)
  3 — 조건부 자동화: 실행 안 함 (에러 감지 TODO)
  4 — 완전 자동화:   HITL 없음
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

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
from agents.ag04_react_agent import insight_agent_node
from agents.ag05_report_agent import report_agent_node
from human_in_the_loop import (
    HITLPoint,
    _generate_question_llm,
    _generate_llm_choices,
    _summarize_context,
)
from tools.output.t20_trace_logger import log_hitl, log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── HITL 레벨 체크 ───────────────────────────────────────────────────

def _should_run_hitl(state: GraphState, hitl_point: str) -> bool:
    """
    user_profile.hitl_level에 따라 HITL 실행 여부 결정
    Level 0, 1 → 항상 실행
    Level 2     → 전체 파이프라인 시에만 (기본값)
    Level 3, 4  → 실행 안 함
    """
    level   = state.get("user_profile", {}).get("hitl_level", 2)
    is_full = state.get("execution_plan", {}).get("is_full_pipeline", False)

    if level >= 3:  return False
    if level == 2:  return is_full
    return True  # level 0, 1


# ── HITL 공통 처리 ───────────────────────────────────────────────────

def _do_hitl(
    state: GraphState,
    hitl_point: str,
    task: str,
    task_context: dict,
) -> dict[str, Any]:
    """
    두 단계 HITL 처리:
    Phase A: LLM 질문 + 동적 선택지 → interrupt()
    Phase B: 결과 요약 → interrupt() → 승인/수정/재실행
    """
    session_id = state.get("session_id", "")

    # Phase A — 정보 수집
    question    = _generate_question_llm(task=task, task_context=task_context,
                                         hitl_point=hitl_point)
    llm_choices = _generate_llm_choices(task=task, task_context=task_context,
                                        hitl_point=hitl_point)
    log_tool_call(session_id, "HITL_phase_a_question",
                  {"task": task}, {"question": question, "choices": llm_choices})

    # Phase A interrupt
    phase_a_payload = {
        "phase":        "A",
        "hitl_point":   hitl_point,
        "llm_question": question,
        "llm_choices":  llm_choices,   # LLM 동적 선택지
        "input_type":   "free_text",
    }
    raw_a       = interrupt(phase_a_payload)
    user_answer = raw_a.get("user_answer", "") if isinstance(raw_a, dict) else str(raw_a)

    log_tool_call(session_id, "HITL_phase_a_collect",
                  {"question": question}, {"user_answer": user_answer})

    # 사용자 답변을 컨텍스트에 반영
    updated_context = {**task_context, "user_requirements": user_answer}
    summary         = _summarize_context(updated_context, hitl_point)
    updated_context["context_summary"] = summary

    # Phase B — 결과 검토
    phase_b_payload = {
        "phase":       "B",
        "hitl_point":  hitl_point,
        "message":     summary,
        "user_answer": user_answer,
        "options":     ["승인", "수정", "재실행"],
        "input_type":  "selection",
        "context":     updated_context,
    }
    raw_b = interrupt(phase_b_payload)

    if isinstance(raw_b, dict):
        response       = raw_b.get("response", "승인")
        modified_input = raw_b.get("modified_input", {})
    else:
        response, modified_input = str(raw_b), {}

    if response not in {"승인", "수정", "재실행"}:
        response = "승인"

    history = list(state.get("hitl_history", []))
    history.append({
        "point":          hitl_point,
        "response":       response,
        "user_answer":    user_answer,
        "modified_input": modified_input,
        "timestamp":      _now(),
    })

    log_hitl(session_id, hitl_point, question, response, decision=response)
    return {"hitl_history": history, "hitl_required": False}


# ── HITL 노드 ─────────────────────────────────────────────────────────

def hitl_plan_node(state: GraphState) -> dict[str, Any]:
    """HITL ① 분석 계획 승인"""
    if not _should_run_hitl(state, HITLPoint.PLAN.value):
        return {"hitl_history": state.get("hitl_history", []), "hitl_required": False}
    plan = state.get("execution_plan", {})
    return _do_hitl(
        state=state,
        hitl_point=HITLPoint.PLAN.value,
        task="분석 계획 수립",
        task_context={
            "stages":      plan.get("stages", []),
            "params":      plan.get("params", {}),
            "description": plan.get("description", ""),
        },
    )


def hitl_preprocess_node(state: GraphState) -> dict[str, Any]:
    """HITL ② 전처리 결과 확인"""
    if not _should_run_hitl(state, HITLPoint.PREPROCESS.value):
        return {"hitl_history": state.get("hitl_history", []), "hitl_required": False}
    ag02 = state.get("agent_results", {}).get("AG-02", {})
    return _do_hitl(
        state=state,
        hitl_point=HITLPoint.PREPROCESS.value,
        task="데이터 전처리 및 Feature Engineering",
        task_context={
            "output_path":  ag02.get("output_path", ""),
            "stages_done":  ag02.get("stages_done", []),
            "row_count":    ag02.get("row_count", "?"),
            "col_count":    ag02.get("col_count", "?"),
            "removed_rows": ag02.get("removed_rows", 0),
        },
    )


def hitl_analysis_node(state: GraphState) -> dict[str, Any]:
    """HITL ③ Feature 선정 확인"""
    if not _should_run_hitl(state, HITLPoint.FEATURE.value):
        return {"hitl_history": state.get("hitl_history", []), "hitl_required": False}
    ag04 = state.get("agent_results", {}).get("AG-04", {})
    fi   = ag04.get("feature_importance", {})
    return _do_hitl(
        state=state,
        hitl_point=HITLPoint.FEATURE.value,
        task="변수 중요도 분석 및 Feature 선정",
        task_context={
            "task":          fi.get("task", "") if isinstance(fi, dict) else "",
            "final_ranking": fi.get("final_ranking", {}) if isinstance(fi, dict) else {},
            "explanation":   fi.get("explanation", "")[:200] if isinstance(fi, dict) else "",
            "insights":      ag04.get("insights", [])[:3],
            "summary":       ag04.get("summary", "")[:200],
        },
    )


def hitl_final_node(state: GraphState) -> dict[str, Any]:
    """HITL ④ 최종 보고서 승인 — 승인 시에만 LTM 저장"""
    session_id = state.get("session_id", "")
    ag05       = state.get("agent_results", {}).get("AG-05", {})
    ag04       = state.get("agent_results", {}).get("AG-04", {})
    summary    = ag04.get("summary", "")

    # Level 4는 자동 승인
    if not _should_run_hitl(state, HITLPoint.FINAL.value):
        _store.update_session_summary(
            session_id=session_id,
            final_output_summary=summary[:500],
            status="completed",
        )
        return {"hitl_history": state.get("hitl_history", []), "hitl_required": False}

    result = _do_hitl(
        state=state,
        hitl_point=HITLPoint.FINAL.value,
        task="최종 보고서 생성",
        task_context={
            "report_path":    ag05.get("report_path", ""),
            "report_format":  ag05.get("report_format", "docx"),
            "report_summary": summary[:300],
        },
    )

    history = result.get("hitl_history", [])
    last    = history[-1] if history else {}
    if last.get("response") == "승인":
        _store.update_session_summary(
            session_id=session_id,
            final_output_summary=summary[:500],
            status="completed",
        )
        log_tool_call(session_id, "LTM_저장", {}, {"status": "saved"})

    return result


# ── 조건부 엣지 ──────────────────────────────────────────────────────

def _last_hitl_response(state: GraphState) -> str:
    history = state.get("hitl_history", [])
    return history[-1].get("response", "승인") if history else "승인"


def route_orchestrator(state: GraphState) -> str:
    next_agent = state.get("next_agent", "respond")
    return {
        "AG-02":     "fe_agent",
        "AG-03":     "sql_agent",
        "AG-04":     "insight_agent",
        "AG-05":     "report_agent",
        "hitl_plan": "hitl_plan",
        "respond":   "orchestrator_respond",
    }.get(next_agent, "orchestrator_respond")


def route_after_hitl_plan(state: GraphState) -> str:
    r = _last_hitl_response(state)
    return "orchestrator" if r in ("재실행", "수정") else "fe_agent"


def route_after_fe(state: GraphState) -> str:
    is_full = state.get("execution_plan", {}).get("is_full_pipeline", False)
    if is_full:
        return "hitl_preprocess"
    return {
        "AG-04":   "insight_agent",
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }.get(state.get("next_agent", "respond"), "orchestrator_respond")


def route_after_hitl_preprocess(state: GraphState) -> str:
    r = _last_hitl_response(state)
    if r == "재실행": return "orchestrator"
    if r == "수정":   return "fe_agent"
    return "insight_agent"


def route_after_insight(state: GraphState) -> str:
    is_full = state.get("execution_plan", {}).get("is_full_pipeline", False)
    if is_full:
        return "hitl_analysis"
    return {
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }.get(state.get("next_agent", "respond"), "orchestrator_respond")


def route_after_hitl_analysis(state: GraphState) -> str:
    r = _last_hitl_response(state)
    if r == "재실행": return "orchestrator"
    if r == "수정":   return "insight_agent"
    return "report_agent"


def route_after_report(state: GraphState) -> str:
    is_full = state.get("execution_plan", {}).get("is_full_pipeline", False)
    if is_full:
        return "hitl_final"
    return "orchestrator_respond"


def route_after_hitl_final(state: GraphState) -> str:
    r = _last_hitl_response(state)
    if r == "재실행": return "orchestrator"
    if r == "수정":   return "report_agent"
    return "orchestrator_respond"


def route_after_sql(state: GraphState) -> str:
    return {
        "AG-05":   "report_agent",
        "respond": "orchestrator_respond",
    }.get(state.get("next_agent", "respond"), "orchestrator_respond")


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(GraphState)

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

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges("orchestrator", route_orchestrator, {
        "fe_agent":             "fe_agent",
        "sql_agent":            "sql_agent",
        "insight_agent":        "insight_agent",
        "report_agent":         "report_agent",
        "hitl_plan":            "hitl_plan",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_conditional_edges("hitl_plan", route_after_hitl_plan, {
        "orchestrator": "orchestrator", "fe_agent": "fe_agent",
    })
    builder.add_conditional_edges("fe_agent", route_after_fe, {
        "hitl_preprocess":      "hitl_preprocess",
        "insight_agent":        "insight_agent",
        "report_agent":         "report_agent",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_conditional_edges("hitl_preprocess", route_after_hitl_preprocess, {
        "orchestrator":  "orchestrator",
        "fe_agent":      "fe_agent",
        "insight_agent": "insight_agent",
    })
    builder.add_conditional_edges("insight_agent", route_after_insight, {
        "hitl_analysis":        "hitl_analysis",
        "report_agent":         "report_agent",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_conditional_edges("hitl_analysis", route_after_hitl_analysis, {
        "orchestrator":  "orchestrator",
        "insight_agent": "insight_agent",
        "report_agent":  "report_agent",
    })
    builder.add_conditional_edges("report_agent", route_after_report, {
        "hitl_final":           "hitl_final",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_conditional_edges("hitl_final", route_after_hitl_final, {
        "orchestrator":         "orchestrator",
        "report_agent":         "report_agent",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_conditional_edges("sql_agent", route_after_sql, {
        "report_agent":         "report_agent",
        "orchestrator_respond": "orchestrator_respond",
    })
    builder.add_edge("orchestrator_respond", END)

    return builder.compile(checkpointer=MemorySaver())


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


graph = build_graph()