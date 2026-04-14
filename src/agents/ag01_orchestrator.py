"""
AG-01 Orchestrator Agent
사용자 요청 분석 → 분석 계획 수립 → Sub-Agent 라우팅 결정

역할:
- T-08: 업로드 파일 처리 (필요 시)
- T-22: 사용자 입력에서 파라미터 추출
- T-15: 분석 실행 계획 수립
- T-16: HITL ① 계획 승인 트리거
- T-20: 모든 단계 SQLite + MD 로깅
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GEMINI_MODEL, GOOGLE_API_KEY
from state import GraphState
from tools.control.t15_plan_parser import parse_plan
from tools.control.t22_param_parser import parse_params
from tools.output.t20_trace_logger import log_tool_call, log_final_response
from tools.database.sqlite_store import TraceStore

_store = TraceStore()
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _init_session(state: GraphState) -> str:
    """세션 초기화 — 없으면 신규 생성"""
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    _store.create_session(
        session_id=session_id,
        task_type="feature_engineering_analysis",
        initial_input=state.get("user_input", ""),
    )
    return session_id


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def orchestrator_node(state: GraphState) -> dict[str, Any]:
    """
    AG-01 메인 노드

    수행 작업:
    1. 세션 초기화
    2. T-22 파라미터 추출
    3. T-15 분석 계획 수립
    4. HITL ① 필요 여부 설정
    """
    import time
    session_id = _init_session(state)
    user_input = state.get("user_input", "")
    data_meta  = state.get("data_meta", {})

    # ── Step 1: T-22 파라미터 추출 ──────────────────────────────────
    t0 = time.time()
    param_result = parse_params(user_input)
    latency_ms   = int((time.time() - t0) * 1000)

    log_tool_call(
        session_id=session_id,
        tool_name="T-22_param_parser",
        params={"user_input": user_input},
        result=param_result,
        latency_ms=latency_ms,
    )

    # 경고 메시지 로깅
    for warning in param_result.get("warnings", []):
        log_tool_call(
            session_id=session_id,
            tool_name="T-22_param_parser_warning",
            params={},
            result={"warning": warning},
        )

    # ── Step 2: T-15 분석 계획 수립 ─────────────────────────────────
    t0 = time.time()
    plan = parse_plan(user_input, data_meta)
    latency_ms = int((time.time() - t0) * 1000)

    # 파라미터 파서 결과를 계획에 병합
    extracted_params = param_result.get("params", {})
    if extracted_params:
        for stage in plan.get("stages", []):
            if stage in plan["params"]:
                plan["params"][stage].update(extracted_params)
            else:
                plan["params"][stage] = extracted_params

    log_tool_call(
        session_id=session_id,
        tool_name="T-15_plan_parser",
        params={"user_input": user_input, "data_meta_path": data_meta.get("path", "")},
        result=plan,
        latency_ms=latency_ms,
    )

    # ── Step 3: HITL ① 계획 승인 설정 ──────────────────────────────
    # hitl_required=True → graph.py에서 hitl_node로 라우팅
    log_tool_call(
        session_id=session_id,
        tool_name="AG-01_orchestrator",
        params={"user_input": user_input},
        result={"plan": plan, "hitl_required": True},
    )

    return {
        "session_id":     session_id,
        "execution_plan": plan,
        "hitl_required":  True,   # HITL ① 트리거
        "agent_results":  state.get("agent_results", {}),
        "hitl_history":   state.get("hitl_history", []),
    }


def orchestrator_after_hitl_node(state: GraphState) -> dict[str, Any]:
    """
    HITL ① 승인 후 처리 노드
    - 승인: 그대로 진행
    - 수정: 계획 재수립
    - 재실행: orchestrator_node 재실행
    """
    session_id  = state.get("session_id", "")
    hitl_history = state.get("hitl_history", [])
    last_hitl   = hitl_history[-1] if hitl_history else {}
    response    = last_hitl.get("response", "승인")
    modified    = last_hitl.get("modified_input", {})

    log_tool_call(
        session_id=session_id,
        tool_name="AG-01_after_hitl",
        params={"hitl_response": response},
        result={"modified_input": modified},
    )

    if response == "수정" and modified:
        # 수정된 입력으로 계획 재수립
        new_plan = parse_plan(
            modified.get("user_input", state.get("user_input", "")),
            state.get("data_meta", {}),
        )
        log_tool_call(
            session_id=session_id,
            tool_name="T-15_plan_reparse",
            params=modified,
            result=new_plan,
        )
        return {
            "execution_plan": new_plan,
            "hitl_required":  False,
        }

    return {"hitl_required": False}