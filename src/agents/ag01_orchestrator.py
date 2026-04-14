"""
AG-01 Orchestrator Agent
- execution_plan이 이미 있으면 재수립 건너뜀 (insight/test 모드용)
"""
from __future__ import annotations

import uuid
import time
from datetime import datetime
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import GEMINI_MODEL, GOOGLE_API_KEY
from state import GraphState
from tools.control.t15_plan_parser import parse_plan
from tools.control.t22_param_parser import parse_params
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


def _init_session(state: GraphState) -> str:
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    _store.create_session(
        session_id=session_id,
        task_type="feature_engineering_analysis",
        initial_input=state.get("user_input", ""),
    )
    return session_id


def orchestrator_node(state: GraphState) -> dict[str, Any]:
    """
    AG-01 메인 노드
    execution_plan이 이미 설정되어 있으면 재수립 건너뜀
    """
    session_id    = _init_session(state)
    user_input    = state.get("user_input", "")
    data_meta     = state.get("data_meta", {})
    existing_plan = state.get("execution_plan", {})

    # ── plan이 이미 있으면 재수립 건너뜀 ────────────────────────────
    if existing_plan.get("stages"):
        log_tool_call(
            session_id=session_id,
            tool_name="AG-01_orchestrator_SKIP",
            params={"reason": "execution_plan already set"},
            result={"stages": existing_plan["stages"]},
        )
        return {
            "session_id":     session_id,
            "execution_plan": existing_plan,
            "hitl_required":  True,
            "agent_results":  state.get("agent_results", {}),
            "hitl_history":   state.get("hitl_history", []),
        }

    # ── T-22 파라미터 추출 ───────────────────────────────────────────
    t0           = time.time()
    param_result = parse_params(user_input)
    log_tool_call(
        session_id=session_id,
        tool_name="T-22_param_parser",
        params={"user_input": user_input},
        result=param_result,
        latency_ms=int((time.time() - t0) * 1000),
    )

    # ── T-15 분석 계획 수립 ──────────────────────────────────────────
    t0   = time.time()
    plan = parse_plan(user_input, data_meta)

    extracted_params = param_result.get("params", {})
    if extracted_params:
        for stage in plan.get("stages", []):
            plan["params"].setdefault(stage, {}).update(extracted_params)

    log_tool_call(
        session_id=session_id,
        tool_name="T-15_plan_parser",
        params={"user_input": user_input},
        result=plan,
        latency_ms=int((time.time() - t0) * 1000),
    )

    return {
        "session_id":     session_id,
        "execution_plan": plan,
        "hitl_required":  True,
        "agent_results":  state.get("agent_results", {}),
        "hitl_history":   state.get("hitl_history", []),
    }


def orchestrator_after_hitl_node(state: GraphState) -> dict[str, Any]:
    """HITL ① 승인 후 처리"""
    session_id   = state.get("session_id", "")
    hitl_history = state.get("hitl_history", [])
    last_hitl    = hitl_history[-1] if hitl_history else {}
    response     = last_hitl.get("response", "승인")
    modified     = last_hitl.get("modified_input", {})

    if response == "수정" and modified:
        new_plan = parse_plan(
            modified.get("user_input", state.get("user_input", "")),
            state.get("data_meta", {}),
        )
        log_tool_call(session_id, "T-15_plan_reparse", modified, new_plan)
        return {"execution_plan": new_plan, "hitl_required": False}

    return {"hitl_required": False}