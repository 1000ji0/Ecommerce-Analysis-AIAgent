"""
AG-01 Orchestrator Agent
사용자 메시지 의도 파악 → 어떤 Sub-Agent를 호출할지 결정
Sub-Agent 결과 정리 → 최종 응답 생성
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY, get_session_output_dir
from state import GraphState
from tools.output.t20_trace_logger import log_tool_call, log_final_response
from tools.database.sqlite_store import TraceStore

_store = TraceStore()
_llm   = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _init_session(state: GraphState) -> str:
    session_id = state.get("session_id", "")
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    _store.create_session(
        session_id=session_id,
        task_type="ecommerce_analysis",
        initial_input=state.get("user_input", ""),
    )
    return session_id


# ── 의도 파악 시스템 프롬프트 ────────────────────────────────────────

INTENT_SYSTEM = """
너는 이커머스 데이터 분석 에이전트다.
사용자 요청이 무엇이든 가장 적합한 Sub-Agent를 판단해라.

판단 기준:
- 데이터 탐색, 확인, 통계, 시각화, 이상치, 분포, 상관관계, 결측값 등
  데이터를 직접 살펴보는 모든 요청 → AG-04
- 변수 중요도, 피처, 예측 관련 → AG-04
- 매출, KPI, SQL, DB 조회 → AG-03
- 전처리, FE, 파이프라인 → AG-02
- 보고서, PDF, 저장 → AG-05
- 전체 분석, 처음부터 끝까지 → FULL_PIPELINE
- 위 어디에도 해당 안 되는 잡담이나 인사 → NONE

애매하면 AG-04로 보내라. 데이터 관련 요청은 거의 다 AG-04다.

아래 JSON 형식으로만 반환해라. 마크다운 금지.
{
  "intent": "AG-02" | "AG-03" | "AG-04" | "AG-05" | "FULL_PIPELINE" | "NONE",
  "sub_intent": "구체적으로 뭘 원하는지 한 줄",
  "params": {
    "task": "eda" | "feature_importance" | "viz" | "insight",
    "question": "사용자 질문 그대로"
  }
}
"""


# ── 메인 노드 ────────────────────────────────────────────────────────

def orchestrator_node(state: GraphState) -> dict[str, Any]:
    """
    사용자 메시지 의도 파악 → Sub-Agent 결정
    state에 next_agent, execution_plan 설정
    """
    session_id = _init_session(state)
    user_input = state.get("user_input", "")
    data_meta  = state.get("data_meta", {})

    t0     = time.time()
    intent = _parse_intent(user_input, data_meta)
    log_tool_call(
        session_id=session_id,
        tool_name="AG-01_intent_parse",
        params={"user_input": user_input},
        result=intent,
        latency_ms=int((time.time() - t0) * 1000),
    )

    next_agent     = _map_intent_to_agent(intent)
    execution_plan = _build_plan(intent, state)

    log_tool_call(
        session_id=session_id,
        tool_name="AG-01_routing",
        params={"intent": intent.get("intent")},
        result={"next_agent": next_agent, "plan": execution_plan},
    )

    return {
        "session_id":     session_id,
        "next_agent":     next_agent,
        "current_agent":  "AG-01",
        "execution_plan": execution_plan,
        "hitl_required":  next_agent == "hitl_plan",
        "agent_results":  state.get("agent_results", {}),
        "hitl_history":   state.get("hitl_history", []),
    }


def orchestrator_respond_node(state: GraphState) -> dict[str, Any]:
    """
    Sub-Agent 결과를 받아 사용자에게 자연어 응답 생성
    """
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    agent_results = state.get("agent_results", {})

    t0       = time.time()
    response = _generate_response(user_input, agent_results)
    log_final_response(session_id=session_id, response=response)
    log_tool_call(
        session_id=session_id,
        tool_name="AG-01_respond",
        params={"user_input": user_input},
        result={"response_length": len(response)},
        latency_ms=int((time.time() - t0) * 1000),
    )

    return {
        "final_response": response,
        "next_agent":     "respond",
    }


# ── 내부 함수 ────────────────────────────────────────────────────────

def _parse_intent(user_input: str, data_meta: dict) -> dict:
    """LLM으로 의도 파악 — JSON 파싱 실패 시 AG-04 기본값"""
    llm  = _get_llm()
    cols = data_meta.get("preview", {}).get("columns", [])
    msg  = f"데이터 컬럼: {cols}\n사용자: {user_input}"

    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=msg),
    ])
    raw = re.sub(r"```(?:json)?|```", "", response.content.strip()).strip()

    try:
        result = json.loads(raw)
        # intent 키가 없거나 유효하지 않으면 AG-04 기본값
        valid = {"AG-02", "AG-03", "AG-04", "AG-05", "FULL_PIPELINE", "NONE"}
        if result.get("intent") not in valid:
            result["intent"] = "AG-04"
        return result
    except json.JSONDecodeError:
        # 파싱 실패 시 AG-04로 fallback (데이터 관련 요청이 대부분)
        return {
            "intent":     "AG-04",
            "sub_intent": user_input,
            "params":     {"task": "eda", "question": user_input},
        }


def _map_intent_to_agent(intent: dict) -> str:
    """intent → next_agent 문자열 변환"""
    mapping = {
        "AG-02":         "AG-02",
        "AG-03":         "AG-03",
        "AG-04":         "AG-04",
        "AG-05":         "AG-05",
        "FULL_PIPELINE": "hitl_plan",
        "NONE":          "respond",
    }
    return mapping.get(intent.get("intent", "AG-04"), "AG-04")


def _build_plan(intent: dict, state: GraphState) -> dict:
    """execution_plan 구성"""
    intent_type = intent.get("intent", "AG-04")
    params      = intent.get("params", {})
    existing    = state.get("execution_plan", {})

    if intent_type == "FULL_PIPELINE":
        return {
            "is_full_pipeline": True,
            "stages":           ["AG-02", "AG-04", "AG-05"],
            "params": {
                "AG-02": {},
                "AG-04": {"top_n": 5, "target_col": "TARGET"},
                "AG-05": {"format": "pdf"},
            },
            "description": "전체 분석 파이프라인 (FE → 인사이트 → 보고서)",
        }

    if intent_type == "AG-04":
        return {
            **existing,
            "is_full_pipeline": False,
            "ag04_task":   params.get("task", "eda"),
            "ag04_params": params,
        }

    return {
        **existing,
        "is_full_pipeline": False,
        "intent_params": params,
    }


def _generate_response(user_input: str, agent_results: dict) -> str:
    """Sub-Agent 결과 기반 자연어 응답 생성"""
    llm = _get_llm()

    summary_parts = []
    for agent_id, result in agent_results.items():
        if not result or "error" in result:
            continue

        if agent_id == "AG-04":
            # EDA 결과 (이상치, 결측값, 상관관계 등 포함)
            eda = result.get("eda_result", {})
            if eda:
                s = eda.get("summary", {})
                summary_parts.append(
                    f"EDA 결과: "
                    f"shape={s.get('shape')}, "
                    f"결측값={s.get('missing')}, "
                    f"이상치비율={s.get('outlier_ratio')}, "
                    f"타겟상관top5={s.get('target_corr_top5')}"
                )

            # 변수 중요도
            fi = result.get("feature_importance", {})
            if fi:
                summary_parts.append(
                    f"변수 중요도 (task={fi.get('task')}): "
                    f"final_ranking={fi.get('final_ranking')}, "
                    f"explanation={fi.get('explanation', '')[:200]}"
                )

            # 인사이트
            insights = result.get("insights", [])
            actions  = result.get("actions", [])
            if insights:
                summary_parts.append(f"인사이트: {insights}")
            if actions:
                summary_parts.append(f"액션 아이템: {actions}")

            # 차트
            charts = result.get("image_paths", [])
            if charts:
                summary_parts.append(f"차트 저장 경로: {charts}")

        elif agent_id == "AG-05":
            summary_parts.append(f"보고서 경로: {result.get('report_path', '')}")

        elif agent_id == "AG-03":
            kpi = result.get("kpi_result", {})
            if kpi:
                summary_parts.append(f"KPI 결과: {kpi}")

    context = "\n".join(summary_parts) if summary_parts else "분석 결과 없음"

    system = (
        "너는 이커머스 데이터 분석 에이전트다.\n"
        "분석 결과를 사용자에게 친절하고 명확하게 설명해라.\n"
        "수치가 있으면 구체적으로 언급해라.\n"
        "분석 결과가 없으면 무엇을 요청했는지 다시 물어봐라.\n"
        "한국어로 답해라. 마크다운 볼드 최소화."
    )
    msg = f"사용자 요청: {user_input}\n\n분석 결과:\n{context}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    return response.content.strip()