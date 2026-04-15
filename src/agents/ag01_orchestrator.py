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

from config import GEMINI_MODEL, GOOGLE_API_KEY
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
업로드된 CSV 데이터를 분석하는 것이 주요 역할이다.

Sub-Agent 판단 기준:

AG-04 (데이터 분석) — 아래 모든 경우:
  - 채널별, 그룹별, 비교, 높은/낮은, 순위
  - 매출, 전환율, 성과, 지표 확인
  - 이상치, 분포, 상관관계, 통계
  - 변수 중요도, 피처, 예측
  - 인사이트, 분석, 확인, 알려줘
  - 데이터를 보거나 분석하는 모든 요청

AG-03 — 오직 이 경우만:
  - "DB에서 조회해줘", "SQL로 뽑아줘" 처럼 명시적 DB/SQL 언급

AG-02 — 전처리, FE 파이프라인 명시적 요청
AG-05 — 보고서, PDF, CSV 저장 명시적 요청
FULL_PIPELINE — "전체 분석", "처음부터 끝까지"
NONE — 완전한 잡담, 인사

중요: "매출 높은 채널", "채널별 성과", "어떤 채널이 좋아" 등은
모두 AG-04다. AG-03이 아니다.
애매하면 무조건 AG-04.

JSON 형식으로만 반환. 마크다운 금지.
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
    HITL 수정 답변이 있으면 user_input에 반영
    """
    session_id = _init_session(state)
    user_input = state.get("user_input", "")
    data_meta  = state.get("data_meta", {})

    # HITL 수정 답변 반영
    hitl_history = state.get("hitl_history", [])
    last_hitl    = hitl_history[-1] if hitl_history else {}
    if last_hitl.get("response") == "수정" and last_hitl.get("user_answer"):
        user_input = last_hitl["user_answer"]
        log_tool_call(
            session_id=session_id,
            tool_name="AG-01_apply_hitl_modification",
            params={"original": state.get("user_input", "")},
            result={"modified_user_input": user_input},
        )

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
        "user_input":     user_input,
        "next_agent":     next_agent,
        "current_agent":  "AG-01",
        "execution_plan": execution_plan,
        "hitl_required":  next_agent == "hitl_plan",
        "agent_results":  state.get("agent_results", {}),
        "hitl_history":   hitl_history,
    }


def orchestrator_respond_node(state: GraphState) -> dict[str, Any]:
    """Sub-Agent 결과를 받아 사용자에게 자연어 응답 생성"""
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    agent_results = state.get("agent_results", {})
    user_profile  = state.get("user_profile", {})

    t0       = time.time()
    response = _generate_response(user_input, agent_results, user_profile)
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
    """LLM으로 의도 파악 — JSON 파싱 실패 시 AG-04 fallback"""
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
        valid  = {"AG-02", "AG-03", "AG-04", "AG-05", "FULL_PIPELINE", "NONE"}
        if result.get("intent") not in valid:
            result["intent"] = "AG-04"
        return result
    except json.JSONDecodeError:
        return {
            "intent":     "AG-04",
            "sub_intent": user_input,
            "params":     {"task": "eda", "question": user_input},
        }


def _map_intent_to_agent(intent: dict) -> str:
    return {
        "AG-02":         "AG-02",
        "AG-03":         "AG-03",
        "AG-04":         "AG-04",
        "AG-05":         "AG-05",
        "FULL_PIPELINE": "hitl_plan",
        "NONE":          "respond",
    }.get(intent.get("intent", "AG-04"), "AG-04")


def _build_plan(intent: dict, state: GraphState) -> dict:
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


# ── 페르소나 감지 키워드 ────────────────────────────────────────────
# 사용자 표현 스타일로 P-01(마케터) / P-02(분석가) 자동 판단

_ANALYST_KEYWORDS = {
    "상관계수", "분산", "표준편차", "회귀", "p-value", "통계",
    "feature importance", "borda", "vif", "lasso", "random forest",
    "정규분포", "이분산", "다중공선성", "교차검증", "hyperparameter",
    "coefficient", "r2", "rmse", "mae", "모델", "알고리즘",
}

_MARKETER_KEYWORDS = {
    "매출", "전환율", "cvr", "roas", "광고비", "채널", "캠페인",
    "클릭", "노출", "구매", "고객", "ctr", "cpc", "kpi",
    "성과", "효율", "예산", "타겟", "리텐션", "이탈",
}


def _detect_persona(user_input: str) -> str:
    """
    사용자 입력 스타일로 페르소나 판단
    P-01: 마케터 (비즈니스 용어 중심)
    P-02: 분석가 (기술 용어 중심)
    """
    text  = user_input.lower()
    score = sum(1 for kw in _ANALYST_KEYWORDS if kw in text)           - sum(1 for kw in _MARKETER_KEYWORDS if kw in text)
    return "P-02" if score > 0 else "P-01"


PERSONA_SYSTEM = {
    "퍼포먼스 마케터": (
        "너는 퍼포먼스 마케터를 돕는 이커머스 분석 에이전트다.\n"
        "규칙:\n"
        "- 통계·기술 용어 사용 금지. 마케터가 모르는 단어 쓰지 마라\n"
        "- 비즈니스 임팩트 중심 (매출, 전환율, ROAS, 광고 효율 관점)\n"
        "- 숫자는 쉽게 해석 (0.977 → '매출과 거의 직결')\n"
        "- 결론 먼저, 근거는 짧게, 액션 아이템 포함\n"
        "- 2~4문장으로 핵심만. 한국어. 마크다운 최소화"
    ),
    "데이터 분석가 / 데이터 사이언티스트": (
        "너는 데이터 분석가를 돕는 이커머스 분석 에이전트다.\n"
        "규칙:\n"
        "- 수치와 통계 근거 포함\n"
        "- 분석 방법론 명시 (Borda Count, Pearson, IQR 등)\n"
        "- 변수명, 지표명 정확히 표기\n"
        "- 해석과 한계점 함께 제시\n"
        "- 한국어. 마크다운 최소화"
    ),
    "기획자 / 전략": (
        "너는 전략 기획자를 돕는 이커머스 분석 에이전트다.\n"
        "규칙:\n"
        "- 전략적 시사점과 액션 아이템 중심\n"
        "- 핵심 수치만 선별해서 간결하게\n"
        "- 임원 보고 수준의 언어 사용\n"
        "- 한국어. 마크다운 최소화"
    ),
    "기타": (
        "너는 이커머스 데이터 분석 에이전트다.\n"
        "분석 결과를 친절하고 명확하게 설명해라.\n"
        "한국어. 마크다운 최소화."
    ),
}

PURPOSE_CONTEXT = {
    "광고 성과 확인 (ROAS, CVR)": "사용자는 광고 채널 성과와 효율을 파악하고 싶어한다.",
    "매출 원인 파악":              "사용자는 매출에 영향을 주는 핵심 요인을 찾고 싶어한다.",
    "데이터 탐색 / EDA":           "사용자는 데이터 전체 현황을 파악하고 싶어한다.",
    "보고서 작성":                 "사용자는 분석 결과를 보고서 형태로 정리하고 싶어한다.",
}


def _get_persona_system(user_profile: dict) -> str:
    """user_profile 기반 페르소나 시스템 프롬프트 반환"""
    role    = user_profile.get("role", "기타")
    purpose = user_profile.get("purpose", "")
    base    = PERSONA_SYSTEM.get(role, PERSONA_SYSTEM["기타"])
    extra   = PURPOSE_CONTEXT.get(purpose, "")
    return f"{base}\n{extra}" if extra else base


def _generate_response(
    user_input: str,
    agent_results: dict,
    user_profile: dict | None = None,
) -> str:
    """
    user_profile 기반 페르소나 맞춤 응답 생성
    """
    llm    = _get_llm()
    system = _get_persona_system(user_profile or {})

    summary_parts = []
    for agent_id, result in agent_results.items():
        if result is None or "error" in result:
            continue

        if agent_id == "AG-04":
            eda = result.get("eda_result", {})
            if eda:
                # 질문 맞춤 분석 결과 (채널별, 이상치, 상관관계 등)
                analysis = eda.get("analysis", {})
                if analysis and analysis.get("result"):
                    summary_parts.append(
                        f"맞춤 분석 ({analysis.get('type', '')}): "
                        f"{str(analysis.get('result', {}))[:800]}"
                    )
                # 기본 통계
                s = eda.get("summary", {})
                if s:
                    summary_parts.append(
                        f"기본통계: shape={s.get('shape')}, "
                        f"타겟상관top5={s.get('target_corr_top5')}, "
                        f"이상치={s.get('outlier_ratio')}"
                    )

            fi = result.get("feature_importance", {})
            if fi and fi.get("final_ranking"):
                summary_parts.append(
                    f"변수중요도: {fi.get('final_ranking')}, "
                    f"설명={fi.get('explanation', '')[:300]}"
                )

            if result.get("insights"):
                summary_parts.append(f"인사이트: {result['insights']}")
            if result.get("actions"):
                summary_parts.append(f"액션: {result['actions']}")
            if result.get("image_paths"):
                summary_parts.append(f"차트저장: {result['image_paths']}")

        elif agent_id == "AG-05":
            summary_parts.append(f"보고서: {result.get('report_path', '')}")

        elif agent_id == "AG-03":
            if result.get("kpi_result"):
                summary_parts.append(f"KPI: {result['kpi_result']}")

    # 결과가 비어도 LLM이 데이터 메타 기반으로 추론
    if not summary_parts:
        context = (
            "아직 분석 결과가 없습니다.\n"
            "하지만 사용자 질문에 대해 가능한 범위 내에서 답변하거나, "
            "어떻게 분석하면 좋을지 안내해주세요.\n"
            "데이터가 없다는 말은 하지 마세요. "
            "무엇을 해드릴 수 있는지 적극적으로 제안하세요."
        )
    else:
        context = "\n".join(summary_parts)

    msg = f"사용자 요청: {user_input}\n\n분석 결과:\n{context}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    return response.content.strip()