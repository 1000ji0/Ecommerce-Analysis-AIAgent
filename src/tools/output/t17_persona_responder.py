"""
T-17 Persona Response Generator (production-ready v2)

사용자 질문과 분석 결과를 바탕으로
사용자 유형(마케터 / 데이터 분석가)을 판별하고,
그에 맞는 톤과 구조로 최종 응답을 생성한다.

특징:
- 룰 기반 1차 persona 분류 + LLM fallback
- 분석 결과 핵심 필드만 정리해서 프롬프트에 전달
- 예외 처리 강화
- 로깅 실패가 본 처리에 영향 주지 않음
- 반환값 구조화
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.output.t20_trace_logger import log_tool_call

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None


# ============================================================
# Persona definitions
# ============================================================

PERSONAS: dict[str, dict[str, str]] = {
    "P-01": {
        "name": "마케터",
        "style": (
            "마케터 관점에서 비즈니스 임팩트 중심으로 설명해라. "
            "전문 통계 용어는 꼭 필요할 때만 최소한으로 사용하고, "
            "매출·고객·전환·캠페인·ROI 관점의 언어를 우선 사용해라. "
            "핵심 결과와 액션 아이템을 먼저 제시해라."
        ),
        "format": (
            "응답 형식:\n"
            "1. 한 줄 결론\n"
            "2. 핵심 성과/문제\n"
            "3. 비즈니스 해석\n"
            "4. 바로 실행할 액션 2~3개\n"
        ),
    },
    "P-02": {
        "name": "데이터 분석가",
        "style": (
            "데이터 분석가 관점에서 기술적으로 상세하게 설명해라. "
            "통계 용어, 모델 성능 지표, 피처 중요도, 해석 근거를 포함해라. "
            "결론뿐 아니라 방법론과 한계도 함께 제시해라."
        ),
        "format": (
            "응답 형식:\n"
            "1. 요약\n"
            "2. 주요 결과\n"
            "3. 기술적 해석\n"
            "4. 근거 및 주의사항\n"
        ),
    },
}

DEFAULT_PERSONA = "P-02"


# ============================================================
# Rule-based persona detection
# ============================================================

MARKETER_KEYWORDS = {
    "매출", "고객", "캠페인", "광고", "roi", "roas", "전환", "전환율",
    "성과", "비즈니스", "리텐션", "이탈", "세그먼트", "퍼널", "마케팅",
    "클릭", "구매", "재구매", "고객군", "운영", "액션", "실무", "매출액",
}

ANALYST_KEYWORDS = {
    "데이터", "모델", "변수", "피처", "feature", "importance",
    "통계", "상관", "상관계수", "회귀", "분포", "가설", "검정",
    "유의성", "p-value", "pvalue", "rmse", "mae", "mse", "r2",
    "성능", "클러스터", "샘플", "이상치", "전처리", "스케일링", "검증",
    "fold", "k-fold", "kfold", "시계열", "예측", "추정", "모형",
}


# ============================================================
# LLM
# ============================================================

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _safe_llm_invoke(messages: list[Any]) -> str:
    """
    LLM 호출 후 문자열 응답으로 정규화.
    실패 시 예외를 상위로 전달.
    """
    llm = _get_llm()
    response = llm.invoke(messages)

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ============================================================
# Persona detection
# ============================================================

def detect_persona(user_input: str) -> str:
    """
    사용자 입력 기반 페르소나 자동 판단

    Returns:
        "P-01" (마케터) | "P-02" (분석가)
    """
    if not isinstance(user_input, str):
        logger.warning("[T-17] user_input is not str. fallback to default persona")
        return DEFAULT_PERSONA

    text = user_input.strip()
    if not text:
        return DEFAULT_PERSONA

    # 1차: 룰 기반
    rule_based = _detect_persona_by_rules(text)
    if rule_based is not None:
        return rule_based

    # 2차: LLM fallback
    try:
        system = (
            "사용자 입력을 보고 페르소나를 판단해라.\n"
            "P-01: 마케터 — 매출, 캠페인, 고객, 전환, ROI, 운영 성과, 액션 중심 질문\n"
            "P-02: 데이터 분석가 — 데이터, 모델, 변수, 통계, 검증, 피처, 방법론 중심 질문\n"
            "반드시 P-01 또는 P-02 중 하나만 반환하라. 설명 금지."
        )
        persona = _safe_llm_invoke([
            SystemMessage(content=system),
            HumanMessage(content=text),
        ])
        return persona if persona in PERSONAS else DEFAULT_PERSONA

    except Exception as exc:
        logger.exception("[T-17] persona detection failed: %s", exc)
        return DEFAULT_PERSONA


def _detect_persona_by_rules(user_input: str) -> str | None:
    """
    키워드 기반 1차 페르소나 판별.
    점수가 비슷하면 None 반환해서 LLM fallback 사용.
    """
    text = user_input.lower()

    marketer_score = _count_keyword_hits(text, MARKETER_KEYWORDS)
    analyst_score = _count_keyword_hits(text, ANALYST_KEYWORDS)

    # 명시적 표현 우선
    if re.search(r"(마케터|마케팅팀|캠페인 담당|실무자용|경영진용)", text):
        marketer_score += 2

    if re.search(r"(분석가|데이터팀|통계적으로|기술적으로|모델 관점)", text):
        analyst_score += 2

    if marketer_score == 0 and analyst_score == 0:
        return None

    if marketer_score >= analyst_score + 2:
        return "P-01"

    if analyst_score >= marketer_score + 2:
        return "P-02"

    return None


def _count_keyword_hits(text: str, keywords: set[str]) -> int:
    count = 0
    for kw in keywords:
        if kw.lower() in text:
            count += 1
    return count


# ============================================================
# Response generation
# ============================================================

def generate_response(
    session_id: str,
    user_input: str,
    analysis_result: dict[str, Any],
    persona: str | None = None,
) -> dict[str, str]:
    """
    페르소나에 맞는 최종 응답 생성

    Args:
        session_id:      세션 ID
        user_input:      원래 사용자 질문
        analysis_result: 분석 결과 통합 dict
        persona:         "P-01" | "P-02" | None

    Returns:
        {
            "persona": str,
            "response": str,
        }
    """
    if persona is None:
        persona = detect_persona(user_input)

    if persona not in PERSONAS:
        logger.warning("[T-17] invalid persona '%s'. fallback to default", persona)
        persona = DEFAULT_PERSONA

    persona_info = PERSONAS[persona]

    # analysis_result 정리
    prepared_analysis = _prepare_analysis_context(analysis_result)

    try:
        system = (
            f"너는 {persona_info['name']}에게 분석 결과를 설명하는 AI다.\n"
            f"{persona_info['style']}\n"
            f"{persona_info['format']}\n"
            "한국어로 응답해라.\n"
            "불필요한 군더더기 없이 명확하게 작성해라.\n"
            "분석 결과에 없는 내용을 단정적으로 지어내지 마라."
        )

        user_msg = (
            f"사용자 질문:\n{user_input}\n\n"
            f"정리된 분석 결과:\n{json.dumps(prepared_analysis, ensure_ascii=False, indent=2)}"
        )

        response_text = _safe_llm_invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])

        if not response_text:
            response_text = _build_fallback_response(persona, prepared_analysis)

        result = {
            "persona": f"{persona} ({persona_info['name']})",
            "response": response_text,
        }

    except Exception as exc:
        logger.exception("[T-17] response generation failed: %s", exc)

        result = {
            "persona": f"{persona} ({persona_info['name']})",
            "response": _build_fallback_response(persona, prepared_analysis),
        }

    _safe_log_tool_call(
        session_id=session_id,
        tool_name="persona_responder",
        tool_input={
            "persona": persona,
            "user_input": user_input,
        },
        tool_output=result,
    )

    return result


# ============================================================
# Analysis context preparation
# ============================================================

def _prepare_analysis_context(analysis_result: dict[str, Any]) -> dict[str, Any]:
    """
    응답 생성용으로 analysis_result를 정리한다.
    너무 큰 raw dict를 그대로 넣지 않도록 핵심 필드만 추린다.
    """
    if not isinstance(analysis_result, dict):
        return {"summary": str(analysis_result)}

    priority_keys = [
        "summary",
        "insight",
        "insights",
        "kpi",
        "metrics",
        "feature_importance",
        "top_features",
        "important_features",
        "recommendation",
        "recommendations",
        "action_items",
        "risk",
        "risks",
        "model_performance",
        "evaluation",
        "statistical_test",
        "segment",
        "segments",
        "outlier",
        "outliers",
        "trend",
        "trends",
    ]

    prepared: dict[str, Any] = {}

    for key in priority_keys:
        if key in analysis_result:
            prepared[key] = _truncate_value(analysis_result[key])

    # 우선 키가 하나도 없으면 상위 일부만 제한적으로 사용
    if not prepared:
        for idx, (key, value) in enumerate(analysis_result.items()):
            if idx >= 10:
                break
            prepared[key] = _truncate_value(value)

    return prepared


def _truncate_value(value: Any, max_str_len: int = 1200, max_list_items: int = 10) -> Any:
    """
    프롬프트 과다 주입을 막기 위한 길이 제한.
    """
    if isinstance(value, str):
        return value[:max_str_len]

    if isinstance(value, list):
        return [_truncate_value(v, max_str_len=max_str_len, max_list_items=max_list_items)
                for v in value[:max_list_items]]

    if isinstance(value, dict):
        truncated: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 20:
                break
            truncated[k] = _truncate_value(v, max_str_len=max_str_len, max_list_items=max_list_items)
        return truncated

    return value


# ============================================================
# Fallback response
# ============================================================

def _build_fallback_response(persona: str, prepared_analysis: dict[str, Any]) -> str:
    """
    LLM 실패 시 최소한의 안전한 응답 생성.
    """
    if persona == "P-01":
        summary = _extract_summary_text(prepared_analysis)
        return (
            "한 줄 결론\n"
            f"- {summary}\n\n"
            "핵심 성과/문제\n"
            "- 주요 분석 결과를 바탕으로 성과와 문제 지점을 우선 확인할 필요가 있습니다.\n\n"
            "비즈니스 해석\n"
            "- 고객, 전환, 운영 성과 관점에서 의미 있는 변화를 점검해야 합니다.\n\n"
            "바로 실행할 액션\n"
            "- 핵심 KPI 변화를 우선 확인하세요.\n"
            "- 영향이 큰 요인을 기준으로 우선순위를 정리하세요.\n"
            "- 추가 분석이 필요하면 세그먼트별로 재확인하세요."
        )

    summary = _extract_summary_text(prepared_analysis)
    return (
        "요약\n"
        f"- {summary}\n\n"
        "주요 결과\n"
        "- 현재 제공된 분석 결과를 바탕으로 핵심 지표와 주요 패턴을 확인할 수 있습니다.\n\n"
        "기술적 해석\n"
        "- 세부 모델 성능, 피처 중요도, 통계적 근거는 추가 확인이 필요할 수 있습니다.\n\n"
        "근거 및 주의사항\n"
        "- 자동 응답 생성 과정에서 일부 세부 해석은 축약되었을 수 있으므로 원본 결과와 함께 검토하는 것이 좋습니다."
    )


def _extract_summary_text(prepared_analysis: dict[str, Any]) -> str:
    for key in ("summary", "insight", "insights"):
        value = prepared_analysis.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "제공된 분석 결과를 기반으로 핵심 내용을 요약했습니다."


# ============================================================
# Safe logging
# ============================================================

def _safe_log_tool_call(
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: dict[str, Any],
) -> None:
    """
    로깅 실패가 본 로직을 깨지 않도록 보호.
    """
    try:
        log_tool_call(session_id, tool_name, tool_input, tool_output)
    except Exception as exc:
        logger.exception("[T-17] trace logging failed: %s", exc)