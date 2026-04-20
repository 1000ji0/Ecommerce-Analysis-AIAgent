"""
llm_fallback.py
Tool 실패 시 LLM이 직접 데이터를 보고 분석하는 fallback 함수 모음

AG-04 내부에서 T-12/T-13/T-14 실패 시 호출
"""
from __future__ import annotations

import json
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GEMINI_MODEL, GOOGLE_API_KEY

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _df_to_context(df: pd.DataFrame, target_col: str, max_rows: int = 5) -> str:
    """DataFrame을 LLM이 이해할 수 있는 텍스트로 변환"""
    numeric_df = df.select_dtypes(include="number")

    context = {
        "shape":       {"rows": len(df), "cols": len(df.columns)},
        "columns":     list(df.columns),
        "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
        "sample":      df.head(max_rows).to_dict(orient="records"),
        "describe":    numeric_df.describe().round(3).to_dict(),
    }

    # 타겟 상관관계
    if target_col in numeric_df.columns:
        corr = numeric_df.corrwith(numeric_df[target_col]).abs()
        context["target_corr"] = (
            corr.drop(target_col, errors="ignore")
                .sort_values(ascending=False)
                .head(10).round(4).to_dict()
        )

    return json.dumps(context, ensure_ascii=False, default=str)


# ── fallback 함수들 ───────────────────────────────────────────────────

def llm_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 5,
) -> dict:
    """
    T-13 실패 시 LLM이 상관관계 기반으로 변수 중요도 추론
    """
    llm     = _get_llm()
    context = _df_to_context(df, target_col)

    system = (
        "너는 데이터 분석 전문가다.\n"
        "주어진 데이터 정보를 보고 TARGET 변수에 영향을 주는 "
        f"상위 {top_n}개 변수를 분석해라.\n"
        "상관계수를 기반으로 중요도를 판단하고 비즈니스 관점에서 설명해라.\n"
        "JSON으로만 반환:\n"
        "{\n"
        '  "final_ranking": {"변수명": 점수, ...},\n'
        '  "task": "llm_fallback",\n'
        '  "explanation": "설명 텍스트"\n'
        "}\n"
        "마크다운 금지."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"데이터 정보:\n{context}"),
    ])

    import re
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        # 파싱 실패 시 상관관계로 직접 계산
        numeric_df = df.select_dtypes(include="number")
        if target_col in numeric_df.columns:
            corr = numeric_df.corrwith(numeric_df[target_col]).abs()
            ranking = (
                corr.drop(target_col, errors="ignore")
                    .sort_values(ascending=False)
                    .head(top_n).round(4).to_dict()
            )
            return {
                "final_ranking": ranking,
                "task":          "correlation_fallback",
                "explanation":   f"T-13 실패로 상관계수 기반 중요도 사용. "
                                 f"상위 {top_n}개: {list(ranking.keys())}",
            }
        return {}


def llm_eda_analysis(
    df: pd.DataFrame,
    question: str,
    target_col: str,
) -> dict:
    """
    T-12 실패 시 LLM이 데이터를 보고 EDA 수행
    """
    llm     = _get_llm()
    context = _df_to_context(df, target_col)

    system = (
        "너는 데이터 분석 전문가다.\n"
        "주어진 데이터 정보를 보고 사용자 질문에 맞는 EDA를 수행해라.\n"
        "JSON으로만 반환:\n"
        "{\n"
        '  "summary": {"shape": ..., "missing": {}, "outlier_ratio": {}, "target_corr_top5": {}},\n'
        '  "analysis": {"type": "llm_fallback", "result": {...}},\n'
        '  "chart_type": "bar",\n'
        '  "chart_code": ""\n'
        "}\n"
        "마크다운 금지."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"질문: {question}\n\n데이터 정보:\n{context}"),
    ])

    import re
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "summary":    {},
            "analysis":   {"type": "llm_fallback", "result": {"error": "파싱 실패"}},
            "chart_type": "bar",
            "chart_code": "",
        }


def llm_insight(
    df: pd.DataFrame,
    question: str,
    target_col: str,
    eda_result: dict | None = None,
    fi_result: dict | None = None,
) -> dict:
    """
    T-14 실패 시 LLM이 직접 인사이트 생성
    """
    llm     = _get_llm()
    context = _df_to_context(df, target_col)

    extra = ""
    if eda_result:
        extra += f"\nEDA 결과: {json.dumps(eda_result.get('analysis', {}), ensure_ascii=False, default=str)[:500]}"
    if fi_result:
        extra += f"\n변수 중요도: {json.dumps(fi_result.get('final_ranking', {}), ensure_ascii=False)}"

    system = (
        "너는 이커머스 데이터 분석 전문가다.\n"
        "주어진 데이터와 분석 결과를 보고 핵심 인사이트와 액션 아이템을 도출해라.\n"
        "JSON으로만 반환:\n"
        "{\n"
        '  "summary": "한 문장 요약",\n'
        '  "insights": ["인사이트1", "인사이트2", ...],\n'
        '  "actions": ["액션1", "액션2", ...],\n'
        '  "viz_suggestions": ["시각화 제안1"]\n'
        "}\n"
        "마크다운 금지."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"질문: {question}\n\n데이터:\n{context}{extra}"),
    ])

    import re
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "summary":         "분석 결과를 파악했습니다.",
            "insights":        [],
            "actions":         [],
            "viz_suggestions": [],
        }