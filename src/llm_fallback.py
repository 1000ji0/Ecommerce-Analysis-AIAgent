"""
llm_fallback.py
Tool 실패 시 LLM이 직접 데이터를 보고 분석하는 fallback 함수 모음
"""
from __future__ import annotations

import json
import re
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_factory import get_llm


def _get_llm():
    return get_llm()


def _df_to_context(df: pd.DataFrame, target_col: str, max_rows: int = 5) -> str:
    numeric_df = df.select_dtypes(include="number")
    context = {
        "shape":    {"rows": len(df), "cols": len(df.columns)},
        "columns":  list(df.columns),
        "dtypes":   {c: str(t) for c, t in df.dtypes.items()},
        "sample":   df.head(max_rows).to_dict(orient="records"),
        "describe": numeric_df.describe().round(3).to_dict(),
    }
    if target_col in numeric_df.columns:
        corr = numeric_df.corrwith(numeric_df[target_col]).abs()
        context["target_corr"] = (
            corr.drop(target_col, errors="ignore")
                .sort_values(ascending=False)
                .head(10).round(4).to_dict()
        )
    return json.dumps(context, ensure_ascii=False, default=str)


def llm_feature_importance(df: pd.DataFrame, target_col: str, top_n: int = 5) -> dict:
    llm     = _get_llm()
    context = _df_to_context(df, target_col)
    system  = (
        "너는 데이터 분석 전문가다.\n"
        f"TARGET 변수에 영향을 주는 상위 {top_n}개 변수를 분석해라.\n"
        "JSON으로만 반환:\n"
        '{"final_ranking": {"변수명": 점수}, "task": "llm_fallback", "explanation": "설명"}\n'
        "마크다운 금지."
    )
    response    = llm.invoke([SystemMessage(content=system),
                               HumanMessage(content=f"데이터:\n{context}")])
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw         = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        numeric_df = df.select_dtypes(include="number")
        if target_col in numeric_df.columns:
            corr    = numeric_df.corrwith(numeric_df[target_col]).abs()
            ranking = (corr.drop(target_col, errors="ignore")
                           .sort_values(ascending=False)
                           .head(top_n).round(4).to_dict())
            return {
                "final_ranking": ranking,
                "task":          "correlation_fallback",
                "explanation":   f"상위 {top_n}개: {list(ranking.keys())}",
            }
        return {}


def llm_eda_analysis(df: pd.DataFrame, question: str, target_col: str) -> dict:
    llm     = _get_llm()
    context = _df_to_context(df, target_col)
    system  = (
        "너는 데이터 분석 전문가다.\n"
        "JSON으로만 반환:\n"
        '{"summary": {"shape":{},"missing":{},"outlier_ratio":{},"target_corr_top5":{}}, '
        '"analysis": {"type":"llm_fallback","result":{}}, "chart_type":"bar", "chart_code":""}\n'
        "마크다운 금지."
    )
    response    = llm.invoke([SystemMessage(content=system),
                               HumanMessage(content=f"질문: {question}\n\n데이터:\n{context}")])
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw         = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"summary": {}, "analysis": {"type": "llm_fallback", "result": {}},
                "chart_type": "bar", "chart_code": ""}


def llm_insight(
    df: pd.DataFrame,
    question: str,
    target_col: str,
    eda_result: dict | None = None,
    fi_result:  dict | None = None,
) -> dict:
    llm     = _get_llm()
    context = _df_to_context(df, target_col)
    extra   = ""
    if eda_result:
        extra += f"\nEDA: {json.dumps(eda_result.get('analysis', {}), ensure_ascii=False, default=str)[:500]}"
    if fi_result:
        extra += f"\n변수 중요도: {json.dumps(fi_result.get('final_ranking', {}), ensure_ascii=False)}"

    system = (
        "너는 이커머스 데이터 분석 전문가다.\n"
        "JSON으로만 반환:\n"
        '{"summary":"한 문장","insights":[],"actions":[],"viz_suggestions":[]}\n'
        "마크다운 금지."
    )
    response    = llm.invoke([SystemMessage(content=system),
                               HumanMessage(content=f"질문: {question}\n\n데이터:\n{context}{extra}")])
    content_str = response.content if isinstance(response.content, str) else str(response.content)
    raw         = re.sub(r"```(?:json)?|```", "", content_str).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"summary": "", "insights": [], "actions": [], "viz_suggestions": []}