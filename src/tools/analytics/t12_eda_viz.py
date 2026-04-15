"""
T-12 EDA & Viz Tool
기본 통계·분포·결측값·상관관계 요약 및 차트 코드 생성
"""
import json
import re

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.output.t20_trace_logger import log_tool_call

_llm = None  # LLM 인스턴스 캐싱용 전역 변수 (lazy initialization)


### 내부 함수: LLM 인스턴스 생성 (lazy initialization) ###
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


###### main 함수: EDA 수행 및 시각화 코드 생성 ######
def run_eda(
    session_id: str,
    df: pd.DataFrame,
    question: str = "이 데이터를 분석해줘",
    target_col: str = "TARGET",
) -> dict:
    """
    EDA 수행 및 시각화 코드 생성

    Returns:
        {
            "summary": dict,    # 기본 통계 요약 (shape, dtypes, missing, outlier_ratio, describe, target_corr_top5)
            "chart_type": str,  # 추천 차트 유형
            "chart_code": str,  # Plotly 차트 코드 (실행 가능)
        }
    """
    summary    = _make_summary(df, target_col)
    chart_type = _recommend_chart(df, question)
    chart_code = _generate_chart_code(df, question, chart_type, target_col)

    result = {
        "summary": summary,
        "chart_type": chart_type,
        "chart_code": chart_code,
    }
    log_tool_call(session_id, "eda_viz", {"question": question, "target_col": target_col}, result)
    return result


### 내부 함수: 기본 통계 요약 생성 ###
def _make_summary(df: pd.DataFrame, target_col: str) -> dict:
    """
    기본 통계 요약 생성
    - shape, dtypes, missing, outlier_ratio, describe, target_corr_top5
    """
    numeric_df = df.select_dtypes(include="number")

    # 결측값 현황
    missing = {
        col: int(cnt)
        for col, cnt in df.isnull().sum().items()
        if cnt > 0
    }

    # 이상치 비율 (IQR 기준) — 이상치 있는 컬럼만 포함
    outlier_ratio = {}
    for col in numeric_df.columns:
        q1, q3 = numeric_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_mask = (
            (numeric_df[col] < q1 - 1.5 * iqr) |
            (numeric_df[col] > q3 + 1.5 * iqr)
        )
        outlier_cnt = int(outlier_mask.sum())
        if outlier_cnt > 0:
            outlier_ratio[col] = round(outlier_cnt / len(df), 4)

    summary = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing,
        "outlier_ratio": outlier_ratio,
        "describe": (
            numeric_df.describe().round(4).to_dict()
            if not numeric_df.empty else {}
        ),
    }

    # 타겟 컬럼 상관관계 상위 5개
    if target_col in numeric_df.columns:
        corr = numeric_df.corrwith(numeric_df[target_col]).abs()
        summary["target_corr_top5"] = (
            corr.drop(target_col, errors="ignore")
            .sort_values(ascending=False)
            .head(5)
            .round(4)
            .to_dict()
        )

    return summary


### 내부 함수: 차트 유형 추천 ###
def _recommend_chart(df: pd.DataFrame, question: str) -> str:
    """
    질문 키워드 + 컬럼명 기반으로 차트 유형 추천
    LLM 호출 없이 규칙 기반으로 처리 (속도 최적화)
    """
    q = question.lower()
    cols_lower = [c.lower() for c in df.columns]

    # 시계열 감지 — 질문 키워드 또는 컬럼명
    if any(k in q for k in ["시계열", "추이", "트렌드", "날짜", "월", "일별"]):
        return "line"
    if any(k in cols_lower for k in ["date", "time", "datetime", "month", "year", "일자", "날짜"]):
        return "line"

    # 상관관계
    if any(k in q for k in ["상관", "관계", "히트맵"]):
        return "heatmap"

    # 분포
    if any(k in q for k in ["분포", "히스토그램"]):
        return "histogram"

    # 비교·카테고리
    if any(k in q for k in ["비교", "카테고리", "그룹", "채널", "세그먼트"]):
        return "bar"

    # 기본: 수치형 컬럼 2개 이상이면 scatter, 아니면 bar
    numeric_count = len(df.select_dtypes(include="number").columns)
    return "scatter" if numeric_count >= 2 else "bar"


### 내부 함수: 차트 코드 생성 ###
def _generate_chart_code(
    df: pd.DataFrame,
    question: str,
    chart_type: str,
    target_col: str,
) -> str:
    """
    LLM 기반 Plotly 차트 코드 생성
    생성 실패 시 fallback 템플릿 반환
    """
    llm = _get_llm()
    cols_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

    system = (
        "Plotly를 사용한 Python 차트 코드를 생성해라.\n"
        "규칙:\n"
        "- 변수명은 df 고정\n"
        "- import 문 포함\n"
        "- fig.show() 제외\n"
        "- 코드만 반환, 마크다운 금지\n"
        "- fig 변수에 결과 저장"
    )
    msg = (
        f"질문: {question}\n"
        f"차트 유형: {chart_type}\n"
        f"타겟 컬럼: {target_col}\n"
        f"컬럼 정보: {json.dumps(cols_info, ensure_ascii=False)}"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    code = response.content.strip()
    code = re.sub(r"```(?:python)?|```", "", code).strip()

    # fallback: 코드가 비었거나 import 없으면 기본 템플릿 반환
    if not code or "import" not in code:
        return _fallback_chart_code(chart_type, target_col)

    return code


### 내부 함수: fallback 차트 템플릿 ###
def _fallback_chart_code(chart_type: str, target_col: str) -> str:
    """LLM 코드 생성 실패 시 기본 차트 코드 반환"""
    templates = {
        "bar": (
            f"import plotly.express as px\n"
            f"fig = px.bar(df, y='{target_col}', title='Bar Chart')"
        ),
        "line": (
            f"import plotly.express as px\n"
            f"fig = px.line(df, y='{target_col}', title='Line Chart')"
        ),
        "scatter": (
            "import plotly.express as px\n"
            "fig = px.scatter(df, title='Scatter Plot')"
        ),
        "histogram": (
            f"import plotly.express as px\n"
            f"fig = px.histogram(df, x='{target_col}', title='Distribution')"
        ),
        "heatmap": (
            "import plotly.express as px\n"
            "fig = px.imshow(\n"
            "    df.select_dtypes(include='number').corr().round(2),\n"
            "    title='Correlation Heatmap'\n"
            ")"
        ),
        "box": (
            f"import plotly.express as px\n"
            f"fig = px.box(df, y='{target_col}', title='Box Plot')"
        ),
    }
    return templates.get(chart_type, templates["bar"])