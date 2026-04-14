"""
T-11 KPI Calculator
SQL 결과 DataFrame에서 이커머스 핵심 KPI 자동 계산
세그먼트 분석 지원
"""
import pandas as pd
import numpy as np
from tools.output.t20_trace_logger import log_tool_call


# 지원 KPI 목록
SUPPORTED_KPIS = [
    "revenue",      # 총 매출
    "aov",          # 평균 주문 금액 (Average Order Value)
    "cvr",          # 전환율 (Conversion Rate)
    "cart_abandon", # 장바구니 방치율
    "cac",          # 고객 획득 비용 (Customer Acquisition Cost)
    "roas",         # 광고 수익률 (Return on Ad Spend)
    "repurchase",   # 재구매율
]


def calculate_kpi(
    session_id: str,
    df: pd.DataFrame,
    kpis: list[str] | None = None,
    segment_col: str | None = None,
) -> dict:
    """
    DataFrame에서 KPI 계산

    Args:
        session_id:  세션 ID (로깅용)
        df:          SQL 결과 DataFrame
        kpis:        계산할 KPI 목록. None이면 가능한 전부 계산
        segment_col: 세그먼트 기준 컬럼 (예: "channel", "region")

    Returns:
        {
            "kpi_result": {"revenue": 1000000, "aov": 50000, ...},
            "segment_result": {"online": {...}, "offline": {...}},  # segment_col 지정 시
        }
    """
    if kpis is None:
        kpis = SUPPORTED_KPIS

    kpi_result = _compute(df, kpis)

    segment_result = {}
    if segment_col and segment_col in df.columns:
        for seg_val, seg_df in df.groupby(segment_col):
            segment_result[str(seg_val)] = _compute(seg_df, kpis)

    result = {"kpi_result": kpi_result, "segment_result": segment_result}
    log_tool_call(session_id, "kpi_calculator", {"kpis": kpis, "segment_col": segment_col}, result)
    return result


def _compute(df: pd.DataFrame, kpis: list[str]) -> dict:
    """실제 KPI 계산 로직"""
    result = {}
    cols = set(df.columns.str.lower())

    for kpi in kpis:
        try:
            if kpi == "revenue" and "revenue" in cols:
                result["revenue"] = float(df["revenue"].sum())

            elif kpi == "aov" and {"revenue", "orders"}.issubset(cols):
                orders = df["orders"].sum()
                result["aov"] = float(df["revenue"].sum() / orders) if orders > 0 else 0.0

            elif kpi == "cvr" and {"conversions", "sessions"}.issubset(cols):
                sessions = df["sessions"].sum()
                result["cvr"] = float(df["conversions"].sum() / sessions) if sessions > 0 else 0.0

            elif kpi == "cart_abandon" and {"add_to_cart", "purchases"}.issubset(cols):
                cart = df["add_to_cart"].sum()
                purchases = df["purchases"].sum()
                result["cart_abandon"] = float(1 - purchases / cart) if cart > 0 else 0.0

            elif kpi == "roas" and {"revenue", "ad_spend"}.issubset(cols):
                ad_spend = df["ad_spend"].sum()
                result["roas"] = float(df["revenue"].sum() / ad_spend) if ad_spend > 0 else 0.0

            elif kpi == "repurchase" and {"customer_id", "order_count"}.issubset(cols):
                total = len(df["customer_id"].unique())
                repeat = (df["order_count"] > 1).sum()
                result["repurchase"] = float(repeat / total) if total > 0 else 0.0

        except Exception:
            # 컬럼 없거나 계산 불가 → 건너뜀
            continue

    return result
