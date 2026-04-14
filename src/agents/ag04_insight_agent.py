"""
AG-04 분석·인사이트 에이전트
EDA → 변수 중요도 → 인사이트 도출 → 시각화 생성

역할:
- T-12: EDA & Viz Tool
- T-13: Feature Importance Tool
- T-14: Insight & Action Tool
- T-19: 시각화 생성기 (차트 코드 실행 → PNG 저장)
- T-20: SQLite + MD 로깅
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_DIR, DEFAULT_TARGET_COL
from state import GraphState
from tools.analytics.t12_eda_viz import run_eda
from tools.analytics.t13_feature_importance import analyze_importance
from tools.analytics.t14_insight_action import generate_insight
from tools.output.t19_visualizer import generate_chart
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def insight_agent_node(state: GraphState) -> dict[str, Any]:
    """
    AG-04 메인 노드

    수행 작업:
    1. AG-02 결과에서 데이터 로드
    2. T-12: EDA 수행 + 차트 코드 생성
    3. T-13: 변수 중요도 분석
    4. T-19: 차트 코드 실행 → PNG 저장
    5. T-14: 인사이트 + 액션 아이템 생성
    6. HITL ③ 결과 확인 트리거
    """
    session_id   = state.get("session_id", "")
    user_input   = state.get("user_input", "")
    plan         = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})
    ag04_params  = plan.get("params", {}).get("AG-04", {})

    target_col = ag04_params.get("target_col", DEFAULT_TARGET_COL)
    top_n      = ag04_params.get("top_n", 5)

    # ── 데이터 로드 ──────────────────────────────────────────────────
    # AG-02 결과 경로 or 직접 업로드 데이터 사용
    ag02_result = agent_results.get("AG-02", {})
    data_path   = _resolve_data_path(ag02_result, state)

    if not data_path:
        error_msg = "AG-04: 분석할 데이터 경로를 찾을 수 없습니다."
        log_tool_call(session_id, "AG-04_insight_agent", {}, None, error=error_msg)
        return {"agent_results": {**agent_results, "AG-04": {"error": error_msg}}}

    try:
        df = _load_data(data_path)
    except Exception as e:
        log_tool_call(session_id, "AG-04_data_load", {"path": data_path}, None, error=str(e))
        return {"agent_results": {**agent_results, "AG-04": {"error": str(e)}}}

    # ── Step 1: T-12 EDA ─────────────────────────────────────────────
    t0 = time.time()
    eda_result = run_eda(
        session_id=session_id,
        df=df,
        question=user_input,
        target_col=target_col,
    )
    log_tool_call(
        session_id=session_id,
        tool_name="T-12_eda_viz",
        params={"question": user_input, "target_col": target_col},
        result={
            "chart_type":  eda_result.get("chart_type"),
            "shape":       eda_result.get("summary", {}).get("shape"),
            "missing_cnt": len(eda_result.get("summary", {}).get("missing", {})),
        },
        latency_ms=int((time.time() - t0) * 1000),
    )

    # ── Step 2: T-13 Feature Importance ──────────────────────────────
    t0 = time.time()
    fi_result = {}
    try:
        fi_result = analyze_importance(
            session_id=session_id,
            df=df,
            target_col=target_col,
            top_n=top_n,
        )
        log_tool_call(
            session_id=session_id,
            tool_name="T-13_feature_importance",
            params={"target_col": target_col, "top_n": top_n},
            result={
                "task":          fi_result.get("task"),
                "valid_rows":    fi_result.get("valid_rows"),
                "final_ranking": fi_result.get("final_ranking"),
            },
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        log_tool_call(session_id, "T-13_feature_importance", {}, None, error=str(e))

    # ── Step 3: T-19 차트 생성 ───────────────────────────────────────
    image_paths = []
    chart_code  = eda_result.get("chart_code", "")
    chart_type  = eda_result.get("chart_type", "chart")

    if chart_code:
        t0 = time.time()
        viz_result = generate_chart(
            session_id=session_id,
            chart_code=chart_code,
            df=df,
            chart_type=chart_type,
        )
        log_tool_call(
            session_id=session_id,
            tool_name="T-19_visualizer",
            params={"chart_type": chart_type},
            result=viz_result,
            error=viz_result.get("error") if not viz_result.get("success") else None,
            latency_ms=int((time.time() - t0) * 1000),
        )
        if viz_result.get("success"):
            image_paths.append(viz_result["image_path"])

    # ── Step 4: T-14 인사이트 생성 ───────────────────────────────────
    # AG-03 KPI 결과가 있으면 함께 활용
    kpi_result = agent_results.get("AG-03", {})

    t0 = time.time()
    insight_result = generate_insight(
        session_id=session_id,
        question=user_input,
        eda_result=eda_result,
        kpi_result=kpi_result if kpi_result else None,
        feature_importance=fi_result if fi_result else None,
    )
    log_tool_call(
        session_id=session_id,
        tool_name="T-14_insight_action",
        params={"question": user_input},
        result={
            "insights_cnt": len(insight_result.get("insights", [])),
            "actions_cnt":  len(insight_result.get("actions", [])),
            "summary":      insight_result.get("summary", "")[:100],
        },
        latency_ms=int((time.time() - t0) * 1000),
    )

    # ── 결과 집계 ────────────────────────────────────────────────────
    result = {
        **insight_result,
        "eda_result":        eda_result,
        "feature_importance": fi_result,
        "image_paths":       image_paths,
    }

    log_tool_call(
        session_id=session_id,
        tool_name="AG-04_insight_agent_complete",
        params={"user_input": user_input},
        result={
            "insights": result.get("insights"),
            "actions":  result.get("actions"),
            "images":   image_paths,
        },
    )

    current_results = {**agent_results, "AG-04": result}

    return {
        "agent_results": current_results,
        "hitl_required": True,   # HITL ③ 결과 확인 트리거
    }


# ── 내부 헬퍼 ────────────────────────────────────────────────────────

def _resolve_data_path(ag02_result: dict, state: GraphState) -> str:
    """
    AG-02 결과 → 분석용 데이터 경로 결정
    우선순위: augmentation → multi → outlier → common → 원본 업로드
    """
    output_dir = Path(ag02_result.get("output_dir", str(OUTPUT_DIR)))

    candidates = [
        output_dir / "feature_data_augmentation.pickle",
        output_dir / "feature_data_multi.pickle",
        output_dir / "feature_data_outlier.pickle",
        output_dir / "feature_data_common.pickle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # AG-02 없으면 직접 업로드 파일 사용
    return state.get("data_meta", {}).get("path", "")


def _load_data(data_path: str) -> pd.DataFrame:
    """파일 경로에서 DataFrame 로드 (.pickle / .csv)"""
    p = Path(data_path)
    if p.suffix == ".pickle" or p.suffix == ".pkl":
        return pd.read_pickle(data_path)
    elif p.suffix == ".csv":
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {p.suffix}")