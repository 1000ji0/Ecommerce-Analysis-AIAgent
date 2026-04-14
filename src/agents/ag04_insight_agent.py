"""
AG-04 분석·인사이트 에이전트
AG-01이 결정한 task에 따라 필요한 tool만 실행

task 종류:
  eda            → T-12
  feature_importance → T-13
  viz            → T-12 + T-19
  insight        → T-14 (EDA + FI 결과 기반)
  full           → T-12 + T-13 + T-19 + T-14 (전체 분석 시)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_TARGET_COL, get_session_output_dir
from state import GraphState
from tools.analytics.t12_eda_viz import run_eda
from tools.analytics.t13_feature_importance import analyze_importance
from tools.analytics.t14_insight_action import generate_insight
from tools.output.t19_visualizer import generate_chart
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


def insight_agent_node(state: GraphState) -> dict[str, Any]:
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})

    # AG-01이 결정한 task
    ag04_params = plan.get("ag04_params", {})
    task        = ag04_params.get("task", "eda")
    target_col  = ag04_params.get("target_col", DEFAULT_TARGET_COL)
    top_n       = ag04_params.get("top_n", 5)
    is_full     = plan.get("is_full_pipeline", False)

    if is_full:
        task = "full"

    # 데이터 로드
    data_path = _resolve_data_path(agent_results, session_id, state)
    if not data_path:
        error = "분석할 데이터를 찾을 수 없습니다."
        log_tool_call(session_id, "AG-04_error", {}, None, error=error)
        return {"agent_results": {**agent_results, "AG-04": {"error": error}},
                "current_agent": "AG-04", "next_agent": "respond"}

    try:
        df = _load_data(data_path)
        log_tool_call(session_id, "AG-04_data_load",
                      {"path": data_path},
                      {"rows": len(df), "cols": len(df.columns)})
    except Exception as e:
        log_tool_call(session_id, "AG-04_data_load", {"path": data_path}, None, error=str(e))
        return {"agent_results": {**agent_results, "AG-04": {"error": str(e)}},
                "current_agent": "AG-04", "next_agent": "respond"}

    result = {}

    # ── task별 실행 ───────────────────────────────────────────────────
    if task in ("eda", "viz", "full"):
        t0 = time.time()
        eda_result = run_eda(session_id=session_id, df=df,
                             question=user_input, target_col=target_col)
        log_tool_call(session_id, "T-12_eda_viz",
                      {"task": task, "target_col": target_col},
                      {"chart_type": eda_result.get("chart_type"),
                       "shape": eda_result.get("summary", {}).get("shape")},
                      latency_ms=int((time.time() - t0) * 1000))
        result["eda_result"] = eda_result

    if task in ("feature_importance", "full"):
        t0 = time.time()
        try:
            fi_result = analyze_importance(session_id=session_id, df=df,
                                           target_col=target_col, top_n=top_n)
            log_tool_call(session_id, "T-13_feature_importance",
                          {"target_col": target_col, "top_n": top_n},
                          {"task": fi_result.get("task"),
                           "final_ranking": fi_result.get("final_ranking")},
                          latency_ms=int((time.time() - t0) * 1000))
            result["feature_importance"] = fi_result
        except Exception as e:
            log_tool_call(session_id, "T-13_feature_importance", {}, None, error=str(e))

    if task in ("viz", "full") and result.get("eda_result"):
        chart_code = result["eda_result"].get("chart_code", "")
        chart_type = result["eda_result"].get("chart_type", "chart")
        if chart_code:
            t0 = time.time()
            viz_result = generate_chart(session_id=session_id, chart_code=chart_code,
                                        df=df, chart_type=chart_type)
            log_tool_call(session_id, "T-19_visualizer",
                          {"chart_type": chart_type}, viz_result,
                          error=viz_result.get("error") if not viz_result.get("success") else None,
                          latency_ms=int((time.time() - t0) * 1000))
            result["image_paths"] = [viz_result["image_path"]] if viz_result.get("success") else []

    if task in ("insight", "full"):
        kpi_result = agent_results.get("AG-03") or None
        t0 = time.time()
        insight_result = generate_insight(
            session_id=session_id,
            question=user_input,
            eda_result=result.get("eda_result"),
            kpi_result=kpi_result,
            feature_importance=result.get("feature_importance"),
        )
        log_tool_call(session_id, "T-14_insight_action",
                      {"question": user_input},
                      {"insights_cnt": len(insight_result.get("insights", [])),
                       "summary": insight_result.get("summary", "")[:100]},
                      latency_ms=int((time.time() - t0) * 1000))
        result.update(insight_result)

    log_tool_call(session_id, "AG-04_complete", {"task": task}, result)

    # 전체 분석 파이프라인 → 다음은 AG-05
    next_agent = "AG-05" if is_full else "respond"

    current = {**agent_results, "AG-04": result}
    return {
        "agent_results": current,
        "current_agent": "AG-04",
        "next_agent":    next_agent,
    }


def _resolve_data_path(agent_results: dict, session_id: str, state: GraphState) -> str:
    session_output = get_session_output_dir(session_id)
    output_dir     = Path(agent_results.get("AG-02", {}).get("output_dir", str(session_output)))

    for name in ["feature_data_augmentation.pickle", "feature_data_multi.pickle",
                 "feature_data_outlier.pickle", "feature_data_common.pickle"]:
        p = output_dir / name
        if p.exists():
            return str(p)

    return state.get("data_meta", {}).get("path", "")


def _load_data(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix in (".pickle", ".pkl"):
        return pd.read_pickle(data_path)
    return pd.read_csv(data_path)