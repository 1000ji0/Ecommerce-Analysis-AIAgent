"""
AG-04 분석·인사이트 에이전트
커스텀 LLM (tool calling 미지원) → llm_fallback으로 직접 분석
Gemini 등 tool calling 지원 모델 → ReAct 에이전트
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
from llm_fallback import llm_feature_importance, llm_eda_analysis, llm_insight
from llm_factory import get_llm_info

_store = TraceStore()


def _is_tool_calling_supported() -> bool:
    """커스텀 LLM은 tool calling 미지원 → False"""
    return get_llm_info().get("provider") != "custom"


def insight_agent_node(state: GraphState) -> dict[str, Any]:
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})

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
        return {
            "agent_results": {**agent_results, "AG-04": {"error": error}},
            "current_agent": "AG-04", "next_agent": "respond",
        }

    try:
        df = _load_data(data_path)
        log_tool_call(session_id, "AG-04_data_load",
                      {"path": data_path},
                      {"rows": len(df), "cols": len(df.columns)})
    except Exception as e:
        log_tool_call(session_id, "AG-04_data_load", {"path": data_path}, None, error=str(e))
        return {
            "agent_results": {**agent_results, "AG-04": {"error": str(e)}},
            "current_agent": "AG-04", "next_agent": "respond",
        }

    # tool calling 지원 여부에 따라 분기
    if _is_tool_calling_supported():
        result = _run_react(session_id, user_input, df, target_col,
                            task, top_n, is_full, agent_results)
    else:
        result = _run_direct(session_id, user_input, df, target_col,
                             task, top_n, agent_results)

    next_agent = "AG-05" if is_full else "respond"
    return {
        "agent_results": {**agent_results, "AG-04": result},
        "current_agent": "AG-04",
        "next_agent":    next_agent,
    }


# ── tool calling 지원 모델 (Gemini 등) ──────────────────────────────

def _run_react(session_id, user_input, df, target_col,
               task, top_n, is_full, agent_results):
    """ReAct 에이전트로 분석"""
    try:
        from agents.ag04_react_agent import (
            REACT_TOOLS, REACT_SYSTEM, _set_ctx, _extract_tool_results, _safe_get
        )
        from langgraph.prebuilt import create_react_agent
        from llm_factory import get_llm

        _set_ctx(session_id=session_id, df=df, target_col=target_col)

        question = user_input
        if is_full:
            question = (f"{user_input}\n"
                        "전체 분석: EDA → 변수 중요도 → 인사이트 → 시각화 순서로.")

        agent    = create_react_agent(model=get_llm(), tools=REACT_TOOLS, prompt=REACT_SYSTEM)
        response = agent.invoke({"messages": [("user", question)]})

        final_msg    = response["messages"][-1]
        final_answer = final_msg.content if isinstance(final_msg.content, str) \
                       else str(final_msg.content)
        tool_results = _extract_tool_results(response["messages"])

        log_tool_call(session_id, "AG-04_react_complete",
                      {"question": question},
                      {"tools_used": list(tool_results.keys())})

        return {
            "react_answer":       final_answer,
            "tools_used":         list(tool_results.keys()),
            "eda_result":         tool_results.get("eda_analysis"),
            "feature_importance": tool_results.get("feature_importance"),
            "insights":           _safe_get(tool_results, "generate_insight_tool", "insights", []),
            "actions":            _safe_get(tool_results, "generate_insight_tool", "actions", []),
            "summary":            _safe_get(tool_results, "generate_insight_tool", "summary", ""),
            "image_paths":        [r["image_path"]
                                   for r in [tool_results.get("create_visualization") or {}]
                                   if r.get("success") and r.get("image_path")],
        }
    except Exception as e:
        log_tool_call(session_id, "AG-04_react_error", {}, None, error=str(e))
        # ReAct 실패 시 direct로 fallback
        return _run_direct(session_id, user_input, df, target_col,
                           task, top_n, {})


# ── tool calling 미지원 모델 (Ollama Qwen 등) ────────────────────────

def _run_direct(session_id, user_input, df, target_col,
                task, top_n, agent_results):
    """LLM 직접 호출로 분석 (tool calling 없이)"""
    result = {}

    log_tool_call(session_id, "AG-04_direct_start",
                  {"task": task}, {"mode": "llm_direct"})

    # EDA
    if task in ("eda", "viz", "insight", "full"):
        t0 = time.time()
        try:
            eda_result = run_eda(session_id=session_id, df=df,
                                 question=user_input, target_col=target_col)
            log_tool_call(session_id, "T-12_eda_viz",
                          {"task": task},
                          {"analysis_type": eda_result.get("analysis", {}).get("type")},
                          latency_ms=int((time.time() - t0) * 1000))
        except Exception as e:
            log_tool_call(session_id, "T-12_FAILED_fallback",
                          {"error": str(e)}, {"status": "llm_fallback"})
            eda_result = llm_eda_analysis(df, user_input, target_col)
        result["eda_result"] = eda_result

    # Feature Importance
    if task in ("feature_importance", "full"):
        t0 = time.time()
        try:
            fi_result = analyze_importance(session_id=session_id, df=df,
                                           target_col=target_col, top_n=top_n)
            log_tool_call(session_id, "T-13_feature_importance",
                          {"top_n": top_n},
                          {"final_ranking": fi_result.get("final_ranking")},
                          latency_ms=int((time.time() - t0) * 1000))
        except Exception as e:
            log_tool_call(session_id, "T-13_FAILED_fallback",
                          {"error": str(e)}, {"status": "llm_fallback"})
            fi_result = llm_feature_importance(df, target_col, top_n)
        result["feature_importance"] = fi_result

    # 시각화
    if task in ("viz", "full") and result.get("eda_result"):
        chart_code = result["eda_result"].get("chart_code", "")
        chart_type = result["eda_result"].get("chart_type", "bar")
        if chart_code:
            t0         = time.time()
            viz_result = generate_chart(session_id=session_id, chart_code=chart_code,
                                        df=df, chart_type=chart_type)
            log_tool_call(session_id, "T-19_visualizer",
                          {"chart_type": chart_type}, viz_result,
                          latency_ms=int((time.time() - t0) * 1000))
            if viz_result.get("success"):
                result["image_paths"] = [viz_result["image_path"]]

    # 인사이트
    if task in ("insight", "full"):
        t0 = time.time()
        try:
            insight_result = generate_insight(
                session_id=session_id, question=user_input,
                eda_result=result.get("eda_result"),
                kpi_result=agent_results.get("AG-03"),
                feature_importance=result.get("feature_importance"),
            )
            log_tool_call(session_id, "T-14_insight_action",
                          {"question": user_input},
                          {"insights_cnt": len(insight_result.get("insights", []))},
                          latency_ms=int((time.time() - t0) * 1000))
        except Exception as e:
            log_tool_call(session_id, "T-14_FAILED_fallback",
                          {"error": str(e)}, {"status": "llm_fallback"})
            insight_result = llm_insight(
                df=df, question=user_input, target_col=target_col,
                eda_result=result.get("eda_result"),
                fi_result=result.get("feature_importance"),
            )
        result.update(insight_result)

    # 결과 없으면 전체 llm_fallback
    if not result:
        insight_result = llm_insight(df=df, question=user_input, target_col=target_col)
        result.update(insight_result)

    log_tool_call(session_id, "AG-04_complete", {"task": task},
                  {"has_eda": "eda_result" in result,
                   "has_fi":  "feature_importance" in result,
                   "insights_cnt": len(result.get("insights", []))})
    return result


# ── 헬퍼 ─────────────────────────────────────────────────────────────

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