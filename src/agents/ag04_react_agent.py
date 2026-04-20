"""
AG-04 ReAct 분석 에이전트 — 1단계 강화

변경사항:
1. ReAct 루프 강화 — LLM이 중간 결과 보고 다음 분석 스스로 결정
2. 파생 변수 자동 계산 — ROAS, CVR, CTR, CPC 없으면 스스로 계산
3. max_iterations 10으로 확대 — 복잡한 질문도 여러 단계 분석 가능
4. 분석 후 "더 볼 것 제안" — 다음 분석 방향 스스로 제안
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config import GEMINI_MODEL, GOOGLE_API_KEY, DEFAULT_TARGET_COL, get_session_output_dir
from state import GraphState
from tools.analytics.t12_eda_viz import run_eda
from tools.analytics.t13_feature_importance import analyze_importance
from tools.analytics.t14_insight_action import generate_insight
from tools.output.t19_visualizer import generate_chart
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore
from llm_fallback import llm_feature_importance, llm_eda_analysis, llm_insight

_store = TraceStore()
_ctx: dict = {}


def _set_ctx(session_id: str, df: pd.DataFrame, target_col: str):
    _ctx["session_id"] = session_id
    _ctx["df"]         = df
    _ctx["target_col"] = target_col


# ── 파생 변수 자동 계산 ──────────────────────────────────────────────

def _enrich_df(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    질문에 필요한 파생 변수가 없으면 자동으로 계산해서 추가
    원본 df는 건드리지 않고 copy해서 반환
    """
    df    = df.copy()
    cols  = [c.lower() for c in df.columns]
    q     = question.lower()

    # ROAS = 매출 / 광고비
    if "roas" in q and "roas" not in cols:
        target = _ctx.get("target_col", DEFAULT_TARGET_COL)
        if target in df.columns and "ad_spend" in df.columns:
            df["ROAS"] = (df[target] / df["ad_spend"].replace(0, float("nan"))).round(4)
            log_tool_call(_ctx.get("session_id", ""), "derived_ROAS",
                          {}, {"added": True})

    # CVR = 구매수 / 세션수
    if any(k in q for k in ("cvr", "전환율", "conversion")):
        if "cvr" not in cols and "purchases" in df.columns and "sessions" in df.columns:
            df["CVR"] = (df["purchases"] / df["sessions"].replace(0, float("nan"))).round(4)
            log_tool_call(_ctx.get("session_id", ""), "derived_CVR",
                          {}, {"added": True})

    # CTR = 클릭수 / 노출수
    if any(k in q for k in ("ctr", "클릭률", "click-through")):
        if "ctr" not in cols and "clicks" in df.columns and "impressions" in df.columns:
            df["CTR"] = (df["clicks"] / df["impressions"].replace(0, float("nan"))).round(4)
            log_tool_call(_ctx.get("session_id", ""), "derived_CTR",
                          {}, {"added": True})

    # CPC = 광고비 / 클릭수
    if any(k in q for k in ("cpc", "클릭당비용", "cost per click")):
        if "cpc" not in cols and "ad_spend" in df.columns and "clicks" in df.columns:
            df["CPC"] = (df["ad_spend"] / df["clicks"].replace(0, float("nan"))).round(0)
            log_tool_call(_ctx.get("session_id", ""), "derived_CPC",
                          {}, {"added": True})

    # AOV = 매출 / 구매수
    if any(k in q for k in ("aov", "평균주문금액", "average order")):
        if "aov" not in cols and "purchases" in df.columns:
            target = _ctx.get("target_col", DEFAULT_TARGET_COL)
            if target in df.columns:
                df["AOV"] = (df[target] / df["purchases"].replace(0, float("nan"))).round(0)
                log_tool_call(_ctx.get("session_id", ""), "derived_AOV",
                              {}, {"added": True})

    return df


# ── Tool 정의 ────────────────────────────────────────────────────────

@tool
def eda_analysis(question: str) -> str:
    """
    데이터 탐색 분석 (EDA).
    이상치, 결측값, 분포, 상관관계, 채널별 비교 등.
    ROAS/CVR/CTR 같은 파생 변수가 필요하면 자동으로 계산해서 분석.
    """
    session_id = _ctx.get("session_id", "")
    df         = _enrich_df(_ctx.get("df"), question) if _ctx.get("df") is not None else None
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return "데이터가 로드되지 않았습니다."

    t0 = time.time()
    try:
        result = run_eda(session_id=session_id, df=df,
                         question=question, target_col=target_col)
        log_tool_call(session_id, "ReAct_T12_eda",
                      {"question": question},
                      {"analysis_type": result.get("analysis", {}).get("type"),
                       "new_cols": [c for c in df.columns if c not in _ctx["df"].columns]},
                      latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        log_tool_call(session_id, "ReAct_T12_FALLBACK", {"error": str(e)}, {})
        result = llm_eda_analysis(df, question, target_col)

    analysis = result.get("analysis", {})
    summary  = result.get("summary", {})
    derived  = [c for c in df.columns if c not in _ctx["df"].columns]

    return json.dumps({
        "analysis_type":   analysis.get("type"),
        "analysis_result": str(analysis.get("result", {}))[:1000],
        "shape":           summary.get("shape"),
        "missing":         summary.get("missing"),
        "outlier_ratio":   summary.get("outlier_ratio"),
        "target_corr_top5": summary.get("target_corr_top5"),
        "derived_columns_added": derived,
    }, ensure_ascii=False)


@tool
def feature_importance(top_n: int = 5) -> str:
    """
    변수 중요도 분석.
    어떤 변수가 TARGET(매출)에 가장 큰 영향을 주는지 Borda Count로 계산.
    """
    session_id = _ctx.get("session_id", "")
    df         = _ctx.get("df")
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return "데이터가 로드되지 않았습니다."

    t0 = time.time()
    try:
        result = analyze_importance(session_id=session_id, df=df,
                                    target_col=target_col, top_n=top_n)
        log_tool_call(session_id, "ReAct_T13_fi",
                      {"top_n": top_n},
                      {"final_ranking": result.get("final_ranking")},
                      latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        log_tool_call(session_id, "ReAct_T13_FALLBACK", {"error": str(e)}, {})
        result = llm_feature_importance(df, target_col, top_n)

    return json.dumps({
        "final_ranking": result.get("final_ranking", {}),
        "task":          result.get("task", ""),
        "explanation":   result.get("explanation", "")[:400],
    }, ensure_ascii=False)


@tool
def generate_insight_tool(question: str) -> str:
    """
    분석 결과 기반 비즈니스 인사이트 및 액션 아이템 생성.
    EDA나 변수 중요도 분석 결과를 종합해서 실행 가능한 인사이트 도출.
    """
    session_id = _ctx.get("session_id", "")
    df         = _enrich_df(_ctx.get("df"), question) if _ctx.get("df") is not None else None
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return "데이터가 로드되지 않았습니다."

    t0 = time.time()
    try:
        eda_result = run_eda(session_id=session_id, df=df,
                             question=question, target_col=target_col)
        result = generate_insight(session_id=session_id, question=question,
                                  eda_result=eda_result,
                                  kpi_result=None, feature_importance=None)
        log_tool_call(session_id, "ReAct_T14_insight",
                      {"question": question},
                      {"insights_cnt": len(result.get("insights", []))},
                      latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        log_tool_call(session_id, "ReAct_T14_FALLBACK", {"error": str(e)}, {})
        result = llm_insight(df=df, question=question, target_col=target_col)

    return json.dumps({
        "summary":  result.get("summary", ""),
        "insights": result.get("insights", []),
        "actions":  result.get("actions", []),
    }, ensure_ascii=False)


@tool
def create_visualization(chart_type: str = "auto", question: str = "") -> str:
    """
    데이터 시각화 차트 생성 → PNG 파일 저장.
    ROAS/CVR/CTR 같은 파생 변수도 자동으로 계산해서 시각화.
    chart_type: bar, line, scatter, histogram, heatmap, box, auto
    question: 시각화할 내용 (예: "채널별 ROAS boxplot")
    """
    session_id = _ctx.get("session_id", "")
    q          = question or chart_type
    df         = _enrich_df(_ctx.get("df"), q) if _ctx.get("df") is not None else None
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return json.dumps({"success": False, "error": "데이터 없음"})

    derived = [c for c in df.columns if c not in _ctx["df"].columns]
    if derived:
        log_tool_call(session_id, "viz_derived_cols",
                      {"question": q}, {"derived": derived})

    t0 = time.time()
    try:
        eda_result = run_eda(session_id=session_id, df=df,
                             question=q, target_col=target_col)
        chart_code    = eda_result.get("chart_code", "")
        detected_type = eda_result.get("chart_type", chart_type)

        if not chart_code:
            return json.dumps({"success": False, "error": "차트 코드 생성 실패"})

        viz_result = generate_chart(session_id=session_id, chart_code=chart_code,
                                    df=df, chart_type=detected_type)
        log_tool_call(session_id, "ReAct_T19_viz",
                      {"chart_type": detected_type, "derived": derived},
                      viz_result,
                      error=viz_result.get("error") if not viz_result.get("success") else None,
                      latency_ms=int((time.time() - t0) * 1000))

        if viz_result.get("success"):
            return json.dumps({
                "success":    True,
                "image_path": viz_result["image_path"],
                "chart_type": detected_type,
                "derived_columns_used": derived,
            }, ensure_ascii=False)
        return json.dumps({"success": False, "error": viz_result.get("error", "")[:200]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)[:200]})


# ── ReAct Agent 설정 ─────────────────────────────────────────────────

REACT_TOOLS = [eda_analysis, feature_importance, generate_insight_tool, create_visualization]

REACT_SYSTEM = """
너는 이커머스 데이터 분석 전문가 에이전트다.
사용자 질문에 대해 스스로 분석 계획을 세우고 단계적으로 실행해라.

핵심 원칙:
1. 반드시 tool을 호출해서 실제 데이터를 분석해라. 추론으로 답하지 마라.
2. 중간 결과를 보고 다음에 뭘 더 봐야 할지 스스로 판단해라.
3. ROAS, CVR, CTR 같은 파생 변수가 없어도 tool이 자동으로 계산한다.
4. 시각화 요청(boxplot, 그래프, 차트 등)은 반드시 create_visualization 호출.

tool 선택 기준:
- eda_analysis: 데이터 탐색, 이상치, 분포, 채널 비교, 상관관계
  → "boxplot", "분포", "이상치", "비교", "채널별", "ROAS", "CVR"
- feature_importance: 어떤 변수가 중요한지
  → "중요도", "영향", "피처", "변수"
- generate_insight_tool: 비즈니스 인사이트, 액션 아이템
  → "인사이트", "전략", "개선", "왜", "원인"
- create_visualization: 시각화
  → "그려줘", "시각화", "차트", "그래프", "plot", "boxplot", "히스토그램"

자율 분석 루프:
- 단순 질문: tool 1개
- "분석해줘" 같은 포괄적 질문: EDA → 중요도 → 인사이트 순서로 여러 tool 호출
- 결과를 보고 "이건 더 파봐야겠다" 싶으면 추가 tool 호출

마지막에는 반드시:
1. 분석 결과 요약
2. 추가로 분석하면 좋을 것 1~2개 제안

한국어로 답해라.
"""


def _build_react_agent():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    return create_react_agent(
        model=llm,
        tools=REACT_TOOLS,
        prompt=REACT_SYSTEM,
        max_iterations=10,   # 복잡한 분석도 여러 단계 가능
    )


_react_agent = None


def _get_react_agent():
    global _react_agent
    if _react_agent is None:
        _react_agent = _build_react_agent()
    return _react_agent


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def insight_agent_node(state: GraphState) -> dict[str, Any]:
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})
    is_full       = plan.get("is_full_pipeline", False)
    target_col    = plan.get("ag04_params", {}).get("target_col", DEFAULT_TARGET_COL)

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

    # 전역 컨텍스트 주입
    _set_ctx(session_id=session_id, df=df, target_col=target_col)

    question = user_input
    if is_full:
        question = (
            f"{user_input}\n"
            "전체 분석 수행: EDA → 변수 중요도 → 인사이트 → 시각화 순서로 단계적으로 분석해라. "
            "각 단계 결과를 보고 다음 분석 방향을 스스로 결정해라."
        )

    log_tool_call(session_id, "AG-04_react_start",
                  {"question": question, "is_full": is_full}, {})
    t0 = time.time()

    try:
        react_agent  = _get_react_agent()
        response     = react_agent.invoke({"messages": [("user", question)]})
        final_msg    = response["messages"][-1]
        final_answer = final_msg.content if isinstance(final_msg.content, str) \
                       else str(final_msg.content)
        tool_results = _extract_tool_results(response["messages"])

        log_tool_call(session_id, "AG-04_react_complete",
                      {"question": question},
                      {"tools_used":    list(tool_results.keys()),
                       "answer_length": len(final_answer)},
                      latency_ms=int((time.time() - t0) * 1000))

        result = {
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
        result = {"error": str(e), "react_answer": ""}

    next_agent = "AG-05" if is_full else "respond"
    return {
        "agent_results": {**agent_results, "AG-04": result},
        "current_agent": "AG-04",
        "next_agent":    next_agent,
    }


def _safe_get(d: dict, key: str, field: str, default):
    val = d.get(key)
    if isinstance(val, dict):
        return val.get(field, default)
    return default


def _extract_tool_results(messages: list) -> dict:
    results = {}
    for msg in messages:
        if hasattr(msg, "name") and msg.name and hasattr(msg, "content"):
            try:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                results[msg.name] = json.loads(content)
            except Exception:
                results[msg.name] = {"raw": str(msg.content)[:200]}
    return results


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