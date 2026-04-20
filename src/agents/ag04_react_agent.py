"""
AG-04 ReAct 분석 에이전트
LangGraph create_react_agent 기반

LLM이 스스로 tool 선택·순서 결정:
  Thought → Action → Observation → Thought → ...

사용 가능한 tool:
  eda_analysis          T-12 기반 EDA
  feature_importance    T-13 기반 변수 중요도
  generate_insight      T-14 기반 인사이트
  create_visualization  T-19 기반 시각화
"""
from __future__ import annotations

import json
import re
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

# ── 전역 컨텍스트 (tool 내부에서 사용) ──────────────────────────────
# tool 함수는 state 직접 접근 불가 → 전역으로 주입
_ctx: dict = {}


def _set_ctx(session_id: str, df: pd.DataFrame, target_col: str):
    _ctx["session_id"] = session_id
    _ctx["df"]         = df
    _ctx["target_col"] = target_col


# ── Tool 정의 ────────────────────────────────────────────────────────

@tool
def eda_analysis(question: str) -> str:
    """
    데이터 탐색 분석 (EDA) 수행.
    이상치, 결측값, 분포, 상관관계, 채널별 비교 등 데이터 현황 파악에 사용.
    질문 예시: "이상치 있는 컬럼 알려줘", "채널별 매출 비교", "상관관계 분석"
    """
    session_id = _ctx.get("session_id", "")
    df         = _ctx.get("df")
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
                       "shape": result.get("summary", {}).get("shape")},
                      latency_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        log_tool_call(session_id, "ReAct_T12_FALLBACK", {"error": str(e)}, {})
        result = llm_eda_analysis(df, question, target_col)

    # 핵심 결과만 문자열로 반환 (LLM이 읽을 수 있게)
    analysis = result.get("analysis", {})
    summary  = result.get("summary", {})
    return json.dumps({
        "analysis_type":      analysis.get("type"),
        "analysis_result":    str(analysis.get("result", {}))[:800],
        "shape":              summary.get("shape"),
        "missing":            summary.get("missing"),
        "outlier_ratio":      summary.get("outlier_ratio"),
        "target_corr_top5":   summary.get("target_corr_top5"),
    }, ensure_ascii=False)


@tool
def feature_importance(top_n: int = 5) -> str:
    """
    변수 중요도 분석. 어떤 변수가 TARGET(매출)에 가장 큰 영향을 주는지 파악.
    Borda Count 방식으로 여러 모델의 순위를 통합.
    예시: top_n=5 → 상위 5개 변수 중요도 반환
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
                      {"task": result.get("task"),
                       "final_ranking": result.get("final_ranking")},
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
    분석 결과 기반 인사이트 및 액션 아이템 생성.
    EDA나 변수 중요도 분석 후 비즈니스 인사이트 도출 시 사용.
    question: 사용자의 원래 질문 또는 분석 목적
    """
    session_id = _ctx.get("session_id", "")
    df         = _ctx.get("df")
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return "데이터가 로드되지 않았습니다."

    t0 = time.time()
    try:
        # EDA 결과를 컨텍스트로 넘겨서 더 풍부한 인사이트 생성
        eda_result = run_eda(session_id=session_id, df=df,
                             question=question, target_col=target_col)
        result = generate_insight(session_id=session_id, question=question,
                                  eda_result=eda_result,
                                  kpi_result=None, feature_importance=None)
        log_tool_call(session_id, "ReAct_T14_insight",
                      {"question": question},
                      {"insights_cnt": len(result.get("insights", [])),
                       "summary": result.get("summary", "")[:100]},
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
    데이터 시각화 차트 생성. PNG 파일로 저장.
    chart_type: "bar", "line", "scatter", "histogram", "heatmap", "box", "auto"
    question: 시각화할 내용 설명 (예: "채널별 매출 막대 차트")
    """
    session_id = _ctx.get("session_id", "")
    df         = _ctx.get("df")
    target_col = _ctx.get("target_col", DEFAULT_TARGET_COL)

    if df is None:
        return "데이터가 로드되지 않았습니다."

    t0 = time.time()
    try:
        eda_result = run_eda(session_id=session_id, df=df,
                             question=question or chart_type,
                             target_col=target_col)
        chart_code     = eda_result.get("chart_code", "")
        detected_type  = eda_result.get("chart_type", chart_type)

        if not chart_code:
            return "차트 코드 생성에 실패했습니다."

        viz_result = generate_chart(session_id=session_id, chart_code=chart_code,
                                    df=df, chart_type=detected_type)
        log_tool_call(session_id, "ReAct_T19_viz",
                      {"chart_type": detected_type},
                      viz_result,
                      error=viz_result.get("error") if not viz_result.get("success") else None,
                      latency_ms=int((time.time() - t0) * 1000))

        if viz_result.get("success"):
            return json.dumps({
                "success":    True,
                "image_path": viz_result["image_path"],
                "chart_type": detected_type,
            }, ensure_ascii=False)
        else:
            return json.dumps({"success": False, "error": viz_result.get("error", "")[:200]})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)[:200]})


# ── ReAct Agent 생성 ─────────────────────────────────────────────────

REACT_TOOLS = [eda_analysis, feature_importance, generate_insight_tool, create_visualization]

REACT_SYSTEM = """
너는 이커머스 데이터 분석 전문가 에이전트다.
사용자의 질문을 분석하고 반드시 적절한 tool을 호출해서 분석을 수행해라.

사용 가능한 tool:
- eda_analysis: 데이터 탐색, 이상치, 분포, 채널 비교, 상관관계 분석
- feature_importance: 변수 중요도 분석 (어떤 변수가 매출에 영향을 주는지)
- generate_insight_tool: 분석 결과 기반 비즈니스 인사이트 및 액션 아이템 생성
- create_visualization: 차트/그래프 생성
  → boxplot, bar, line, scatter, histogram, heatmap 등 모든 시각화 요청
  → "그려줘", "시각화", "차트", "그래프", "plot", "박스플롯" 등

중요 규칙:
- 시각화 관련 단어가 있으면 반드시 create_visualization 호출
- 분석 요청이면 반드시 tool을 1개 이상 호출
- tool 없이 직접 답변하지 마라
- 한국어로 최종 응답
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
    )


_react_agent = None


def _get_react_agent():
    global _react_agent
    if _react_agent is None:
        _react_agent = _build_react_agent()
    return _react_agent


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

def insight_agent_node(state: GraphState) -> dict[str, Any]:
    """
    AG-04 ReAct 노드
    LLM이 tool을 스스로 선택하고 순서를 결정
    """
    session_id    = state.get("session_id", "")
    user_input    = state.get("user_input", "")
    plan          = state.get("execution_plan", {})
    agent_results = state.get("agent_results", {})
    is_full       = plan.get("is_full_pipeline", False)

    target_col = plan.get("ag04_params", {}).get("target_col", DEFAULT_TARGET_COL)

    # 데이터 로드
    data_path = _resolve_data_path(agent_results, session_id, state)
    if not data_path:
        error = "분석할 데이터를 찾을 수 없습니다."
        log_tool_call(session_id, "AG-04_error", {}, None, error=error)
        return {
            "agent_results": {**agent_results, "AG-04": {"error": error}},
            "current_agent": "AG-04",
            "next_agent":    "respond",
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
            "current_agent": "AG-04",
            "next_agent":    "respond",
        }

    # 전역 컨텍스트 주입
    _set_ctx(session_id=session_id, df=df, target_col=target_col)

    # 전체 파이프라인이면 더 구체적인 지시 추가
    question = user_input
    if is_full:
        question = (
            f"{user_input}\n"
            "전체 분석을 수행해줘: EDA → 변수 중요도 → 인사이트 → 시각화 순서로."
        )

    # ReAct 에이전트 실행
    log_tool_call(session_id, "AG-04_react_start",
                  {"question": question, "is_full": is_full}, {})
    t0 = time.time()

    try:
        react_agent = _get_react_agent()
        response    = react_agent.invoke({"messages": [("user", question)]})

        # 최종 응답 추출
        final_msg = response["messages"][-1]
        final_answer = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

        # tool 실행 결과 수집
        tool_results = _extract_tool_results(response["messages"])

        log_tool_call(session_id, "AG-04_react_complete",
                      {"question": question},
                      {"answer_length": len(final_answer),
                       "tools_used": list(tool_results.keys())},
                      latency_ms=int((time.time() - t0) * 1000))

        result = {
            "react_answer":       final_answer,
            "tools_used":         list(tool_results.keys()),
            "eda_result":         tool_results.get("eda_analysis"),
            "feature_importance": tool_results.get("feature_importance"),
            "insights":           tool_results.get("generate_insight_tool", {}).get("insights", []),
            "actions":            tool_results.get("generate_insight_tool", {}).get("actions", []),
            "summary":            tool_results.get("generate_insight_tool", {}).get("summary", ""),
            "image_paths":        _collect_visualization_paths(response["messages"]),
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


def _tool_content_to_dict(content: Any) -> dict:
    """ToolMessage.content → dict (JSON·코드펜스·비문자 처리)."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        content = str(content)
    text = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"raw": content[:500]}


def _extract_tool_results(messages: list) -> dict:
    """ReAct 메시지에서 tool 실행 결과 추출 (동일 tool 여러 호출 시 마지막 파싱 결과 유지)."""
    results: dict[str, Any] = {}
    for msg in messages:
        name = getattr(msg, "name", None) or ""
        if not name or not hasattr(msg, "content"):
            continue
        results[name] = _tool_content_to_dict(msg.content)
    return results


def _collect_visualization_paths(messages: list) -> list[str]:
    """create_visualization tool 호출마다 성공 시 image_path 수집 (중복·순서 유지)."""
    paths: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        name = getattr(msg, "name", None) or ""
        if name != "create_visualization":
            continue
        data = _tool_content_to_dict(getattr(msg, "content", ""))
        if data.get("success") and data.get("image_path"):
            p = str(data["image_path"])
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


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