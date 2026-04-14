"""
AG-02 Feature Engineering Agent
MCP T-01~T-07 파이프라인 실행

파이프라인 순서:
  T-01 implement_fc         → 피처 생성
  T-02 delete_outlier       → 이상치 제거
  T-03 smart_correlation    → 다중공선성 제거 (T-04와 택1 또는 병행)
  T-04 mrmr_selection       → 변수 선택
  T-05 gaussian_augmentation→ 데이터 증강
  T-06 rank_matrix          → 변수 중요도 rank matrix
  T-07 select_best_model    → 최적 모델 선택

MCP 연결: langchain-mcp-adapters (SSE)
로깅: 각 MCP tool 호출마다 SQLite + MD 기록
"""
from __future__ import annotations

import time
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MCP_SERVER_URL, OUTPUT_DIR, DEFAULT_TARGET_COL
from state import GraphState
from tools.data.t21_feature_cache import get_cache, set_cache
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store = TraceStore()


# ── LangGraph 노드 함수 ──────────────────────────────────────────────

async def fe_agent_node(state: GraphState) -> dict[str, Any]:
    """
    AG-02 메인 노드 (async — MCP 클라이언트가 async)

    실행 순서:
    1. MCP 서버에서 tool 목록 로드
    2. execution_plan의 AG-02 params 적용
    3. 파이프라인 단계별 실행 + 로깅
    4. HITL ② 전처리 확인 트리거
    """
    session_id = state.get("session_id", "")
    data_meta  = state.get("data_meta", {})
    plan       = state.get("execution_plan", {})
    ag02_params = plan.get("params", {}).get("AG-02", {})

    data_path   = data_meta.get("path", "")
    target_col  = ag02_params.get("target_col", DEFAULT_TARGET_COL)
    output_dir  = str(OUTPUT_DIR)

    # ── MCP 클라이언트 연결 ─────────────────────────────────────────
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async with MultiServerMCPClient({
        "analytics": {
            "url":       MCP_SERVER_URL,
            "transport": "sse",
        }
    }) as client:
        mcp_tools = {t.name: t for t in client.get_tools()}

        # ── Stage 1: 피처 생성 (T-01) ───────────────────────────────
        exec_tools = ag02_params.get("exec_tools", ["fpca", "nds", "timeseries", "stats"])

        # 캐시 확인
        cached_path = get_cache(session_id, data_path, exec_tools)
        if cached_path:
            log_tool_call(session_id, "T-01_implement_fc_CACHE_HIT",
                          {"data_path": data_path}, {"cached_path": cached_path})
            fc_output = cached_path
        else:
            fc_output = await _call_mcp(
                session_id, mcp_tools, "implement_fc",
                {"data_dir": data_path, "exec_tools": exec_tools},
            )
            if fc_output and not isinstance(fc_output, dict):
                set_cache(session_id, data_path, exec_tools, str(fc_output))

        # ── Stage 2: 이상치 제거 (T-02) ─────────────────────────────
        outlier_output = await _call_mcp(
            session_id, mcp_tools, "delete_outlier",
            {
                "data_path":      str(OUTPUT_DIR / "feature_data_common.pickle"),
                "target":         target_col,
                "outlier_method": ag02_params.get("outlier_method", "gaussian"),
            },
        )

        # ── Stage 3: 다중공선성 제거 (T-03 or T-04) ─────────────────
        mc_output = await _call_mcp(
            session_id, mcp_tools, "smart_correlation",
            {
                "data_path":  str(OUTPUT_DIR / "feature_data_outlier.pickle"),
                "target_col": target_col,
                "threshold":  ag02_params.get("threshold", 0.8),
            },
        )

        # ── Stage 4: 데이터 증강 (T-05) ─────────────────────────────
        aug_output = await _call_mcp(
            session_id, mcp_tools, "gaussian_augmentation",
            {
                "data_path":      str(OUTPUT_DIR / "feature_data_multi.pickle"),
                "time_path":      str(OUTPUT_DIR.parent / "time_info.csv"),
                "ycol":           target_col,
                "n_new_samples":  ag02_params.get("n_new_samples", 300),
            },
        )

        # ── Stage 5: Rank Matrix (T-06) ──────────────────────────────
        rank_output = await _call_mcp(
            session_id, mcp_tools, "rank_matrix",
            {
                "data_path": str(OUTPUT_DIR / "feature_data_augmentation.pickle"),
                "fold":      ag02_params.get("fold", 1),
            },
        )

        # ── Stage 6: 최적 모델 선택 (T-07) ──────────────────────────
        model_output = await _call_mcp(
            session_id, mcp_tools, "select_best_model",
            {
                "data_path": output_dir,
                "train":     str(OUTPUT_DIR / "wu_train_sample.pickle"),
                "test":      str(OUTPUT_DIR / "wu_test_sample.pickle"),
                "ncol":      ag02_params.get("ncol", 10),
                "criterion": ag02_params.get("criterion", "MSE"),
            },
        )

    # ── 결과 집계 ────────────────────────────────────────────────────
    result = {
        "output_path":   output_dir,
        "output_dir":    output_dir,
        "stages_done":   ["implement_fc", "delete_outlier", "smart_correlation",
                          "gaussian_augmentation", "rank_matrix", "select_best_model"],
        "model_result":  model_output,
        "rank_result":   rank_output,
    }

    log_tool_call(
        session_id=session_id,
        tool_name="AG-02_fe_agent_complete",
        params={"ag02_params": ag02_params},
        result=result,
    )

    current_results = state.get("agent_results", {})
    current_results["AG-02"] = result

    return {
        "agent_results": current_results,
        "hitl_required": True,   # HITL ② 전처리 확인 트리거
    }


# ── 내부 헬퍼: MCP tool 호출 + 로깅 ─────────────────────────────────

async def _call_mcp(
    session_id: str,
    mcp_tools: dict,
    tool_name: str,
    params: dict,
) -> Any:
    """MCP tool 호출 후 T-20으로 로깅"""
    t0 = time.time()
    tool = mcp_tools.get(tool_name)

    if tool is None:
        error_msg = f"MCP tool '{tool_name}' 을 찾을 수 없습니다."
        log_tool_call(session_id, f"MCP_{tool_name}", params, None, error=error_msg)
        return None

    try:
        result     = await tool.ainvoke(params)
        latency_ms = int((time.time() - t0) * 1000)
        log_tool_call(
            session_id=session_id,
            tool_name=f"MCP_{tool_name}",
            params=params,
            result=result,
            latency_ms=latency_ms,
        )
        return result
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        log_tool_call(
            session_id=session_id,
            tool_name=f"MCP_{tool_name}",
            params=params,
            result=None,
            error=str(e),
            latency_ms=latency_ms,
        )
        return None