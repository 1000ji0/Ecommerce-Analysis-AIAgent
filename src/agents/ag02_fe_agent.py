"""
AG-02 Feature Engineering Agent
USE_MCP_MOCK=true  → 샘플 데이터 사용 (MCP 서버 불필요)
USE_MCP_MOCK=false → 실제 MCP 서버 연결
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MCP_SERVER_URL, OUTPUT_DIR, DEFAULT_TARGET_COL
from state import GraphState
from tools.data.t21_feature_cache import get_cache, set_cache
from tools.output.t20_trace_logger import log_tool_call
from tools.database.sqlite_store import TraceStore

_store  = TraceStore()
USE_MOCK = os.getenv("USE_MCP_MOCK", "true").lower() == "true"


async def fe_agent_node(state: GraphState) -> dict[str, Any]:
    session_id  = state.get("session_id", "")
    data_meta   = state.get("data_meta", {})
    plan        = state.get("execution_plan", {})
    ag02_params = plan.get("params", {}).get("AG-02", {})

    if USE_MOCK:
        return await _run_mock(state, session_id, ag02_params)
    else:
        return await _run_mcp(state, session_id, data_meta, ag02_params)


async def _run_mock(state, session_id, ag02_params) -> dict[str, Any]:
    """MCP 서버 없이 기존 샘플 데이터 사용"""
    log_tool_call(session_id, "AG-02_MOCK", {}, {"output_dir": str(OUTPUT_DIR)})

    mock_stages = [
        ("T-01_implement_fc",          {"exec_tools": ag02_params.get("exec_tools", ["fpca"])}),
        ("T-02_delete_outlier",        {"outlier_method": ag02_params.get("outlier_method", "gaussian")}),
        ("T-03_smart_correlation",     {"threshold": ag02_params.get("threshold", 0.8)}),
        ("T-05_gaussian_augmentation", {"n_new_samples": ag02_params.get("n_new_samples", 300)}),
        ("T-06_rank_matrix",           {"fold": ag02_params.get("fold", 1)}),
        ("T-07_select_best_model",     {"ncol": ag02_params.get("ncol", 10)}),
    ]
    for tool_name, params in mock_stages:
        log_tool_call(session_id, f"MOCK_{tool_name}", params, {"status": "mock"})
        import asyncio; await asyncio.sleep(0.05)

    result = {
        "output_path": str(OUTPUT_DIR),
        "output_dir":  str(OUTPUT_DIR),
        "mode":        "mock",
        "stages_done": [s[0] for s in mock_stages],
    }
    log_tool_call(session_id, "AG-02_complete", {}, result)

    current = state.get("agent_results", {})
    current["AG-02"] = result
    return {"agent_results": current, "hitl_required": True}


async def _run_mcp(state, session_id, data_meta, ag02_params) -> dict[str, Any]:
    """실제 MCP 서버 연결 — langchain-mcp-adapters 0.1.0+ 방식"""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    target_col = ag02_params.get("target_col", DEFAULT_TARGET_COL)
    output_dir = str(OUTPUT_DIR)

    # 0.1.0+ 방식: async with 대신 직접 get_tools() 호출
    client    = MultiServerMCPClient({"analytics": {"url": MCP_SERVER_URL, "transport": "sse"}})
    mcp_tools = {t.name: t for t in await client.get_tools()}

    exec_tools  = ag02_params.get("exec_tools", ["fpca", "nds", "timeseries", "stats"])
    cached_path = get_cache(session_id, data_meta.get("path", ""), exec_tools)

    if cached_path:
        log_tool_call(session_id, "T-01_CACHE_HIT", {}, {"cached": cached_path})
    else:
        await _call(session_id, mcp_tools, "implement_fc",
                    {"data_dir": data_meta.get("path", ""), "exec_tools": exec_tools})

    await _call(session_id, mcp_tools, "delete_outlier", {
        "data_path": str(OUTPUT_DIR / "feature_data_common.pickle"),
        "target": target_col,
        "outlier_method": ag02_params.get("outlier_method", "gaussian"),
    })
    await _call(session_id, mcp_tools, "smart_correlation", {
        "data_path": str(OUTPUT_DIR / "feature_data_outlier.pickle"),
        "target_col": target_col,
        "threshold": ag02_params.get("threshold", 0.8),
    })
    await _call(session_id, mcp_tools, "gaussian_augmentation", {
        "data_path": str(OUTPUT_DIR / "feature_data_multi.pickle"),
        "time_path": str(OUTPUT_DIR.parent / "time_info.csv"),
        "ycol": target_col,
        "n_new_samples": ag02_params.get("n_new_samples", 300),
    })
    await _call(session_id, mcp_tools, "rank_matrix", {
        "data_path": str(OUTPUT_DIR / "feature_data_augmentation.pickle"),
        "fold": ag02_params.get("fold", 1),
    })
    model_output = await _call(session_id, mcp_tools, "select_best_model", {
        "data_path": output_dir,
        "train": str(OUTPUT_DIR / "wu_train_sample.pickle"),
        "test":  str(OUTPUT_DIR / "wu_test_sample.pickle"),
        "ncol":  ag02_params.get("ncol", 10),
        "criterion": ag02_params.get("criterion", "MSE"),
    })

    result = {"output_path": output_dir, "output_dir": output_dir,
              "mode": "mcp", "model_result": model_output}
    log_tool_call(session_id, "AG-02_complete", {}, result)

    current = state.get("agent_results", {})
    current["AG-02"] = result
    return {"agent_results": current, "hitl_required": True}


async def _call(session_id, mcp_tools, tool_name, params) -> Any:
    t0   = time.time()
    tool = mcp_tools.get(tool_name)
    if not tool:
        log_tool_call(session_id, f"MCP_{tool_name}", params, None,
                      error=f"'{tool_name}' not found")
        return None
    try:
        result = await tool.ainvoke(params)
        log_tool_call(session_id, f"MCP_{tool_name}", params, result,
                      latency_ms=int((time.time() - t0) * 1000))
        return result
    except Exception as e:
        log_tool_call(session_id, f"MCP_{tool_name}", params, None,
                      error=str(e), latency_ms=int((time.time() - t0) * 1000))
        return None