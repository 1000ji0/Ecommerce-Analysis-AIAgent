"""
T-20 Reasoning Trace Logger
- 모든 Tool 호출·결과·HITL 이력을 세션별 MD 파일로 기록
- 동시에 SQLite DB에도 저장
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..database.sqlite_store import TraceStore

logger = logging.getLogger(__name__)

_store = TraceStore()


def _project_root() -> Path:
    """
    src/tools/output/t20_trace_logger.py -> 프로젝트 루트
    """
    return Path(__file__).resolve().parents[3]


def _log_base_dir() -> Path:
    path = _project_root() / "logs" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_log_file(session_id: str) -> Path:
    session_dir = _log_base_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / "trace.md"


def _now_text() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _safe_json_text(value: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except Exception:
        text = str(value)

    if len(text) > limit:
        return text[:limit] + "\n... (truncated)"
    return text


def _append_markdown_block(session_id: str, content: str) -> None:
    try:
        log_file = _session_log_file(session_id)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as exc:
        logger.exception("[T-20] markdown logging failed: %s", exc)


def log_tool_call(
    session_id: str,
    tool_name: str,
    params: dict[str, Any],
    result: Any,
    error: str | None = None,
    *,
    model_name: str | None = None,
    prompt_version: str | None = None,
    tool_version: str | None = None,
    data_version: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """
    Tool 호출 및 결과를
    1) logs/sessions/<session_id>/trace.md
    2) data/agent_trace.db
    둘 다 기록
    """
    timestamp = _now_text()
    status = "ERROR" if error else "OK"

    md = []
    md.append(f"\n## [{timestamp}] {tool_name} — {status}\n")
    md.append("- 파라미터\n")
    md.append("```json\n")
    md.append(_safe_json_text(params))
    md.append("\n```\n")

    if error:
        md.append("- 에러\n")
        md.append("```text\n")
        md.append(str(error))
        md.append("\n```\n")
    else:
        md.append("- 결과\n")
        md.append("```json\n")
        md.append(_safe_json_text(result))
        md.append("\n```\n")

    _append_markdown_block(session_id, "".join(md))

    try:
        _store.log_event(
            session_id=session_id,
            event_type="error" if error else "tool_result",
            event_name=tool_name,
            input_payload=params,
            output_payload=None if error else result,
            error_message=error,
            model_name=model_name,
            prompt_version=prompt_version,
            tool_version=tool_version,
            data_version=data_version,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.exception("[T-20] DB logging failed: %s", exc)


def log_hitl(
    session_id: str,
    hitl_point: str,
    message: str,
    response: str,
    *,
    decision: str | None = None,
) -> None:
    """
    HITL 이력을 MD + SQLite에 기록
    """
    timestamp = _now_text()

    md = []
    md.append(f"\n## [{timestamp}] HITL — {hitl_point}\n")
    md.append("- 메시지\n")
    md.append("```text\n")
    md.append(message)
    md.append("\n```\n")
    md.append("- 사용자 응답\n")
    md.append("```text\n")
    md.append(response)
    md.append("\n```\n")

    if decision:
        md.append(f"- 결정: {decision}\n")

    _append_markdown_block(session_id, "".join(md))

    try:
        _store.log_event(
            session_id=session_id,
            event_type="hitl",
            event_name=hitl_point,
            input_payload={"message": message},
            output_payload={"response": response},
            hitl_flag=True,
            hitl_decision=decision,
        )
    except Exception as exc:
        logger.exception("[T-20] DB HITL logging failed: %s", exc)


def log_final_response(
    session_id: str,
    response: str,
    *,
    persona: str | None = None,
    model_name: str | None = None,
    prompt_version: str | None = None,
) -> None:
    """
    최종 사용자 응답 기록
    """
    timestamp = _now_text()

    md = []
    md.append(f"\n## [{timestamp}] FINAL_RESPONSE\n")
    if persona:
        md.append(f"- 페르소나: {persona}\n")
    md.append("- 응답\n")
    md.append("```text\n")
    md.append(response[:3000])
    if len(response) > 3000:
        md.append("\n... (truncated)")
    md.append("\n```\n")

    _append_markdown_block(session_id, "".join(md))

    try:
        _store.log_event(
            session_id=session_id,
            event_type="final_response",
            event_name="final_response",
            output_payload={"persona": persona, "response": response},
            model_name=model_name,
            prompt_version=prompt_version,
        )
    except Exception as exc:
        logger.exception("[T-20] DB final response logging failed: %s", exc)


def get_trace(session_id: str) -> str:
    """
    세션의 전체 trace.md 내용 반환
    """
    try:
        log_file = _session_log_file(session_id)
        if not log_file.exists():
            return "trace 없음"
        return log_file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception("[T-20] get_trace failed: %s", exc)
        return "trace 조회 실패"


def get_trace_events(session_id: str) -> list[dict[str, Any]]:
    """
    SQLite에 저장된 세션 이벤트 조회
    """
    try:
        return _store.get_trace_events(session_id)
    except Exception as exc:
        logger.exception("[T-20] get_trace_events failed: %s", exc)
        return []