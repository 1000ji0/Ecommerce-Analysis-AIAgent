"""
T-16 HITL 인터럽터
LangGraph interrupt() 기반 사용자 승인/수정/재실행 처리

HITL 4개 포인트:
  ① 분석 계획 승인   (AG-01 계획 수립 후)
  ② 전처리 결과 확인 (AG-02 완료 후)
  ③ 분석 결과 확인   (AG-04 완료 후)
  ④ 최종 보고서 승인 (AG-05 완료 후)

특징:
- hitl_interrupt: response 타입 방어 (dict/str/None 모두 처리)
- hitl_interrupt: 응답 유효성 검증 (HITL_OPTIONS 외 값은 "승인"으로 처리)
- 각 포인트별 헬퍼: context 핵심 요약만 포함 (UI 과부하 방지)
"""
from langgraph.types import interrupt
from tools.output.t20_trace_logger import log_hitl

# HITL 옵션
HITL_OPTIONS  = ["승인", "수정", "재실행"]
VALID_RESPONSES = set(HITL_OPTIONS)


###### main 함수: HITL 인터럽트 ######
def hitl_interrupt(
    session_id: str,
    hitl_point: str,
    message: str,
    context: dict | None = None,
) -> dict:
    """
    interrupt() 호출 → 사용자 응답 대기 → 결과 반환

    Args:
        session_id:  현재 세션 ID
        hitl_point:  HITL 포인트 식별자 (예: "HITL-①-계획승인")
        message:     사용자에게 보여줄 메시지
        context:     함께 보여줄 데이터 (핵심 요약만 포함할 것)

    Returns:
        {
            "response":       "승인" | "수정" | "재실행",
            "modified_input": dict   # 수정 시 사용자가 입력한 내용
        }
    """
    payload = {
        "message":    message,
        "options":    HITL_OPTIONS,
        "hitl_point": hitl_point,
    } # 사용자에게 보여줄 내용
    if context:
        payload["context"] = context

    # LangGraph interrupt — 여기서 실행이 멈추고 사용자 입력 대기
    raw_response = interrupt(payload)

    # response 타입 방어 (dict / str / None 모두 처리)
    if isinstance(raw_response, dict):
        user_response  = raw_response.get("response", "승인")
        modified_input = raw_response.get("modified_input", {})
    elif isinstance(raw_response, str):
        user_response  = raw_response
        modified_input = {}
    else:
        user_response  = "승인"
        modified_input = {}

    # 응답 유효성 검증 — HITL_OPTIONS 외 값은 "승인"으로 처리
    if user_response not in VALID_RESPONSES:
        user_response = "승인"

    # trace 기록
    log_hitl(
        session_id=session_id,
        hitl_point=hitl_point,
        message=message,
        response=user_response,
    )

    return {
        "response":       user_response,
        "modified_input": modified_input,
    }


# ── HITL 포인트별 헬퍼 ───────────────────────────────────────────────

def hitl_plan_approval(session_id: str, plan: dict) -> dict:
    """
    HITL ① 분석 계획 승인
    context: stages, params, description 핵심만 포함
    """
    summary = {
        "stages":      plan.get("stages", []),
        "params":      plan.get("params", {}),
        "description": plan.get("description", ""),
    }
    return hitl_interrupt(
        session_id=session_id,
        hitl_point="HITL-①-계획승인",
        message="분석 계획을 확인하고 승인해주세요.",
        context=summary,
    )


def hitl_preprocessing_check(session_id: str, result: dict) -> dict:
    """
    HITL ② 전처리 결과 확인
    context: 파일 경로, 행/열 수, 제거된 행 수만 포함
    """
    summary = {
        "output_path":  result.get("output_path", ""),
        "row_count":    result.get("row_count", "?"),
        "col_count":    result.get("col_count", "?"),
        "removed_rows": result.get("removed_rows", 0),
        "stages_done":  result.get("stages_done", []),
    }
    return hitl_interrupt(
        session_id=session_id,
        hitl_point="HITL-②-전처리확인",
        message="전처리 결과를 확인해주세요.",
        context=summary,
    )


def hitl_analysis_check(session_id: str, result: dict) -> dict:
    """
    HITL ③ 분석 결과 확인
    context: 인사이트, 액션, 요약만 포함 (전체 분석 결과 X)
    """
    summary = {
        "insights": result.get("insights", [])[:3],
        "actions":  result.get("actions",  [])[:3],
        "summary":  result.get("summary",  ""),
    }
    return hitl_interrupt(
        session_id=session_id,
        hitl_point="HITL-③-결과확인",
        message="분석 결과를 확인해주세요.",
        context=summary,
    )


def hitl_final_approval(session_id: str, report_path: str, report_summary: str = "") -> dict:
    """
    HITL ④ 최종 보고서 승인
    context: 보고서 경로 + 요약
    """
    summary = {
        "report_path":    report_path,
        "report_summary": report_summary,
    }
    return hitl_interrupt(
        session_id=session_id,
        hitl_point="HITL-④-최종승인",
        message="최종 보고서를 확인하고 승인해주세요.",
        context=summary,
    )