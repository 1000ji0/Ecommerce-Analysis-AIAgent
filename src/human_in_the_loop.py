"""
human_in_the_loop.py
HITL 워크플로 — 두 단계 구조

Phase A — 정보 수집:
  LLM이 작업에 필요한 요구사항 질문 (자유 텍스트 답변)
  예) "어떤 이상치 제거 방법을 원하시나요?"
  → 사용자: "IQR 방식으로 해줘"

Phase B — 결과 검토:
  에이전트 작업 결과 보여주고 승인/수정/재실행 선택
  예) "전처리 완료됐습니다. 계속 진행할까요?"
  → 사용자: 1.승인 / 2.수정 / 3.재실행

설계 원칙 (문서 기반):
  - 최소 개입: 4개 핵심 포인트에만 적용
  - 비차단: 타임아웃 시 세션 보존
  - STM 기록: hitl_history에 타임스탬프 포함
  - LTM 저장: HITL ④ 승인 시에만 영구 저장
  - 페르소나: 마케터/분석가 관점 메시지
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.output.t20_trace_logger import log_hitl, log_tool_call


# ── Enum ─────────────────────────────────────────────────────────────

class HITLPoint(str, Enum):
    PLAN       = "HITL-①-계획승인"
    PREPROCESS = "HITL-②-전처리확인"
    FEATURE    = "HITL-③-Feature선정확인"
    FINAL      = "HITL-④-최종승인"


class HITLPhase(str, Enum):
    INFO_COLLECT = "info_collect"   # Phase A: 정보 수집
    APPROVAL     = "approval"       # Phase B: 결과 검토


# ── Pydantic State ───────────────────────────────────────────────────

class HITLResponse(BaseModel):
    """사용자 HITL 응답"""
    response:       str  = Field(description="승인 | 수정 | 재실행")
    user_answer:    str  = Field(default="", description="Phase A 자유 답변")
    modified_input: dict = Field(default_factory=dict)
    timestamp:      str  = Field(default_factory=lambda: _now())


class HITLState(BaseModel):
    """HITL 워크플로 전체 상태"""
    session_id:    str  = Field(default="")
    task:          str  = Field(default="")
    task_context:  dict = Field(default_factory=dict)
    hitl_point:    str  = Field(default="")

    # Phase A
    llm_question:  str  = Field(default="")
    user_answer:   str  = Field(default="")

    # Phase B
    hitl_response: HITLResponse | None = Field(default=None)
    hitl_history:  list[HITLResponse]  = Field(default_factory=list)

    # 제어
    current_phase: str  = Field(default=HITLPhase.INFO_COLLECT.value)
    is_approved:   bool = Field(default=False)
    needs_retry:   bool = Field(default=False)
    is_complete:   bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True


# ── LLM ─────────────────────────────────────────────────────────────

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


# ── Phase A 노드: 정보 수집 ──────────────────────────────────────────

def phase_a_generate_node(state: HITLState) -> dict[str, Any]:
    """
    Phase A - 1단계: LLM이 ?로 끝나는 질문 생성
    작업 전 사용자 요구사항·구체적 정보 수집용
    """
    question = _generate_question_llm(
        task=state.task,
        task_context=state.task_context,
        hitl_point=state.hitl_point,
    )
    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_phase_a_question",
        params={"task": state.task, "hitl_point": state.hitl_point},
        result={"question": question},
    )
    return {"llm_question": question, "current_phase": HITLPhase.INFO_COLLECT.value}


def phase_a_collect_node(state: HITLState) -> dict[str, Any]:
    """
    Phase A - 2단계: interrupt()로 자유 텍스트 답변 수집
    선택지 없음 — 사용자가 자유롭게 요구사항 입력
    """
    payload = {
        "phase":        "A",
        "hitl_point":   state.hitl_point,
        "llm_question": state.llm_question,
        "input_type":   "free_text",   # main.py에서 자유 입력 처리
    }
    raw = interrupt(payload)

    user_answer = raw.get("user_answer", "") if isinstance(raw, dict) else str(raw)

    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_phase_a_collect",
        params={"question": state.llm_question},
        result={"user_answer": user_answer},
    )

    # 사용자 답변을 컨텍스트에 반영
    updated_context = {**state.task_context, "user_requirements": user_answer}

    return {
        "user_answer":   user_answer,
        "task_context":  updated_context,
        "current_phase": HITLPhase.APPROVAL.value,
    }


# ── Phase B 노드: 결과 검토 ──────────────────────────────────────────

def phase_b_show_node(state: HITLState) -> dict[str, Any]:
    """
    Phase B - 1단계: 작업 결과 요약 표시
    사용자 답변 반영 후 최종 계획/결과 보여줌
    """
    summary = _summarize_context(state.task_context, state.hitl_point)
    updated = {**state.task_context, "context_summary": summary}

    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_phase_b_show",
        params={"hitl_point": state.hitl_point},
        result={"summary": summary},
    )
    return {"task_context": updated}


def phase_b_approve_node(state: HITLState) -> dict[str, Any]:
    """
    Phase B - 2단계: interrupt()로 승인/수정/재실행 수집
    에이전트 결과 검토 후 진행 여부 결정
    """
    payload = {
        "phase":           "B",
        "hitl_point":      state.hitl_point,
        "message":         state.task_context.get("context_summary", ""),
        "user_answer":     state.user_answer,
        "options":         ["승인", "수정", "재실행"],
        "input_type":      "selection",   # main.py에서 선택 처리
        "context":         state.task_context,
    }
    raw = interrupt(payload)

    if isinstance(raw, dict):
        response       = raw.get("response", "승인")
        modified_input = raw.get("modified_input", {})
    else:
        response, modified_input = str(raw), {}

    if response not in {"승인", "수정", "재실행"}:
        response = "승인"

    hitl_resp = HITLResponse(
        response=response,
        user_answer=state.user_answer,
        modified_input=modified_input,
    )

    history = list(state.hitl_history)
    history.append(hitl_resp)

    log_hitl(
        session_id=state.session_id,
        hitl_point=state.hitl_point,
        message=state.llm_question,
        response=response,
        decision=response,
    )

    return {
        "hitl_response": hitl_resp,
        "hitl_history":  history,
        "is_approved":   response == "승인",
        "needs_retry":   response == "재실행",
    }


# ── 결과 처리 노드 ───────────────────────────────────────────────────

def complete_node(state: HITLState) -> dict[str, Any]:
    log_tool_call(state.session_id, "HITL_complete",
                  {"hitl_point": state.hitl_point}, {"status": "approved"})
    return {"is_complete": True}


def retry_node(state: HITLState) -> dict[str, Any]:
    log_tool_call(state.session_id, "HITL_retry",
                  {"hitl_point": state.hitl_point}, {"status": "retry"})
    return {"needs_retry": False, "llm_question": "", "user_answer": ""}


def modify_node(state: HITLState) -> dict[str, Any]:
    modified = state.hitl_response.modified_input if state.hitl_response else {}
    updated  = {**state.task_context, **modified}
    log_tool_call(state.session_id, "HITL_modify",
                  {"modified": modified}, {"updated_context": updated})
    return {"task_context": updated, "is_approved": False}


# ── 조건부 엣지 ──────────────────────────────────────────────────────

def route_after_approval(state: HITLState) -> Literal["complete", "retry", "modify"]:
    if not state.hitl_response:
        return "complete"
    r = state.hitl_response.response
    if r == "승인":   return "complete"
    if r == "재실행": return "retry"
    return "modify"


def route_after_retry(state: HITLState) -> Literal["phase_a_generate"]:
    return "phase_a_generate"


def route_after_modify(state: HITLState) -> Literal["phase_a_generate"]:
    return "phase_a_generate"


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_hitl_graph():
    """
    두 단계 HITL 워크플로 그래프

    START
      → phase_a_generate  (LLM 질문 생성)
      → phase_a_collect   (interrupt — 자유 텍스트 수집)
      → phase_b_show      (결과 요약)
      → phase_b_approve   (interrupt — 승인/수정/재실행)
      → [승인]  → complete → END
      → [수정]  → modify  → phase_a_generate (반복)
      → [재실행] → retry  → phase_a_generate (반복)
    """
    builder = StateGraph(HITLState)

    builder.add_node("phase_a_generate", phase_a_generate_node)
    builder.add_node("phase_a_collect",  phase_a_collect_node)
    builder.add_node("phase_b_show",     phase_b_show_node)
    builder.add_node("phase_b_approve",  phase_b_approve_node)
    builder.add_node("complete",         complete_node)
    builder.add_node("retry",            retry_node)
    builder.add_node("modify",           modify_node)

    builder.add_edge(START,               "phase_a_generate")
    builder.add_edge("phase_a_generate",  "phase_a_collect")
    builder.add_edge("phase_a_collect",   "phase_b_show")
    builder.add_edge("phase_b_show",      "phase_b_approve")

    builder.add_conditional_edges("phase_b_approve", route_after_approval, {
        "complete": "complete",
        "retry":    "retry",
        "modify":   "modify",
    })
    builder.add_conditional_edges("retry",  route_after_retry,  {"phase_a_generate": "phase_a_generate"})
    builder.add_conditional_edges("modify", route_after_modify, {"phase_a_generate": "phase_a_generate"})

    builder.add_edge("complete", END)

    return builder.compile(checkpointer=MemorySaver())


hitl_graph = build_hitl_graph()


# ── graph.py에서 사용하는 헬퍼 함수 ─────────────────────────────────

# ── HITL 포인트별 질문 템플릿 ───────────────────────────────────────
# 도메인 지식 불필요, 누구나 답할 수 있는 비즈니스 수준 질문

HITL_QUESTION_PROMPTS = {
    HITLPoint.PLAN.value: (
        "이커머스 데이터 분석을 시작합니다. "
        "분석에서 가장 중요하게 보고 싶은 것을 한 가지만 물어보세요. "
        "예: 매출, 특정 채널, 고객 행동 등 비즈니스 관점으로 질문하세요. "
        "기술적인 질문은 절대 하지 마세요. "
        "반드시 ?로 끝내세요."
    ),
    HITLPoint.PREPROCESS.value: (
        "데이터 전처리가 완료됐습니다. "
        "분석에서 제외하고 싶은 항목이 있는지 간단히 물어보세요. "
        "예: 특정 기간, 특정 채널, 이상한 데이터 등. "
        "기술적인 질문은 절대 하지 마세요. "
        "반드시 ?로 끝내세요."
    ),
    HITLPoint.FEATURE.value: (
        "분석 결과가 나왔습니다. "
        "이 결과 중 더 자세히 알고 싶은 부분이 있는지 물어보세요. "
        "예: 특정 변수, 특정 채널의 영향력 등 비즈니스 관점으로 질문하세요. "
        "기술적인 질문은 절대 하지 마세요. "
        "반드시 ?로 끝내세요."
    ),
    HITLPoint.FINAL.value: (
        "보고서가 완성됐습니다. "
        "이 보고서를 누구에게 보여줄 예정인지 간단히 물어보세요. "
        "예: 팀 내부 공유, 임원 보고, 클라이언트 제출 등. "
        "반드시 ?로 끝내세요."
    ),
}


def _generate_question_llm(task: str, task_context: dict, hitl_point: str) -> str:
    """
    LLM으로 ?로 끝나는 질문 생성

    원칙:
    - 도메인 지식 불필요 (퍼포먼스 마케터 수준)
    - 기술적 파라미터 X, 비즈니스 목적 O
    - 한 문장, 짧고 명확하게
    """
    system_prompt = HITL_QUESTION_PROMPTS.get(
        hitl_point,
        (
            "사용자에게 비즈니스 목적으로 한 가지만 질문하세요. "
            "기술적인 질문은 절대 하지 마세요. "
            "반드시 ?로 끝내세요."
        )
    )

    prompt = (
        f"현재 상황: {hitl_point}\n"
        f"작업: {task}\n"
        f"지시: {system_prompt}"
    )

    llm      = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    question = response.content.strip()

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    return question


def _summarize_context(task_context: dict, hitl_point: str) -> str:
    """HITL 포인트별 컨텍스트 요약"""
    summaries = {
        HITLPoint.PLAN.value: (
            f"분석 단계: {task_context.get('stages', [])}\n"
            f"설명: {task_context.get('description', '')}\n"
            f"사용자 요구사항: {task_context.get('user_requirements', '없음')}"
        ),
        HITLPoint.PREPROCESS.value: (
            f"완료된 단계: {task_context.get('stages_done', [])}\n"
            f"출력 경로: {task_context.get('output_path', '')}\n"
            f"사용자 요구사항: {task_context.get('user_requirements', '없음')}"
        ),
        HITLPoint.FEATURE.value: (
            f"변수 중요도: {task_context.get('final_ranking', {})}\n"
            f"분석 유형: {task_context.get('task', '')}\n"
            f"사용자 요구사항: {task_context.get('user_requirements', '없음')}"
        ),
        HITLPoint.FINAL.value: (
            f"보고서 경로: {task_context.get('report_path', '')}\n"
            f"요약: {task_context.get('report_summary', '')[:200]}\n"
            f"사용자 요구사항: {task_context.get('user_requirements', '없음')}"
        ),
    }
    return summaries.get(hitl_point, str(task_context)[:300])


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")