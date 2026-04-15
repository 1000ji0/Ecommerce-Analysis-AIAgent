"""
human_in_the_loop.py
HITL 워크플로 — Pydantic State + LangGraph interrupt()

단계:
  1) 중단점 제공  — LLM이 상세 정보 질문 생성 (항상 ?로 끝남)
  2) 컨텍스트 제공 — 사용자에게 질문 표시
  3) 피드백 수집  — interrupt()로 사용자 답변 대기
  4) 재개         — 답변 기반으로 실제 작업 실행

HITL 4개 포인트:
  HITL-① 분석 계획 승인
  HITL-② 전처리 결과 확인
  HITL-③ Feature 선정 확인
  HITL-④ 최종 보고서 승인
"""
from __future__ import annotations

import re
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


# ── Pydantic 모델 정의 ───────────────────────────────────────────────

class HITLResponse(BaseModel):
    """사용자 HITL 응답"""
    response:       str  = Field(description="승인 | 수정 | 재실행")
    user_answer:    str  = Field(default="", description="LLM 질문에 대한 사용자 답변")
    modified_input: dict = Field(default_factory=dict, description="수정 시 변경 내용")
    timestamp:      str  = Field(default_factory=lambda: _now())


class HITLPoint(str, Enum):
    PLAN         = "HITL-①-계획승인"
    PREPROCESS   = "HITL-②-전처리확인"
    FEATURE      = "HITL-③-Feature선정확인"
    FINAL        = "HITL-④-최종승인"


class HITLState(BaseModel):
    """HITL 워크플로 전체 상태"""
    # 작업 정보
    session_id:   str = Field(default="")
    task:         str = Field(default="", description="현재 수행할 작업명")
    task_context: dict = Field(default_factory=dict, description="작업 관련 컨텍스트")

    # HITL 포인트
    hitl_point:   str = Field(default="", description="현재 HITL 포인트")

    # LLM 질문
    llm_question: str = Field(default="", description="LLM이 생성한 상세 정보 질문 (?로 끝남)")

    # 사용자 응답
    hitl_response: HITLResponse | None = Field(default=None)
    hitl_history:  list[HITLResponse]  = Field(default_factory=list)

    # 워크플로 제어
    is_approved:   bool = Field(default=False)
    needs_retry:   bool = Field(default=False)
    is_complete:   bool = Field(default=False)
    error:         str  = Field(default="")

    class Config:
        arbitrary_types_allowed = True


# ── LLM 설정 ────────────────────────────────────────────────────────

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


# ── 노드 함수 ────────────────────────────────────────────────────────

def generate_question_node(state: HITLState) -> dict[str, Any]:
    """
    노드 1: 중단점 제공
    LLM이 작업에 맞는 상세 정보 질문 생성
    반드시 ?로 끝나도록 유도
    """
    task         = state.task
    task_context = state.task_context
    hitl_point   = state.hitl_point
    session_id   = state.session_id

    prompt = (
        f"'{task}' 작업을 수행하려고 합니다.\n"
        f"현재 단계: {hitl_point}\n"
        f"컨텍스트: {task_context}\n\n"
        "사용자에게 작업을 더 잘 수행하기 위한 상세 정보를 묻는 질문을 하나 생성해주세요.\n"
        "어떤 종류의 결과가 필요한지, 구체적인 조건이나 요구사항은 무엇인지 질문해주세요.\n"
        "추가 정보가 필요하면, 반드시 응답의 마지막을 물음표로 끝내주세요.\n"
        "질문은 한국어로, 한 문장으로 작성해주세요."
    )

    llm      = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    question = response.content.strip()

    # 반드시 ?로 끝나도록 보정
    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    log_tool_call(
        session_id=session_id,
        tool_name=f"HITL_generate_question",
        params={"task": task, "hitl_point": hitl_point},
        result={"question": question},
    )

    return {"llm_question": question}


def provide_context_node(state: HITLState) -> dict[str, Any]:
    """
    노드 2: 컨텍스트 제공
    사용자에게 현재 상황과 LLM 질문을 표시
    실제 interrupt()는 다음 노드에서 발생
    """
    session_id   = state.session_id
    hitl_point   = state.hitl_point
    task_context = state.task_context
    question     = state.llm_question

    # 컨텍스트 요약 생성
    context_summary = _summarize_context(task_context, hitl_point)

    log_tool_call(
        session_id=session_id,
        tool_name="HITL_provide_context",
        params={"hitl_point": hitl_point},
        result={"context_summary": context_summary, "question": question},
    )

    return {"task_context": {**task_context, "context_summary": context_summary}}


def collect_feedback_node(state: HITLState) -> dict[str, Any]:
    """
    노드 3: 피드백 수집
    interrupt()로 실행 중단 → 사용자 입력 대기
    사용자가 LLM 질문에 답변 + 승인/수정/재실행 선택
    """
    session_id   = state.session_id
    hitl_point   = state.hitl_point
    question     = state.llm_question
    task_context = state.task_context

    # interrupt() — 여기서 실행 중단, 사용자 입력 대기
    payload = {
        "hitl_point":       hitl_point,
        "message":          task_context.get("context_summary", ""),
        "llm_question":     question,
        "options":          ["승인", "수정", "재실행"],
        "context":          task_context,
    }

    raw_response = interrupt(payload)

    # 응답 파싱
    if isinstance(raw_response, dict):
        user_response  = raw_response.get("response", "승인")
        user_answer    = raw_response.get("user_answer", "")
        modified_input = raw_response.get("modified_input", {})
    elif isinstance(raw_response, str):
        user_response  = raw_response
        user_answer    = ""
        modified_input = {}
    else:
        user_response  = "승인"
        user_answer    = ""
        modified_input = {}

    # 유효성 검증
    valid = {"승인", "수정", "재실행"}
    if user_response not in valid:
        user_response = "승인"

    hitl_resp = HITLResponse(
        response=user_response,
        user_answer=user_answer,
        modified_input=modified_input,
    )

    # 이력 추가
    history = list(state.hitl_history)
    history.append(hitl_resp)

    log_hitl(
        session_id=session_id,
        hitl_point=hitl_point,
        message=question,
        response=user_response,
        decision=user_response,
    )

    return {
        "hitl_response": hitl_resp,
        "hitl_history":  history,
    }


def resume_node(state: HITLState) -> dict[str, Any]:
    """
    노드 4: 재개
    사용자 답변과 선택을 기반으로 작업 계속 또는 수정
    """
    session_id    = state.session_id
    hitl_response = state.hitl_response
    response      = hitl_response.response if hitl_response else "승인"
    user_answer   = hitl_response.user_answer if hitl_response else ""

    is_approved = response == "승인"
    needs_retry = response == "재실행"

    # 사용자 답변을 컨텍스트에 반영
    updated_context = {
        **state.task_context,
        "user_answer":    user_answer,
        "hitl_decision":  response,
    }

    log_tool_call(
        session_id=session_id,
        tool_name="HITL_resume",
        params={"response": response, "user_answer": user_answer},
        result={"is_approved": is_approved, "needs_retry": needs_retry},
    )

    return {
        "is_approved":   is_approved,
        "needs_retry":   needs_retry,
        "task_context":  updated_context,
    }


def complete_node(state: HITLState) -> dict[str, Any]:
    """승인 완료 처리"""
    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_complete",
        params={"hitl_point": state.hitl_point},
        result={"status": "approved"},
    )
    return {"is_complete": True}


def retry_node(state: HITLState) -> dict[str, Any]:
    """재실행 처리 — 질문 재생성"""
    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_retry",
        params={"hitl_point": state.hitl_point},
        result={"status": "retry"},
    )
    return {"needs_retry": False, "llm_question": ""}


def modify_node(state: HITLState) -> dict[str, Any]:
    """수정 처리 — 사용자 답변으로 컨텍스트 업데이트"""
    hitl_response = state.hitl_response
    modified      = hitl_response.modified_input if hitl_response else {}

    updated_context = {**state.task_context, **modified}

    log_tool_call(
        session_id=state.session_id,
        tool_name="HITL_modify",
        params={"modified_input": modified},
        result={"updated_context": updated_context},
    )
    return {"task_context": updated_context, "is_approved": False}


# ── 조건부 엣지 ──────────────────────────────────────────────────────

def route_after_resume(state: HITLState) -> Literal["complete", "retry", "modify"]:
    """
    사용자 선택에 따른 분기:
    - 승인   → complete (작업 계속)
    - 재실행 → retry    (처음부터 재시도)
    - 수정   → modify   (컨텍스트 수정 후 재시도)
    """
    hitl_response = state.hitl_response
    if not hitl_response:
        return "complete"

    if hitl_response.response == "승인":
        return "complete"
    elif hitl_response.response == "재실행":
        return "retry"
    else:
        return "modify"


def route_after_retry(state: HITLState) -> Literal["generate_question", "complete"]:
    """재실행 후 질문 재생성"""
    return "generate_question"


def route_after_modify(state: HITLState) -> Literal["generate_question", "complete"]:
    """수정 후 질문 재생성"""
    return "generate_question"


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_hitl_graph():
    """
    HITL 워크플로 그래프 생성

    흐름:
      START
        → generate_question  (LLM 질문 생성)
        → provide_context    (컨텍스트 표시)
        → collect_feedback   (interrupt — 사용자 입력 대기)
        → resume             (답변 처리)
        → [승인] → complete → END
        → [수정] → modify → generate_question (반복)
        → [재실행] → retry → generate_question (반복)
    """
    builder = StateGraph(HITLState)

    # 노드 등록
    builder.add_node("generate_question", generate_question_node)
    builder.add_node("provide_context",   provide_context_node)
    builder.add_node("collect_feedback",  collect_feedback_node)
    builder.add_node("resume",            resume_node)
    builder.add_node("complete",          complete_node)
    builder.add_node("retry",             retry_node)
    builder.add_node("modify",            modify_node)

    # 엣지 연결
    builder.add_edge(START,               "generate_question")
    builder.add_edge("generate_question", "provide_context")
    builder.add_edge("provide_context",   "collect_feedback")
    builder.add_edge("collect_feedback",  "resume")

    # 조건부 엣지
    builder.add_conditional_edges(
        "resume",
        route_after_resume,
        {
            "complete": "complete",
            "retry":    "retry",
            "modify":   "modify",
        },
    )
    builder.add_conditional_edges(
        "retry",
        route_after_retry,
        {"generate_question": "generate_question"},
    )
    builder.add_conditional_edges(
        "modify",
        route_after_modify,
        {"generate_question": "generate_question"},
    )

    builder.add_edge("complete", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


hitl_graph = build_hitl_graph()


# ── 헬퍼 함수 ────────────────────────────────────────────────────────

def _summarize_context(task_context: dict, hitl_point: str) -> str:
    """HITL 포인트별 컨텍스트 요약"""
    summaries = {
        HITLPoint.PLAN.value: (
            f"분석 계획: stages={task_context.get('stages', [])}\n"
            f"설명: {task_context.get('description', '')}"
        ),
        HITLPoint.PREPROCESS.value: (
            f"전처리 결과: {task_context.get('stages_done', [])}\n"
            f"출력 경로: {task_context.get('output_path', '')}"
        ),
        HITLPoint.FEATURE.value: (
            f"변수 중요도: {task_context.get('final_ranking', {})}\n"
            f"분석 유형: {task_context.get('task', '')}"
        ),
        HITLPoint.FINAL.value: (
            f"보고서 경로: {task_context.get('report_path', '')}\n"
            f"요약: {task_context.get('report_summary', '')[:200]}"
        ),
    }
    return summaries.get(hitl_point, str(task_context)[:300])


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


# ── 편의 함수: 각 HITL 포인트 실행 ──────────────────────────────────

def run_hitl(
    session_id: str,
    hitl_point: str,
    task: str,
    task_context: dict,
    config: dict,
) -> HITLState:
    """
    HITL 포인트 실행 헬퍼
    graph.py에서 각 HITL 노드 대신 이 함수를 호출

    Args:
        session_id:   세션 ID
        hitl_point:   HITL 포인트 (HITLPoint enum 값)
        task:         작업명
        task_context: 작업 컨텍스트 (계획, 결과 등)
        config:       LangGraph thread config

    Returns:
        HITLState (hitl_response, is_approved, task_context 포함)
    """
    initial = HITLState(
        session_id=session_id,
        task=task,
        task_context=task_context,
        hitl_point=hitl_point,
    )
    final_state = hitl_graph.invoke(initial, config=config)
    return HITLState(**final_state) if isinstance(final_state, dict) else final_state