"""
human_in_the_loop.py
HITL 워크플로 — 두 단계 구조 + LLM 동적 선택지

Phase A — 정보 수집:
  LLM 질문 + 동적 선택지 제공
  사용자가 선택지 중 고르거나 직접 입력

Phase B — 결과 검토:
  승인 / 수정 / 재실행 선택
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
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
    INFO_COLLECT = "info_collect"
    APPROVAL     = "approval"


# ── Pydantic State ───────────────────────────────────────────────────

class HITLResponse(BaseModel):
    response:       str  = Field(description="승인 | 수정 | 재실행")
    user_answer:    str  = Field(default="")
    modified_input: dict = Field(default_factory=dict)
    timestamp:      str  = Field(default_factory=lambda: _now())


class HITLState(BaseModel):
    session_id:    str  = Field(default="")
    task:          str  = Field(default="")
    task_context:  dict = Field(default_factory=dict)
    hitl_point:    str  = Field(default="")
    llm_question:  str  = Field(default="")
    llm_choices:   list = Field(default_factory=list)
    user_answer:   str  = Field(default="")
    hitl_response: HITLResponse | None = Field(default=None)
    hitl_history:  list[HITLResponse]  = Field(default_factory=list)
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


# ── Phase A 노드 ─────────────────────────────────────────────────────

def phase_a_generate_node(state: HITLState) -> dict[str, Any]:
    question    = _generate_question_llm(
        task=state.task, task_context=state.task_context,
        hitl_point=state.hitl_point,
    )
    llm_choices = _generate_llm_choices(
        task=state.task, task_context=state.task_context,
        hitl_point=state.hitl_point,
    )
    log_tool_call(state.session_id, "HITL_phase_a_question",
                  {"task": state.task},
                  {"question": question, "choices": llm_choices})
    return {
        "llm_question":  question,
        "llm_choices":   llm_choices,
        "current_phase": HITLPhase.INFO_COLLECT.value,
    }


def phase_a_collect_node(state: HITLState) -> dict[str, Any]:
    payload = {
        "phase":        "A",
        "hitl_point":   state.hitl_point,
        "llm_question": state.llm_question,
        "llm_choices":  state.llm_choices,
        "input_type":   "free_text",
    }
    raw         = interrupt(payload)
    user_answer = raw.get("user_answer", "") if isinstance(raw, dict) else str(raw)

    log_tool_call(state.session_id, "HITL_phase_a_collect",
                  {"question": state.llm_question}, {"user_answer": user_answer})

    updated_context = {**state.task_context, "user_requirements": user_answer}
    return {
        "user_answer":   user_answer,
        "task_context":  updated_context,
        "current_phase": HITLPhase.APPROVAL.value,
    }


# ── Phase B 노드 ─────────────────────────────────────────────────────

def phase_b_show_node(state: HITLState) -> dict[str, Any]:
    summary = _summarize_context(state.task_context, state.hitl_point)
    updated = {**state.task_context, "context_summary": summary}
    log_tool_call(state.session_id, "HITL_phase_b_show",
                  {"hitl_point": state.hitl_point}, {"summary": summary})
    return {"task_context": updated}


def phase_b_approve_node(state: HITLState) -> dict[str, Any]:
    payload = {
        "phase":       "B",
        "hitl_point":  state.hitl_point,
        "message":     state.task_context.get("context_summary", ""),
        "user_answer": state.user_answer,
        "options":     ["승인", "수정", "재실행"],
        "input_type":  "selection",
        "context":     state.task_context,
    }
    raw = interrupt(payload)

    if isinstance(raw, dict):
        response, modified_input = raw.get("response", "승인"), raw.get("modified_input", {})
    else:
        response, modified_input = str(raw), {}

    if response not in {"승인", "수정", "재실행"}:
        response = "승인"

    hitl_resp = HITLResponse(
        response=response, user_answer=state.user_answer,
        modified_input=modified_input,
    )
    history = list(state.hitl_history)
    history.append(hitl_resp)

    log_hitl(state.session_id, state.hitl_point,
             state.llm_question, response, decision=response)
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
    if not state.hitl_response: return "complete"
    r = state.hitl_response.response
    if r == "승인":   return "complete"
    if r == "재실행": return "retry"
    return "modify"

def route_after_retry(state: HITLState)  -> Literal["phase_a_generate"]: return "phase_a_generate"
def route_after_modify(state: HITLState) -> Literal["phase_a_generate"]: return "phase_a_generate"


# ── 그래프 빌드 ──────────────────────────────────────────────────────

def build_hitl_graph():
    builder = StateGraph(HITLState)
    builder.add_node("phase_a_generate", phase_a_generate_node)
    builder.add_node("phase_a_collect",  phase_a_collect_node)
    builder.add_node("phase_b_show",     phase_b_show_node)
    builder.add_node("phase_b_approve",  phase_b_approve_node)
    builder.add_node("complete",         complete_node)
    builder.add_node("retry",            retry_node)
    builder.add_node("modify",           modify_node)

    builder.add_edge(START,              "phase_a_generate")
    builder.add_edge("phase_a_generate", "phase_a_collect")
    builder.add_edge("phase_a_collect",  "phase_b_show")
    builder.add_edge("phase_b_show",     "phase_b_approve")

    builder.add_conditional_edges("phase_b_approve", route_after_approval,
                                  {"complete": "complete", "retry": "retry", "modify": "modify"})
    builder.add_conditional_edges("retry",  route_after_retry,  {"phase_a_generate": "phase_a_generate"})
    builder.add_conditional_edges("modify", route_after_modify, {"phase_a_generate": "phase_a_generate"})
    builder.add_edge("complete", END)

    return builder.compile(checkpointer=MemorySaver())


hitl_graph = build_hitl_graph()


# ── graph.py에서 사용하는 헬퍼 함수 ─────────────────────────────────

HITL_QUESTION_PROMPTS = {
    HITLPoint.PLAN.value: (
        "이커머스 데이터 분석을 시작합니다. "
        "분석에서 가장 중요하게 보고 싶은 것을 한 가지만 물어보세요. "
        "예: 매출, 특정 채널, 고객 행동 등 비즈니스 관점으로 질문하세요. "
        "기술적인 질문은 절대 하지 마세요. 반드시 ?로 끝내세요."
    ),
    HITLPoint.PREPROCESS.value: (
        "데이터 전처리가 완료됐습니다. "
        "분석에서 제외하고 싶은 항목이 있는지 간단히 물어보세요. "
        "예: 특정 기간, 특정 채널, 이상한 데이터 등. "
        "기술적인 질문은 절대 하지 마세요. 반드시 ?로 끝내세요."
    ),
    HITLPoint.FEATURE.value: (
        "분석 결과가 나왔습니다. "
        "이 결과 중 더 자세히 알고 싶은 부분이 있는지 물어보세요. "
        "예: 특정 변수, 특정 채널의 영향력 등 비즈니스 관점으로 질문하세요. "
        "기술적인 질문은 절대 하지 마세요. 반드시 ?로 끝내세요."
    ),
    HITLPoint.FINAL.value: (
        "보고서가 완성됐습니다. "
        "이 보고서를 누구에게 보여줄 예정인지 간단히 물어보세요. "
        "예: 팀 내부 공유, 임원 보고, 클라이언트 제출 등. "
        "반드시 ?로 끝내세요."
    ),
}


def _generate_question_llm(task: str, task_context: dict, hitl_point: str) -> str:
    """LLM으로 ?로 끝나는 질문 생성 — 도메인 지식 불필요, 비즈니스 관점"""
    system_prompt = HITL_QUESTION_PROMPTS.get(
        hitl_point,
        "사용자에게 비즈니스 목적으로 한 가지만 질문하세요. 반드시 ?로 끝내세요."
    )
    prompt = f"현재 상황: {hitl_point}\n작업: {task}\n지시: {system_prompt}"

    llm      = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    question = response.content if isinstance(response.content, str) else str(response.content)
    question = question.strip()

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    return question


def _generate_llm_choices(task: str, task_context: dict, hitl_point: str) -> list[str]:
    """
    LLM이 상황에 맞는 선택지 3개 동적 생성
    사용자가 선택지 중 고르거나 직접 입력할 수 있게

    예시:
      task="채널별 ROAS 분석"
      → ["전체 채널 비교", "상위 성과 채널만", "최근 30일 기준으로"]
    """
    prompt = (
        f"작업: '{task}'\n"
        f"단계: {hitl_point}\n"
        f"컨텍스트: {str(task_context)[:300]}\n\n"
        "사용자가 선택할 수 있는 구체적인 옵션 3개를 생성해라.\n"
        "각 옵션은 10자 이내로 간결하게.\n"
        "도메인 지식 없이도 고를 수 있는 비즈니스 관점 옵션으로.\n"
        "JSON 배열로만 반환. 예: [\"전체 채널 비교\", \"상위 채널만\", \"기간 필터 적용\"]\n"
        "마크다운 금지."
    )
    llm      = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    content  = response.content if isinstance(response.content, str) else str(response.content)
    raw      = re.sub(r"```(?:json)?|```", "", content).strip()

    try:
        choices = json.loads(raw)
        if isinstance(choices, list):
            return [str(c) for c in choices[:3]]
    except Exception:
        pass
    return []


def _summarize_context(task_context: dict, hitl_point: str) -> str:
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