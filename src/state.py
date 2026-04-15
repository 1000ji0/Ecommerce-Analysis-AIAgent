"""
LangGraph GraphState 정의
그래프 전체의 공유 상태 — 노드 간 데이터 전달 통로

STM(단기 메모리) 역할을 겸함:
- 세션 내 모든 데이터는 이 state에 저장
- 각 노드는 state를 읽고 업데이트해서 반환
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict):

    # ── 사용자 입력 ──────────────────────────────────────────────────
    user_input: str
    # 사용자 자연어 요청
    # 예) "이상치 제거하고 변수 중요도 분석해줘"

    session_id: str
    # 세션 식별자 — 로그·캐시·보고서 저장 경로에 사용
    # 예) "20260401_153042"

    # ── 데이터 메타정보 ──────────────────────────────────────────────
    data_meta: dict[str, Any]
    # T-08 upload_handler 반환값
    # {
    #   "path":      str,    # 저장된 파일 경로
    #   "filename":  str,
    #   "encoding":  str,    # 감지된 인코딩
    #   "preview":   dict,   # columns, dtypes, sample
    #   "row_count": int,    # 전체 행 수
    #   "col_count": int,
    #   "size_mb":   float,
    # }

    # ── 실행 계획 ────────────────────────────────────────────────────
    execution_plan: dict[str, Any]
    # T-15 plan_parser 반환값
    # {
    #   "stages":      ["AG-02", "AG-04", "AG-05"],
    #   "params":      {"AG-02": {...}, "AG-04": {...}},
    #   "description": "분석 계획 요약",
    # }

    # ── 에이전트 실행 결과 ───────────────────────────────────────────
    agent_results: dict[str, Any]
    # 각 agent 실행 결과 누적
    # {
    #   "AG-02": {"output_path": str, "row_count": int, ...},
    #   "AG-04": {"insights": [...], "actions": [...], ...},
    #   "AG-05": {"report_path": str, ...},
    # }

    # ── HITL ─────────────────────────────────────────────────────────
    hitl_required: bool
    # 현재 노드에서 interrupt() 필요 여부
    # graph.py의 조건부 엣지에서 이 값을 보고 분기

    hitl_history: list[dict[str, Any]]
    # HITL 승인/수정 이력
    # [
    #   {"point": "HITL-①-계획승인", "response": "승인"},
    #   {"point": "HITL-②-전처리확인", "response": "수정", "modified_input": {...}},
    # ]

    # ── 메시지 히스토리 ──────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    # LangGraph 메시지 히스토리
    # add_messages reducer — 새 메시지를 기존 리스트에 추가
    # LLM 호출 시 대화 맥락 유지용

    # ── 사용자 프로필 ─────────────────────────────────────────────────
    user_profile: dict[str, Any]
    # 세션 시작 시 수집한 사용자 정보
    # {
    #   "role":    "퍼포먼스 마케터" | "데이터 분석가 / 데이터 사이언티스트" | ...
    #   "purpose": "광고 성과 확인" | "매출 원인 파악" | ...
    #   "guide":   "맞춤 안내 메시지"
    # }

# ── 라우팅 ───────────────────────────────────────────────────────
    next_agent:    str
    # AG-01이 결정한 다음 실행 agent
    # "AG-02" | "AG-03" | "AG-04" | "AG-05" | "hitl_plan" | "respond"

    current_agent: str
    # 현재 실행 중인 agent (조건부 엣지 분기용)
    # "AG-01" | "AG-02" | "AG-03" | "AG-04" | "AG-05"

    # ── 최종 응답 ────────────────────────────────────────────────────
    final_response: str

    # 사용자에게 전달할 최종 응답 (T-17 페르소나 응답 생성기 출력)