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

    # ── 최종 응답 ────────────────────────────────────────────────────
    final_response: str
    # 사용자에게 전달할 최종 응답 (T-17 페르소나 응답 생성기 출력)