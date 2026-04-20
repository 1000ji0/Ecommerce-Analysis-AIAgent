"""
main.py — 대화형 에이전트 진입점

실행:
  python main.py --file data/sample/ecommerce_sample.csv

대화 예시:
  > 피처 중요도 뽑아줘
  > EDA 시각화 만들어줘
  > 전체 분석해줘
  > exit
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from typing import Any, Union, cast
from config import SAMPLE_DATA_DIR, get_session_output_dir
from graph import graph
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from state import GraphState


# ── 페르소나 정의 ────────────────────────────────────────────────────

ROLES = {
    "1": "퍼포먼스 마케터",
    "2": "데이터 분석가 / 데이터 사이언티스트",
    "3": "기획자 / 전략",
    "4": "기타",
}

PURPOSES = {
    "1": "광고 성과 확인 (ROAS, CVR)",
    "2": "매출 원인 파악",
    "3": "데이터 탐색 / EDA",
    "4": "보고서 작성",
    "5": "기타",
}

PERSONA_GUIDE = {
    ("퍼포먼스 마케터", "광고 성과 확인 (ROAS, CVR)"):
        "채널별 ROAS, CVR, CTR 중심으로 분석해드릴게요. 어떤 채널부터 볼까요?",
    ("퍼포먼스 마케터", "매출 원인 파악"):
        "매출에 영향을 주는 핵심 지표를 중심으로 분석해드릴게요. 피처 중요도부터 볼까요?",
    ("퍼포먼스 마케터", "데이터 탐색 / EDA"):
        "데이터 전체 현황을 마케팅 관점으로 살펴드릴게요. 기본 EDA부터 시작할까요?",
    ("퍼포먼스 마케터", "보고서 작성"):
        "핵심 성과 지표와 인사이트를 정리해서 보고서로 만들어드릴게요.",
    ("데이터 분석가 / 데이터 사이언티스트", "광고 성과 확인 (ROAS, CVR)"):
        "채널별 통계 분석과 변수 중요도 기반으로 성과를 분석해드릴게요.",
    ("데이터 분석가 / 데이터 사이언티스트", "매출 원인 파악"):
        "상관분석, 피처 중요도, 회귀 기반으로 매출 드라이버를 분석해드릴게요.",
    ("데이터 분석가 / 데이터 사이언티스트", "데이터 탐색 / EDA"):
        "분포, 결측값, 이상치, 상관관계 전체를 통계적으로 탐색해드릴게요.",
    ("데이터 분석가 / 데이터 사이언티스트", "보고서 작성"):
        "수치 기반 분석 결과와 방법론을 포함한 보고서를 작성해드릴게요.",
    ("기획자 / 전략", "광고 성과 확인 (ROAS, CVR)"):
        "전략적 관점에서 채널 성과를 요약해드릴게요.",
    ("기획자 / 전략", "매출 원인 파악"):
        "매출 구조와 핵심 인사이트를 전략적 시각으로 정리해드릴게요.",
    ("기획자 / 전략", "데이터 탐색 / EDA"):
        "데이터 전체 현황을 전략적 시사점 중심으로 요약해드릴게요.",
    ("기획자 / 전략", "보고서 작성"):
        "임원 보고용 핵심 요약과 액션 아이템 중심으로 정리해드릴게요.",
}


# ── 세션 시작 프로필 수집 ────────────────────────────────────────────

def collect_user_profile() -> dict:
    """
    세션 시작 시 직군 + 분석 목적 수집
    → 이후 응답 스타일과 분석 방향 결정에 사용
    """
    print("\n  분석 시작 전에 두 가지만 여쭤볼게요.\n")

    # Q1. 직군
    print("  Q1. 직군을 선택해주세요:")
    for k, v in ROLES.items():
        print(f"    {k}. {v}")
    print()

    while True:
        role_input = input("  > ").strip()
        if role_input in ROLES:
            role = ROLES[role_input]
            break
        if role_input.lower() in ("exit", "quit", "종료"):
            raise KeyboardInterrupt
        print(f"  1~{len(ROLES)} 중에서 선택해주세요.")

    print()

    # Q2. 분석 목적
    print("  Q2. 오늘 분석 목적을 선택해주세요:")
    for k, v in PURPOSES.items():
        print(f"    {k}. {v}")
    print()

    while True:
        purpose_input = input("  > ").strip()
        if purpose_input in PURPOSES:
            purpose = PURPOSES[purpose_input]
            break
        if purpose_input.lower() in ("exit", "quit", "종료"):
            raise KeyboardInterrupt
        print(f"  1~{len(PURPOSES)} 중에서 선택해주세요.")

    # 맞춤 안내 메시지
    guide = PERSONA_GUIDE.get(
        (role, purpose),
        f"{role} + {purpose} 목적으로 분석을 도와드릴게요."
    )

    print(f"\n  {role} + {purpose}이군요.")
    print(f"  {guide}")

    return {
        "role":    role,
        "purpose": purpose,
        "guide":   guide,
    }


# ── 데이터 로드 ──────────────────────────────────────────────────────

def _make_data_meta(file_path: Path) -> dict:
    import pandas as pd
    encoding = "utf-8"
    for enc in ["utf-8", "utf-8-sig", "euc-kr", "cp949"]:
        try:
            df = pd.read_csv(file_path, nrows=5, encoding=enc)
            encoding = enc
            break
        except Exception:
            continue

    df = pd.read_csv(file_path, nrows=5, encoding=encoding)
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        row_count = max(0, sum(1 for _ in f) - 1)

    return {
        "path":      str(file_path),
        "filename":  file_path.name,
        "encoding":  encoding,
        "row_count": row_count,
        "col_count": len(df.columns),
        "size_mb":   round(file_path.stat().st_size / (1024 * 1024), 2),
        "preview": {
            "columns": list(df.columns),
            "dtypes":  {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample":  df.head(2).to_dict(orient="records"),
        },
    }


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


# ── 스피너 ────────────────────────────────────────────────────────────

class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._stop    = threading.Event()
        self._thread  = None
        self._start_t = None
        self._label   = ""

    def start(self, label: str):
        self._label = label
        self._stop.clear()
        self._start_t = time.time()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_t if self._start_t else 0
        print(f"\r  ✓ {self._label} 완료  ({elapsed:.1f}s)                    ")

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - (self._start_t or 0.0)
            frame   = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r  {frame} {self._label} 중...  ({elapsed:.1f}s)", end="", flush=True)
            i += 1
            time.sleep(0.1)


# ── HITL 처리 ────────────────────────────────────────────────────────

class ExitDuringHITL(Exception):
    """HITL 중 exit 입력 시 발생"""
    pass


def handle_hitl(interrupt_value) -> dict:
    """
    Phase A (free_text): LLM 질문 → 자유 텍스트 답변
    Phase B (selection): 결과 요약 → 승인/수정/재실행 선택
    """
    if not isinstance(interrupt_value, dict):
        interrupt_value = {"input_type": "selection", "message": str(interrupt_value)}

    phase      = interrupt_value.get("phase", "B")
    input_type = interrupt_value.get("input_type", "selection")
    point      = interrupt_value.get("hitl_point", "")
    question   = interrupt_value.get("llm_question", "")
    message    = interrupt_value.get("message", "")
    context    = interrupt_value.get("context", {})
    options    = interrupt_value.get("options", ["승인", "수정", "재실행"])

    print("\n" + "─"*50)

    # Phase A — 자유 텍스트 입력
    if input_type == "free_text":
        print(f"[{point}] Phase A — 요구사항 확인")
        if question:
            print(f"\n  🤖  {question}")
        print(f"\n  (exit 입력 시 세션 종료)")
        print("─"*50)

        user_answer = input("  > ").strip()

        if user_answer.lower() in ("exit", "quit", "종료", "끝"):
            raise ExitDuringHITL()

        print(f"  → 답변 입력됨\n")
        return {"user_answer": user_answer}

    # Phase B — 승인/수정/재실행 선택
    print(f"[{point}] Phase B — 결과 검토")
    if message:
        print(f"\n  {message}")

    user_req = interrupt_value.get("user_answer", "")
    if user_req:
        print(f"\n  반영된 요구사항: {user_req}")

    if context:
        import json
        print("\n  [상세 결과]")
        skip_keys = {"context_summary", "user_requirements", "user_answer"}
        for k, v in context.items():
            if k in skip_keys:
                continue
            v_str = json.dumps(v, ensure_ascii=False)[:150] if isinstance(v, (dict, list)) else str(v)
            print(f"  {k}: {v_str}")

    print(f"\n  선택: {' / '.join(f'{i+1}.{o}' for i, o in enumerate(options))}")
    print(f"  (exit 입력 시 세션 종료)")
    print("─"*50)

    while True:
        choice = input("  > ").strip()

        if choice.lower() in ("exit", "quit", "종료", "끝"):
            raise ExitDuringHITL()

        if choice.isdigit() and 1 <= int(choice) <= len(options):
            choice = options[int(choice) - 1]
        if choice in options:
            break
        print(f"  다시 입력하세요: {options}  (exit: 종료)")

    modified_input = {}
    extra_answer   = ""
    if choice == "수정":
        text = input("  수정 내용: ").strip()
        if text:
            extra_answer   = text
            modified_input = {"user_input": text}

    print(f"  → {choice}\n")
    return {
        "response":       choice,
        "user_answer":    extra_answer,
        "modified_input": modified_input,
    }


# ── 단일 턴 실행 ─────────────────────────────────────────────────────

def build_turn_state(
    user_input: str,
    session_id: str,
    data_meta: dict,
    agent_results: dict,
    user_profile: dict | None = None,
) -> dict:
    """LangGraph 한 턴 입력용 초기 state (CLI / Streamlit 공통)."""
    return {
        "session_id":     session_id,
        "user_input":     user_input,
        "data_meta":      data_meta,
        "agent_results":  agent_results,
        "hitl_history":   [],
        "hitl_required":  False,
        "messages":       [],
        "execution_plan": {},
        "final_response": "",
        "next_agent":     "",
        "current_agent":  "",
        "user_profile":   user_profile or {},
    }


async def graph_step(
    stream_input: Union[dict[str, Any], Command],
    config: dict,
    agent_results: dict,
) -> dict[str, Any]:
    """
    그래프를 한 번 실행해 interrupt 또는 완료까지 진행한다.
    Streamlit 등 UI에서 HITL마다 재호출할 때 사용한다.

    Returns:
        {"status": "interrupt", "interrupt": payload, "agent_results", "final_response"}
        {"status": "complete", "agent_results", "final_response"}
    """
    final_response = ""
    typed_config = cast(RunnableConfig, config)
    typed_stream: Union[GraphState, Command] = (
        cast(GraphState, stream_input) if isinstance(stream_input, dict) else stream_input
    )

    async for event in graph.astream(typed_stream, config=typed_config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupts = event["__interrupt__"]
            interrupt_value = interrupts[0].value if interrupts else {}
            return {
                "status":          "interrupt",
                "interrupt":       interrupt_value,
                "agent_results":   agent_results,
                "final_response":  final_response,
            }

        for node_name, node_output in event.items():
            if node_name.startswith("_"):
                continue
            if isinstance(node_output, dict):
                if node_output.get("final_response"):
                    final_response = node_output["final_response"]
                if node_output.get("agent_results"):
                    agent_results.update(node_output["agent_results"])

    return {
        "status":          "complete",
        "agent_results":   agent_results,
        "final_response":  final_response,
    }


async def run_turn(
    user_input: str,
    session_id: str,
    data_meta: dict,
    config: dict,
    agent_results: dict,
    is_first: bool,
    user_profile: dict | None = None,
) -> dict:
    """
    사용자 메시지 한 턴 처리
    HITL interrupt 발생 시 처리 후 재개
    Returns: 업데이트된 agent_results
    """
    spinner = Spinner()

    state = build_turn_state(
        user_input=user_input,
        session_id=session_id,
        data_meta=data_meta,
        agent_results=agent_results,
        user_profile=user_profile,
    )

    stream_input = state
    final_response = ""

    NODE_LABELS = {
        "orchestrator":         "의도 파악",
        "orchestrator_respond": "응답 생성",
        "fe_agent":             "FE 파이프라인",
        "sql_agent":            "SQL 분석",
        "insight_agent":        "데이터 분석",
        "report_agent":         "보고서 생성",
        "hitl_plan":            "계획 확인 대기",
        "hitl_preprocess":      "전처리 확인 대기",
        "hitl_analysis":        "분석 확인 대기",
        "hitl_final":           "최종 확인 대기",
    }

    current_node = ""

    while True:
        interrupted     = False
        interrupt_value = None

        typed_input  = cast(GraphState, stream_input) if isinstance(stream_input, dict) else stream_input
        typed_config = cast(RunnableConfig, config)
        async for event in graph.astream(typed_input, config=typed_config, stream_mode="updates"):
            if "__interrupt__" in event:
                if current_node:
                    spinner.stop()
                    current_node = ""
                interrupts      = event["__interrupt__"]
                interrupt_value = interrupts[0].value if interrupts else {}
                interrupted     = True
                break

            for node_name, node_output in event.items():
                if node_name.startswith("_"):
                    continue

                # 스피너 전환
                if node_name != current_node:
                    if current_node:
                        spinner.stop()
                    current_node = node_name
                    label = NODE_LABELS.get(node_name, node_name)
                    if node_name not in ("hitl_plan", "hitl_preprocess", "hitl_analysis", "hitl_final"):
                        spinner.start(label)

                # 응답 수집
                if isinstance(node_output, dict):
                    if node_output.get("final_response"):
                        final_response = node_output["final_response"]
                    if node_output.get("agent_results"):
                        agent_results = node_output["agent_results"]

        if not interrupted and current_node:
            spinner.stop()
            current_node = ""

        if interrupted and interrupt_value is not None:
            try:
                hitl_response = handle_hitl(interrupt_value)
            except ExitDuringHITL:
                print("\n  세션 종료.\n")
                raise
            stream_input  = Command(resume=hitl_response)
            current_node  = ""
            continue

        break

    # 최종 응답 출력
    if final_response:
        print(f"\n🤖  {final_response}\n")

    # 차트 자동 열기 (macOS)
    _auto_open_charts(agent_results)

    return agent_results


def _auto_open_charts(agent_results: dict) -> None:
    """생성된 차트 파일 자동으로 열기 (macOS: open 명령어)"""
    import subprocess
    import platform

    ag04 = agent_results.get("AG-04", {})
    image_paths = ag04.get("image_paths", [])

    if not image_paths:
        return

    print(f"  📊  차트 {len(image_paths)}개 생성됨:")
    for path in image_paths:
        p = Path(path)
        if p.exists():
            print(f"     {p.name}")
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(p)], check=False)

    # 보고서도 자동 열기
    ag05 = agent_results.get("AG-05", {})
    if report_path := ag05.get("report_path", ""):
        rp = Path(report_path)
        if rp.exists():
            print(f"  📄  보고서: {rp.name}")
            if platform.system() == "Darwin":
                subprocess.run(["open", str(rp)], check=False)
    print()


# ── 메인 대화 루프 ───────────────────────────────────────────────────

def _describe_data(data_meta: dict) -> str:
    """
    데이터 한 문장 설명 (LLM 없이 규칙 기반)
    UI/CLI 공통으로 사용
    """
    rows     = data_meta["row_count"]
    cols     = data_meta["col_count"]
    columns  = data_meta["preview"]["columns"]

    col_lower = [c.lower() for c in columns]

    # 날짜·채널·타겟 컬럼 감지
    has_date    = any(c in col_lower for c in ("date", "날짜", "일자", "time", "datetime"))
    has_channel = "channel" in col_lower
    has_target  = "target" in col_lower

    desc = f"{rows}행 {cols}개 컬럼의 데이터예요."
    if has_date and has_channel:
        desc += " 날짜별 광고 채널 성과 데이터를 포함하고 있어요."
    elif has_date:
        desc += " 날짜별 시계열 데이터를 포함하고 있어요."
    elif has_channel:
        desc += " 광고 채널별 성과 지표가 있어요."
    if has_target:
        desc += " 분석 타겟(TARGET)은 매출 컬럼이에요."

    return desc


async def chat(file_path: Path) -> None:
    """
    메인 대화 루프
    CLI: python main.py --file data.csv
    UI:  Streamlit에서 file_path 전달 후 asyncio.run(chat(path)) 호출
    """
    if not file_path.exists():
        print(f"\n[오류] 파일 없음: {file_path}")
        print(f"  data/sample/ 또는 data/uploads/ 폴더를 확인해주세요.\n")
        return

    session_id    = make_session_id()
    config        = {"configurable": {"thread_id": session_id}}
    data_meta     = _make_data_meta(file_path)
    agent_results = {}

    # 1) 헤더 출력
    print(f"\n{'='*55}")
    print(f"  E_LENS Ecommerce Analysis Agent")
    print(f"{'='*55}")

    # 2) 데이터 설명 (업로드 완료 시점)
    data_desc = _describe_data(data_meta)
    print(f"\n  📂  {file_path.name} 업로드 완료")
    print(f"  🤖  {data_desc}\n")

    # 3) 사용자 프로필 수집 (직군 + 분석 목적)
    try:
        user_profile = collect_user_profile()
    except KeyboardInterrupt:
        print("\n  세션 종료.\n")
        return

    # 4) 대화 시작
    print(f"\n{'─'*55}")
    print(f"  무엇을 분석할까요? (종료: exit)\n")

    turn = 0
    while True:
        try:
            user_input = input("👤  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  세션 종료.\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "종료", "끝"):
            print(f"\n  분석 완료. 세션: {session_id}\n")
            break

        turn += 1
        print()
        try:
            agent_results = await run_turn(
                user_input=user_input,
                session_id=session_id,
                data_meta=data_meta,
                config=config,
                agent_results=agent_results,
                is_first=(turn == 1),
                user_profile=user_profile,
            )
        except ExitDuringHITL:
            break


# ── UI 연동용 업로드 헬퍼 ───────────────────────────────────────────
# Streamlit에서 이 함수로 파일 저장 후 chat() 호출

def save_upload(file_bytes: bytes, filename: str) -> Path:
    """
    업로드된 파일을 data/uploads/에 저장하고 경로 반환
    Streamlit 연동 시 사용:
        path = save_upload(uploaded_file.read(), uploaded_file.name)
        asyncio.run(chat(path))
    """
    import uuid
    from config import UPLOAD_DATA_DIR
    UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    save_path = UPLOAD_DATA_DIR / save_name
    save_path.write_bytes(file_bytes)
    return save_path


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E_LENS Ecommerce Analysis Agent")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="분석할 CSV 파일 경로 (기본값: data/sample/ 첫 번째 파일)",
    )
    args = parser.parse_args()

    if args.file:
        fp = Path(args.file)
    else:
        csvs = list(SAMPLE_DATA_DIR.glob("*.csv"))
        if not csvs:
            print("\n[오류] data/sample/ 에 CSV 파일이 없습니다.")
            sys.exit(1)
        fp = csvs[0]

    asyncio.run(chat(fp))