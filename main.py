"""
main.py
DAISY Feature Engineering Agent 진입점

실행 방법:
  python main.py --mode insight  # AG-02 건너뛰고 AG-04~05만 테스트 (기본값)
  python main.py --mode test     # EDA만 빠르게
  python main.py --mode full     # 전체 파이프라인
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import OUTPUT_DIR, DATA_DIR
from graph import graph
from langgraph.types import Command


# ── 샘플 데이터 ──────────────────────────────────────────────────────
SAMPLE_CSV  = DATA_DIR / "pivoted_data_sample.csv"
SAMPLE_META = {
    "path":      str(SAMPLE_CSV),
    "filename":  "pivoted_data_sample.csv",
    "encoding":  "utf-8",
    "row_count": 100,
    "col_count": 50,
    "size_mb":   0.5,
    "preview": {
        "columns": ["TARGET", "feature_1", "feature_2"],
        "dtypes":  {"TARGET": "float64", "feature_1": "float64"},
        "sample":  [],
    },
}

AG02_SKIP_RESULT = {
    "AG-02": {
        "output_path": str(OUTPUT_DIR),
        "output_dir":  str(OUTPUT_DIR),
        "mode":        "skipped",
        "stages_done": [],
    }
}


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def make_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def make_initial_state(mode: str, session_id: str) -> dict:
    prompts = {
        "full":    "샘플 데이터로 전체 분석 파이프라인 실행해줘.",
        "insight": "샘플 데이터로 EDA, 변수 중요도 분석, 인사이트 도출하고 PDF 보고서 만들어줘.",
        "test":    "데이터 EDA만 빠르게 해줘.",
    }

    base = {
        "session_id":     session_id,
        "user_input":     prompts.get(mode, prompts["insight"]),
        "data_meta":      SAMPLE_META,
        "agent_results":  {},
        "hitl_history":   [],
        "hitl_required":  False,
        "messages":       [],
        "execution_plan": {},
        "final_response": "",
    }

    if mode in ("insight", "test"):
        base["agent_results"] = AG02_SKIP_RESULT
        base["execution_plan"] = {
            "stages":      ["AG-04", "AG-05"],
            "params": {
                "AG-04": {"top_n": 5, "target_col": "TARGET"},
                "AG-05": {"format": "pdf"},
            },
            "description": "EDA + 인사이트 + 보고서 생성 (AG-02 건너뜀)",
        }

    return base


# ── HITL 처리 ────────────────────────────────────────────────────────

def handle_hitl(interrupt_value) -> dict:
    """interrupt() 값에서 메시지 추출 후 터미널 입력 받기"""
    # interrupt_value는 hitl_interrupt()에서 넘긴 payload dict
    if isinstance(interrupt_value, dict):
        message = interrupt_value.get("message", "확인해주세요.")
        options = interrupt_value.get("options", ["승인", "수정", "재실행"])
        context = interrupt_value.get("context", {})
        point   = interrupt_value.get("hitl_point", "")
    else:
        message = str(interrupt_value)
        options = ["승인", "수정", "재실행"]
        context = {}
        point   = ""

    print("\n" + "="*60)
    print(f"[HITL] {point}")
    print(f"  {message}")
    if context:
        import json
        print("\n  [컨텍스트]")
        for k, v in context.items():
            v_str = json.dumps(v, ensure_ascii=False)[:200] if isinstance(v, (dict, list)) else str(v)
            print(f"    {k}: {v_str}")
    print(f"\n  선택지: {' / '.join(f'{i+1}.{o}' for i, o in enumerate(options))}")
    print("="*60)

    while True:
        choice = input("  선택: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            choice = options[int(choice) - 1]
        if choice in options:
            break
        print(f"  다시 선택하세요: {options}")

    modified_input = {}
    if choice == "수정":
        print("\n  수정 내용 입력:")
        text = input("  > ").strip()
        if text:
            modified_input = {"user_input": text}

    print(f"\n  → '{choice}' 선택됨\n")
    return {"response": choice, "modified_input": modified_input}


# ── 메인 실행 ────────────────────────────────────────────────────────

async def run(mode: str = "insight") -> None:
    session_id    = make_session_id()
    config        = make_config(session_id)
    initial_state = make_initial_state(mode, session_id)

    print(f"\n{'='*60}")
    print(f"  DAISY Agent  |  mode: {mode}  |  session: {session_id}")
    print(f"  요청: {initial_state['user_input']}")
    if mode in ("insight", "test"):
        print(f"  [AG-02 건너뜀]")
    print(f"{'='*60}\n")

    # 첫 실행은 initial_state, 이후 HITL 재개는 Command(resume=...)
    stream_input = initial_state

    while True:
        interrupted    = False
        interrupt_value = None

        async for event in graph.astream(
            stream_input,
            config=config,
            stream_mode="values",
        ):
            # 노드 실행 상태 출력
            if "__interrupt__" in event:
                # interrupt() 발생 — 값 추출
                interrupts = event["__interrupt__"]
                interrupt_value = interrupts[0].value if interrupts else {}
                interrupted = True
                break

            # 진행 상황 출력
            for k in ["execution_plan", "agent_results", "final_response"]:
                v = event.get(k)
                if not v:
                    continue
                if k == "execution_plan" and isinstance(v, dict) and v.get("stages"):
                    print(f"  [계획] stages={v['stages']}")
                if k == "agent_results" and isinstance(v, dict):
                    for ag, result in v.items():
                        if isinstance(result, dict) and "error" not in result:
                            print(f"  ✓ {ag} 완료")
                if k == "final_response" and v:
                    print(f"  ✓ 최종 응답 생성됨")

        if interrupted and interrupt_value is not None:
            hitl_response = handle_hitl(interrupt_value)
            # Command(resume=...) 으로 재개
            stream_input = Command(resume=hitl_response)
            continue

        # 그래프 완료
        break

    # ── 결과 출력 ────────────────────────────────────────────────────
    final        = graph.get_state(config)
    state_values = final.values if hasattr(final, "values") else {}

    print(f"\n{'='*60}")
    print("  분석 완료")
    print(f"{'='*60}")

    if resp := state_values.get("final_response", ""):
        print(f"\n[응답]\n{resp}")

    if ag05 := state_values.get("agent_results", {}).get("AG-05", {}):
        report_path = ag05.get("report_path", "")
        print(f"\n[보고서] {report_path if report_path else '생성 실패'}")

    print(f"\n[로그]  logs/sessions/{session_id}/trace.md")
    print(f"[DB]    data/agent_trace.db\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAISY Agent")
    parser.add_argument(
        "--mode",
        choices=["full", "insight", "test"],
        default="insight",
    )
    args = parser.parse_args()
    asyncio.run(run(mode=args.mode))