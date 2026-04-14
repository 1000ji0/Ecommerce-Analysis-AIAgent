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

from config import SAMPLE_DATA_DIR, get_session_output_dir
from graph import graph
from langgraph.types import Command


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
            elapsed = time.time() - self._start_t
            frame   = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r  {frame} {self._label} 중...  ({elapsed:.1f}s)", end="", flush=True)
            i += 1
            time.sleep(0.1)


# ── HITL 처리 ────────────────────────────────────────────────────────

def handle_hitl(interrupt_value) -> dict:
    if isinstance(interrupt_value, dict):
        message = interrupt_value.get("message", "확인해주세요.")
        options = interrupt_value.get("options", ["승인", "수정", "재실행"])
        context = interrupt_value.get("context", {})
        point   = interrupt_value.get("hitl_point", "")
    else:
        message, options, context, point = str(interrupt_value), ["승인", "수정", "재실행"], {}, ""

    print("\n" + "─"*50)
    print(f"[{point}]  {message}")
    if context:
        import json
        for k, v in context.items():
            v_str = json.dumps(v, ensure_ascii=False)[:150] if isinstance(v, (dict, list)) else str(v)
            print(f"  {k}: {v_str}")
    print(f"  선택: {' / '.join(f'{i+1}.{o}' for i, o in enumerate(options))}")
    print("─"*50)

    while True:
        choice = input("  > ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            choice = options[int(choice) - 1]
        if choice in options:
            break
        print(f"  다시 입력하세요: {options}")

    modified_input = {}
    if choice == "수정":
        text = input("  수정 내용: ").strip()
        if text:
            modified_input = {"user_input": text}

    print(f"  → {choice}\n")
    return {"response": choice, "modified_input": modified_input}


# ── 단일 턴 실행 ─────────────────────────────────────────────────────

async def run_turn(
    user_input: str,
    session_id: str,
    data_meta: dict,
    config: dict,
    agent_results: dict,
    is_first: bool,
) -> dict:
    """
    사용자 메시지 한 턴 처리
    HITL interrupt 발생 시 처리 후 재개
    Returns: 업데이트된 agent_results
    """
    spinner = Spinner()

    state = {
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
    }

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

        async for event in graph.astream(stream_input, config=config, stream_mode="updates"):
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
            hitl_response = handle_hitl(interrupt_value)
            stream_input  = Command(resume=hitl_response)
            current_node  = ""
            continue

        break

    # 최종 응답 출력
    if final_response:
        print(f"\n🤖  {final_response}\n")

    return agent_results


# ── 메인 대화 루프 ───────────────────────────────────────────────────

async def chat(file_path: Path) -> None:
    if not file_path.exists():
        print(f"\n[오류] 파일 없음: {file_path}")
        print(f"  data/sample/ 폴더에 CSV 파일을 넣어주세요.\n")
        return

    session_id    = make_session_id()
    config        = {"configurable": {"thread_id": session_id}}
    data_meta     = _make_data_meta(file_path)
    agent_results = {}

    print(f"\n{'='*55}")
    print(f"  DAISY Ecommerce Analysis Agent")
    print(f"  파일  : {file_path.name}")
    print(f"  컬럼  : {data_meta['row_count']}행 × {data_meta['col_count']}컬럼")
    print(f"  컬럼명: {data_meta['preview']['columns']}")
    print(f"{'='*55}")
    print(f"  무엇을 분석할까요? (종료: exit)")
    print(f"  예시: 피처 중요도 뽑아줘 / EDA 해줘 / 전체 분석해줘\n")

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
        agent_results = await run_turn(
            user_input=user_input,
            session_id=session_id,
            data_meta=data_meta,
            config=config,
            agent_results=agent_results,
            is_first=(turn == 1),
        )


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAISY Ecommerce Analysis Agent")
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