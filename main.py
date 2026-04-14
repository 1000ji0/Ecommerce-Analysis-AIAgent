"""
main.py
DAISY Feature Engineering Agent 진입점
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


# ── 스피너 ────────────────────────────────────────────────────────────

class Spinner:
    """추론 중 표시 + 경과 시간 스피너"""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "추론 중"):
        self.label     = label
        self._stop     = threading.Event()
        self._thread   = None
        self._start_t  = None

    def start(self, label: str | None = None):
        if label:
            self.label = label
        self._stop.clear()
        self._start_t = time.time()
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, result_label: str = "완료"):
        self._stop.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_t if self._start_t else 0
        # 스피너 줄 지우고 완료 메시지 출력
        print(f"\r  ✓ {result_label}  ({elapsed:.1f}s)          ")

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._start_t
            frame   = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r  {frame} {self.label}  ({elapsed:.1f}s) ...", end="", flush=True)
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


# ── 노드 이름 → 한국어 ───────────────────────────────────────────────
NODE_LABELS = {
    "orchestrator":      "AG-01  분석 계획 수립",
    "orchestrator_post": "AG-01  HITL 후 처리",
    "fe_agent":          "AG-02  Feature Engineering",
    "sql_agent":         "AG-03  SQL 분석",
    "insight_agent":     "AG-04  EDA & 인사이트 분석",
    "report_agent":      "AG-05  보고서 생성",
    "hitl_plan":         "HITL ①  계획 승인 대기",
    "hitl_preprocess":   "HITL ②  전처리 확인 대기",
    "hitl_analysis":     "HITL ③  분석 결과 확인 대기",
    "hitl_final":        "HITL ④  최종 승인 대기",
}


# ── 메인 실행 ────────────────────────────────────────────────────────

async def run(mode: str = "insight") -> None:
    session_id    = make_session_id()
    config        = make_config(session_id)
    initial_state = make_initial_state(mode, session_id)
    total_start   = time.time()

    print(f"\n{'='*60}")
    print(f"  DAISY Ecommerce Analysis Agent")
    print(f"  세션  : {session_id}")
    print(f"  모드  : {mode}")
    print(f"  요청  : {initial_state['user_input']}")
    if mode in ("insight", "test"):
        print(f"  [AG-02 건너뜀 — MCP 서버 불필요]")
    print(f"{'='*60}\n")

    spinner      = Spinner()
    stream_input = initial_state
    current_node = ""

    while True:
        interrupted     = False
        interrupt_value = None

        async for event in graph.astream(
            stream_input,
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in event:
                spinner.stop(f"{NODE_LABELS.get(current_node, current_node)} 완료") if current_node else None
                interrupts      = event["__interrupt__"]
                interrupt_value = interrupts[0].value if interrupts else {}
                interrupted     = True
                break

            # 노드 실행 감지
            for node_name in event:
                if node_name.startswith("_"):
                    continue
                if node_name != current_node:
                    # 이전 노드 완료 표시
                    if current_node:
                        spinner.stop(f"{NODE_LABELS.get(current_node, current_node)} 완료")
                    current_node = node_name
                    label = NODE_LABELS.get(node_name, node_name)
                    spinner.start(label)

        if not interrupted and current_node:
            spinner.stop(f"{NODE_LABELS.get(current_node, current_node)} 완료")
            current_node = ""

        if interrupted and interrupt_value is not None:
            hitl_response = handle_hitl(interrupt_value)
            stream_input  = Command(resume=hitl_response)
            current_node  = ""
            continue

        break

    # ── 결과 출력 ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    final         = graph.get_state(config)
    state_values  = final.values if hasattr(final, "values") else {}

    print(f"\n{'='*60}")
    print(f"  분석 완료  (총 소요시간: {total_elapsed:.1f}s)")
    print(f"{'='*60}")

    if resp := state_values.get("final_response", ""):
        print(f"\n[응답]\n{resp}")

    if ag05 := state_values.get("agent_results", {}).get("AG-05", {}):
        report_path = ag05.get("report_path", "")
        print(f"\n[보고서] {report_path if report_path else '생성 실패'}")

    print(f"\n[LangSmith] https://smith.langchain.com")
    print(f"[로그]  logs/sessions/{session_id}/trace.md")
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