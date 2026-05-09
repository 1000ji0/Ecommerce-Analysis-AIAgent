"""Simple test runner for local SLLM /chat endpoint using src/sllm_client.py

Run from repository root:
python3 scripts/test_sllm.py
"""
from __future__ import annotations
import sys
from pathlib import Path
# ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sllm_client import query_sllm

if __name__ == "__main__":
    resp = query_sllm("테스트 질문: 안녕하세요? SLLM이 잘 연결되나요?", session_id="script-test")
    print(resp)
