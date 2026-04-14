"""
전역 설정
모든 파일이 from config import ... 로 가져다 씀
경로·모델명·API키 등 한 곳에서 관리
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

# ── 프로젝트 경로 ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent   # agent_dev/
SRC_DIR    = ROOT_DIR / "src"                          # agent_dev/src/
MCP_DIR    = ROOT_DIR / "mcp_claude" / "integrated"   # MCP 서버 경로
DATA_DIR   = MCP_DIR / "data"                          # MCP 원본 데이터
OUTPUT_DIR = DATA_DIR / "output"                       # MCP 파이프라인 출력

# ── 세션 디렉토리 ────────────────────────────────────────────────────
# 세션별 로그·캐시·차트·보고서 저장
SESSION_DIR = ROOT_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)

# ── LLM 설정 ─────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

# ── MCP 서버 ─────────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

# ── 데이터 기본값 ────────────────────────────────────────────────────
DEFAULT_TARGET_COL = "TARGET"