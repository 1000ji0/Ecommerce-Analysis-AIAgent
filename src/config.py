"""
전역 설정
모든 파일이 from config import ... 로 가져다 씀
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

try:
    import streamlit as st
except Exception:
    st = None


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Streamlit secrets -> environment variable -> default 순서로 값 조회."""
    if st is not None:
        try:
            if hasattr(st, "secrets") and key in st.secrets:
                value = st.secrets[key]
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
        except Exception:
            pass
    return os.getenv(key, default)

# ── 프로젝트 경로 ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent   # agent_dev/
SRC_DIR    = ROOT_DIR / "src"
MCP_DIR    = ROOT_DIR / "mcp_claude" / "integrated"
MCP_DATA_DIR = ROOT_DIR / "mcp_claude" / "data"       # MCP 원본 데이터
OUTPUT_DIR   = MCP_DATA_DIR / "output"                 # MCP 기본 output (참조용)

# ── 사용자 데이터 관리 폴더 ──────────────────────────────────────────
USER_DATA_DIR   = ROOT_DIR / "data"
SAMPLE_DATA_DIR = ROOT_DIR / "data" / "sample"
UPLOAD_DATA_DIR = ROOT_DIR / "data" / "uploads"

SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 하위 호환성 유지 ─────────────────────────────────────────────────
# 기존 코드에서 DATA_DIR을 쓰는 곳이 있으면 SAMPLE_DATA_DIR로 연결
DATA_DIR = SAMPLE_DATA_DIR

# ── 세션 디렉토리 ────────────────────────────────────────────────────
SESSION_DIR = ROOT_DIR / "sessions"
SESSION_DIR.mkdir(exist_ok=True)


def get_session_output_dir(session_id: str) -> Path:
    """
    세션별 독립 output 디렉토리 반환
    각 세션의 FE 파이프라인 결과가 서로 섞이지 않도록 분리

    구조:
        sessions/{session_id}/
            ├── output/    ← FE 파이프라인 결과 (pickle 등)
            ├── charts/    ← 시각화 이미지
            ├── reports/   ← PDF/CSV 보고서
            ├── cache/     ← 피처 캐시
            └── trace.md   ← 실행 로그
    """
    path = SESSION_DIR / session_id / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── LLM 설정 ─────────────────────────────────────────────────────────
GOOGLE_API_KEY = _get_secret_or_env("GOOGLE_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

# AG-03 SQL 에이전트용 비즈니스 DB (오케스트레이터 플랜에 db_url이 없을 때 대체)
# 예: sqlite:////absolute/path/to/app.db  또는 mysql+pymysql://user:pass@host/db
AGENT_BUSINESS_DB_URL = _get_secret_or_env("AGENT_BUSINESS_DB_URL", "").strip()

# ── MCP 서버 ─────────────────────────────────────────────────────────
MCP_SERVER_URL = _get_secret_or_env("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")

# ── 데이터 기본값 ────────────────────────────────────────────────────
DEFAULT_TARGET_COL = "TARGET"


def _to_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


# ── LLM 호출 제한 ────────────────────────────────────────────────────
LLM_SESSION_CALL_LIMIT = _to_int(_get_secret_or_env("LLM_SESSION_CALL_LIMIT", "30"), 30)
LLM_DAILY_CALL_LIMIT   = _to_int(_get_secret_or_env("LLM_DAILY_CALL_LIMIT", "120"), 120)

# ── LangSmith ────────────────────────────────────────────────────────
os.environ["LANGCHAIN_API_KEY"]    = _get_secret_or_env("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = _get_secret_or_env("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"]    = _get_secret_or_env("LANGCHAIN_PROJECT", "ecommerce-analysis-agent")