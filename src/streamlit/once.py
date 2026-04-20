"""
DAISY 이커머스 분석 에이전트 — Streamlit UI

실행 (저장소 루트에서):
  uv run streamlit run src/streamlit/once.py
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine, text
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import AGENT_BUSINESS_DB_URL, GOOGLE_API_KEY, SAMPLE_DATA_DIR  # noqa: E402
from tools.database.sqlite_store import TraceStore  # noqa: E402
from main import (  # noqa: E402
    PERSONA_GUIDE,
    PURPOSES,
    ROLES,
    _make_data_meta,
    build_turn_state,
    graph_step,
    make_session_id,
    save_upload,
)


def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "session_id": None,
        "graph_config": None,
        "data_meta": None,
        "user_profile": None,
        "agent_results": {},
        "pending_hitl": False,
        "hitl_value": None,
        "last_user_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _persona_guide(role: str, purpose: str) -> str:
    return PERSONA_GUIDE.get(
        (role, purpose),
        f"{role} + {purpose} 목적으로 분석을 도와드릴게요.",
    )


def _start_session(uploaded_path: Path, role: str, purpose: str) -> None:
    sid = make_session_id()
    st.session_state.session_id = sid
    st.session_state.graph_config = {"configurable": {"thread_id": sid}}
    st.session_state.data_meta = _make_data_meta(uploaded_path)
    st.session_state.user_profile = {
        "role": role,
        "purpose": purpose,
        "guide": _persona_guide(role, purpose),
    }
    st.session_state.agent_results = {}
    st.session_state.messages = []
    st.session_state.pending_hitl = False
    st.session_state.hitl_value = None
    st.session_state.last_user_prompt = None


def _reset_session() -> None:
    st.session_state.session_id = None
    st.session_state.graph_config = None
    st.session_state.data_meta = None
    st.session_state.user_profile = None
    st.session_state.agent_results = {}
    st.session_state.messages = []
    st.session_state.pending_hitl = False
    st.session_state.hitl_value = None
    st.session_state.last_user_prompt = None


def _session_ready() -> bool:
    return bool(
        st.session_state.session_id
        and st.session_state.graph_config
        and st.session_state.data_meta
        and st.session_state.user_profile
    )


def _collect_chart_paths_from_state() -> list[str]:
    """AG-04 결과에 있는 PNG 경로 중 실제 파일만."""
    ag04 = st.session_state.agent_results.get("AG-04", {})
    out: list[str] = []
    for p in ag04.get("image_paths", []) or []:
        pp = Path(str(p))
        if pp.is_file():
            out.append(str(pp.resolve()))
    return out


def _append_assistant_message(text: str) -> None:
    st.session_state.messages.append(
        {
            "role":         "assistant",
            "content":      text,
            "image_paths":  _collect_chart_paths_from_state(),
        }
    )


def _render_db_status() -> None:
    """트레이스 SQLite + AG-03용 비즈니스 DB URL 점검."""
    with st.expander("DB 연결 상태", expanded=False):
        store = TraceStore()
        tpath = store.db_path
        st.caption(f"트레이스·세션 로그 (SQLite): `{tpath}`")
        try:
            with sqlite3.connect(tpath) as conn:
                conn.execute("SELECT 1")
                n_tables = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
            st.success(f"트레이스 DB 연결 OK · 테이블 {n_tables}개")
        except Exception as e:
            st.error(f"트레이스 DB 오류: {e}")

        st.divider()
        if AGENT_BUSINESS_DB_URL:
            st.caption("AG-03 SQL 에이전트용 `AGENT_BUSINESS_DB_URL`")
            try:
                eng = create_engine(AGENT_BUSINESS_DB_URL)
                with eng.connect() as c:
                    c.execute(text("SELECT 1"))
                st.success("비즈니스 DB (AG-03) 연결 OK")
            except Exception as e:
                st.error(f"비즈니스 DB 연결 실패: {e}")
        else:
            st.info(
                "`AGENT_BUSINESS_DB_URL`이 비어 있습니다. "
                "AG-03은 의도 파싱 결과의 `db_url` 또는 이 환경변수가 있어야 "
                "SQL 조회가 동작합니다. (트레이스 DB와는 별개입니다.)"
            )


def _render_hitl() -> None:
    iv = st.session_state.hitl_value or {}
    input_type = iv.get("input_type", "selection")
    question = iv.get("llm_question", "")
    message = iv.get("message", "")
    options = iv.get("options", ["승인", "수정", "재실행"])
    point = iv.get("hitl_point", "")

    st.markdown("#### 에이전트 확인 단계")
    if point:
        st.caption(point)

    with st.form("hitl_form"):
        free_text_answer = ""
        choice = options[0] if options else "승인"
        modify_text = ""
        if input_type == "free_text":
            st.markdown(question or "추가 요구사항을 입력해주세요.")
            free_text_answer = st.text_area(
                "답변",
                key="hitl_free_text",
                height=120,
                label_visibility="collapsed",
            )
        else:
            st.markdown(message or "결과를 검토해주세요.")
            choice = st.radio("선택", options, horizontal=True, key="hitl_choice")
            if choice == "수정":
                modify_text = st.text_input("수정 내용", key="hitl_modify")

        submitted = st.form_submit_button("전달")

    if not submitted:
        return

    if input_type == "free_text":
        if not free_text_answer.strip():
            st.warning("답변을 입력해주세요.")
            return
        resume: dict = {"user_answer": free_text_answer.strip()}
    else:
        resume = {"response": choice, "user_answer": "", "modified_input": {}}
        if choice == "수정" and modify_text.strip():
            resume["user_answer"] = modify_text.strip()
            resume["modified_input"] = {"user_input": modify_text.strip()}

    with st.spinner("에이전트 실행 중..."):
        result = asyncio.run(
            graph_step(
                Command(resume=resume),
                st.session_state.graph_config,
                st.session_state.agent_results,
            )
        )
    st.session_state.agent_results = result["agent_results"]
    if result["status"] == "interrupt":
        st.session_state.hitl_value = result["interrupt"]
        st.session_state.pending_hitl = True
    else:
        st.session_state.pending_hitl = False
        st.session_state.hitl_value = None
        text = (result.get("final_response") or "").strip() or "처리가 완료되었습니다."
        _append_assistant_message(text)
    st.rerun()


def _run_user_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("에이전트 실행 중..."):
        result = asyncio.run(
            graph_step(
                build_turn_state(
                    user_input=prompt,
                    session_id=st.session_state.session_id,
                    data_meta=st.session_state.data_meta,
                    agent_results=st.session_state.agent_results,
                    user_profile=st.session_state.user_profile,
                ),
                st.session_state.graph_config,
                st.session_state.agent_results,
            )
        )
    st.session_state.agent_results = result["agent_results"]
    if result["status"] == "interrupt":
        st.session_state.hitl_value = result["interrupt"]
        st.session_state.pending_hitl = True
    else:
        st.session_state.pending_hitl = False
        st.session_state.hitl_value = None
        text = (result.get("final_response") or "").strip() or "처리가 완료되었습니다."
        _append_assistant_message(text)


def main() -> None:
    _init_session_state()
    st.set_page_config(page_title="DAISY Agent", layout="wide")

    if not GOOGLE_API_KEY:
        st.error(
            "환경 변수 `GOOGLE_API_KEY`가 설정되어 있지 않습니다. "
            "프로젝트 루트에 `.env`를 두거나 셸에서 export 해주세요."
        )

    with st.sidebar:
        st.header("세션")
        sample_files = sorted(SAMPLE_DATA_DIR.glob("*.csv"))
        default_sample = bool(sample_files) and not _session_ready()
        use_sample = st.checkbox("샘플 CSV 사용", value=default_sample, disabled=not sample_files)
        if not sample_files:
            st.caption("`data/sample/`에 CSV가 없으면 업로드만 사용할 수 있습니다.")
        uploaded = None
        if not use_sample:
            uploaded = st.file_uploader("CSV 업로드", type=["csv"])

        st.subheader("프로필")
        role = st.selectbox("직군", list(ROLES.values()), key="profile_role")
        purpose = st.selectbox("분석 목적", list(PURPOSES.values()), key="profile_purpose")

        if st.button("이 설정으로 분석 시작", type="primary", disabled=not GOOGLE_API_KEY):
            path: Path | None = None
            if use_sample and sample_files:
                path = sample_files[0]
            elif uploaded is not None:
                path = save_upload(uploaded.getvalue(), uploaded.name)
            if path is None:
                st.warning("CSV 파일을 선택하거나 샘플 사용을 켜주세요.")
            else:
                _start_session(path, role, purpose)
                st.success(f"세션 시작: `{st.session_state.session_id}`")
                st.rerun()

        if st.button("세션 초기화"):
            _reset_session()
            st.rerun()

        if _session_ready():
            dm = st.session_state.data_meta
            st.caption(f"파일: {dm.get('filename', '')} · {dm.get('row_count', '?')}행")

        _render_db_status()

    st.title("DAISY 이커머스 분석 에이전트")

    if not _session_ready():
        st.info("사이드바에서 CSV와 프로필을 선택한 뒤 **분석 시작**을 눌러주세요.")
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for ip in m.get("image_paths") or []:
                pr = Path(str(ip))
                if pr.is_file():
                    st.image(str(pr), caption=pr.name)
                elif ip:
                    st.caption(f"차트 파일 없음: `{ip}`")

    if st.session_state.pending_hitl:
        _render_hitl()

    if st.session_state.pending_hitl:
        if prompt := st.chat_input("질문… (HITL 완료 후 입력 가능)"):
            st.warning("위 확인 단계를 먼저 완료해주세요.")
    else:
        if prompt := st.chat_input("무엇을 분석할까요?"):
            _run_user_prompt(prompt)
            st.rerun()


# Streamlit 실행 시에도 스크립트가 매 rerun마다 로드되므로 항상 진입한다.
main()
