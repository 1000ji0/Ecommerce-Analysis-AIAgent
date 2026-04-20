"""
E_LENS 이커머스 분석 에이전트 — Streamlit UI
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st
from langgraph.types import Command

# Streamlit Cloud / 로컬 모두 대응
_HERE = Path(__file__).resolve()
for _candidate in [
    _HERE.parents[2],           # root
    _HERE.parents[2] / "src",   # src
    _HERE.parents[1],           # src (로컬)
    _HERE.parent,               # streamlit
]:
    _p = str(_candidate)
    if _candidate.exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from config import GOOGLE_API_KEY, SAMPLE_DATA_DIR
from auth.auth import ensure_db, is_logged_in, get_current_user, logout, render_login_page
from auth.auth_db import save_session, update_session, get_sessions_by_user
from main import (
    PERSONA_GUIDE, PURPOSES, ROLES,
    _make_data_meta, build_turn_state, graph_step,
    make_session_id, save_upload,
)

HITL_LEVELS = {
    0: ("완전 수동",      "모든 단계 직접 확인"),
    1: ("보조 자동화",    "주요 결정 시 확인"),
    2: ("부분 자동화",    "전체 분석 시에만  ★"),
    3: ("조건부 자동화",  "문제 발생 시만"),
    4: ("완전 자동화",    "에이전트 자율 실행"),
}

ROLE_ICONS = {
    "퍼포먼스 마케터":                  "📢",
    "데이터 분석가 / 데이터 사이언티스트": "📊",
    "기획자 / 전략":                    "🗺️",
    "기타":                            "👤",
}

PURPOSE_ICONS = {
    "광고 성과 확인 (ROAS, CVR)": "📈",
    "매출 원인 파악":              "🔍",
    "데이터 탐색 / EDA":          "🔬",
    "보고서 작성":                 "📄",
    "기타":                       "💬",
}

# ── CSS ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    :root {
        --bg:       #0a0f1e;
        --surface:  #111827;
        --surface2: #1a2233;
        --border:   #1e2d45;
        --cyan:     #06b6d4;
        --cyan-dim: rgba(6,182,212,0.12);
        --text:     #e2e8f0;
        --muted:    #64748b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Noto Sans KR', sans-serif !important;
    }

    /* 사이드바 — 다크 유지 */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* 메인 영역 — 밝고 깨끗하게 */
    [data-testid="stMain"] {
        background: #f8fafc !important;
    }
    .main .block-container {
        background: #f8fafc !important;
        max-width: 800px !important;
        padding: 2rem 2rem !important;
    }

    /* 채팅 입력 */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: #1e293b !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #06b6d4 !important;
        box-shadow: 0 2px 12px rgba(6,182,212,0.15) !important;
    }

    /* 사이드바 버튼 */
    [data-testid="stSidebar"] .stButton > button {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--cyan) !important;
        background: var(--cyan-dim) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--cyan) !important;
        border-color: var(--cyan) !important;
        color: #0a0f1e !important;
        font-weight: 600 !important;
    }

    /* 메인 버튼 — 밝은 스타일 */
    [data-testid="stMain"] .stButton > button {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stMain"] .stButton > button:hover {
        border-color: #06b6d4 !important;
        box-shadow: 0 2px 8px rgba(6,182,212,0.15) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"] {
        background: #06b6d4 !important;
        border-color: #06b6d4 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(6,182,212,0.3) !important;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"]:hover {
        background: #0891b2 !important;
        transform: translateY(-1px) !important;
    }

    /* 카드 선택 버튼 */
    .elens-card {
        background: var(--surface2);
        border: 1.5px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .elens-card:hover, .elens-card.selected {
        border-color: var(--cyan);
        background: var(--cyan-dim);
    }
    .elens-card .icon { font-size: 1.4rem; }
    .elens-card .label { font-size: 0.95rem; font-weight: 500; }
    .elens-card .desc  { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

    /* 타이틀 */
    .elens-title {
        font-family: 'Sora', sans-serif !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #06b6d4, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0 !important;
    }
    .elens-sub {
        color: var(--muted) !important;
        font-size: 0.9rem !important;
        margin-top: 4px !important;
        letter-spacing: 0.3px;
    }

    /* 온보딩 스텝 */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        background: var(--cyan-dim);
        border: 1px solid var(--cyan);
        border-radius: 50%;
        color: var(--cyan);
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .step-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }

    /* 업로드 존 */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--cyan) !important;
    }

    /* 세션 카드 */
    .session-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .session-card .s-time { color: var(--muted); font-size: 0.75rem; }
    .session-card .s-tag  {
        display: inline-block;
        background: var(--cyan-dim);
        color: var(--cyan);
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.72rem;
        margin-right: 4px;
    }

    /* divider */
    hr { border-color: var(--border) !important; }

    /* 슬라이더 — 흰색 바, 하늘색 커서 */
    [data-testid="stSelectSlider"] { color: var(--text) !important; }
    [data-testid="stSlider"] [role="slider"] {
        background: #06b6d4 !important;
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 4px rgba(6,182,212,0.2) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
        background: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
        background: #06b6d4 !important;
    }

    /* 채팅 메시지 영역 — 흰색 배경 */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #1e293b !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f0f9ff !important;
        border-color: #bae6fd !important;
    }
    /* 메인 채팅 배경을 밝게 */
    [data-testid="stMain"] > div {
        background: #f8fafc !important;
    }
    .main .block-container {
        background: #f8fafc !important;
        padding: 2rem 3rem !important;
    }

    /* expander */
    [data-testid="stExpander"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── 세션 상태 초기화 ──────────────────────────────────────────────────

def _init() -> None:
    defaults = {
        "messages": [], "session_id": None, "graph_config": None,
        "data_meta": None, "user_profile": None, "agent_results": {},
        "pending_hitl": False, "hitl_value": None,
        # 온보딩 상태
        "onboard_step": 0,      # 0=대기, 1=파일, 2=직군, 3=목적, 4=자율도, 5=완료
        "onboard_file": None,
        "onboard_role": None,
        "onboard_purpose": None,
        "onboard_hitl": 2,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _session_ready() -> bool:
    return bool(st.session_state.session_id and st.session_state.user_profile)


def _collect_chart_paths() -> list[str]:
    ag04 = st.session_state.agent_results.get("AG-04", {})
    return [str(Path(p).resolve()) for p in (ag04.get("image_paths") or [])
            if Path(str(p)).is_file()]


def _append_assistant(text: str) -> None:
    st.session_state.messages.append({
        "role": "assistant", "content": text,
        "image_paths": _collect_chart_paths(),
    })


def _apply_result(result: dict) -> None:
    st.session_state.agent_results = result["agent_results"]
    if result["status"] == "interrupt":
        st.session_state.hitl_value   = result["interrupt"]
        st.session_state.pending_hitl = True
    else:
        st.session_state.pending_hitl = False
        st.session_state.hitl_value   = None
        text = (result.get("final_response") or "").strip() or "처리 완료."
        _append_assistant(text)
        if st.session_state.session_id:
            update_session(st.session_state.session_id, summary=text[:200])


# ── 온보딩 플로우 ─────────────────────────────────────────────────────

def _render_onboarding() -> None:
    """대화창에서 진행하는 온보딩 스텝"""

    step = st.session_state.onboard_step

    # 웰컴 메시지
    if step == 0:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem 2rem;">
            <p class="elens-title">E_LENS</p>
            <p class="elens-sub">이커머스 데이터분석 자동화의 모든 것</p>
        </div>
        """, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🔭"):
            st.markdown("안녕하세요! **E_LENS**입니다. 분석할 데이터를 업로드해주세요.")

        st.session_state.onboard_step = 1
        st.rerun()

    # Step 1 — 파일 업로드
    elif step == 1:
        with st.chat_message("assistant", avatar="🔭"):
            st.markdown('<div class="step-title"><span class="step-badge">1</span>분석할 CSV 파일을 올려주세요.</div>', unsafe_allow_html=True)

            sample_files = sorted(SAMPLE_DATA_DIR.glob("*.csv"))
            col1, col2   = st.columns([3, 2])

            with col1:
                uploaded = st.file_uploader("CSV 파일", type=["csv"],
                                             label_visibility="collapsed")
            with col2:
                if sample_files:
                    if st.button("📂 샘플 데이터 사용", use_container_width=True):
                        st.session_state.onboard_file = str(sample_files[0])
                        st.session_state.onboard_step = 2
                        st.rerun()

            if uploaded:
                path = save_upload(uploaded.getvalue(), uploaded.name)
                st.session_state.onboard_file = str(path)
                st.session_state.onboard_step = 2
                st.rerun()

    # Step 2 — 직군 선택
    elif step == 2:
        with st.chat_message("assistant", avatar="🔭"):
            st.markdown('<div class="step-title"><span class="step-badge">2</span>어떤 역할로 분석하시나요?</div>', unsafe_allow_html=True)

            cols = st.columns(2)
            for i, (role, icon) in enumerate(ROLE_ICONS.items()):
                with cols[i % 2]:
                    if st.button(f"{icon}  {role}", key=f"role_{i}",
                                 use_container_width=True):
                        st.session_state.onboard_role = role
                        st.session_state.onboard_step = 3
                        st.rerun()

    # Step 3 — 분석 목적
    elif step == 3:
        with st.chat_message("assistant", avatar="🔭"):
            st.markdown('<div class="step-title"><span class="step-badge">3</span>오늘 분석 목적은 무엇인가요?</div>', unsafe_allow_html=True)

            for purpose, icon in PURPOSE_ICONS.items():
                if st.button(f"{icon}  {purpose}", key=f"purpose_{purpose}",
                             use_container_width=True):
                    st.session_state.onboard_purpose = purpose
                    st.session_state.onboard_step    = 4
                    st.rerun()

    # Step 4 — 자율도
    elif step == 4:
        with st.chat_message("assistant", avatar="🔭"):
            st.markdown('<div class="step-title"><span class="step-badge">4</span>에이전트 자율도를 선택해주세요.</div>', unsafe_allow_html=True)

            hitl = st.select_slider(
                "자율도",
                options=[0, 1, 2, 3, 4],
                value=2,
                format_func=lambda v: f"L{v}  {HITL_LEVELS[v][0]}",
                label_visibility="collapsed",
            )
            name, desc = HITL_LEVELS[hitl]
            st.caption(f"**Level {hitl} — {name}:** {desc}")

            if st.button("🚀  분석 시작", type="primary", use_container_width=True):
                st.session_state.onboard_hitl = hitl
                _finish_onboarding()
                st.rerun()

    # Step 5 — 완료 (분석 시작)
    elif step == 5:
        pass  # _session_ready()가 True → 채팅 모드로 전환


def _finish_onboarding() -> None:
    user    = get_current_user()
    sid     = make_session_id()
    path    = Path(st.session_state.onboard_file)
    role    = st.session_state.onboard_role
    purpose = st.session_state.onboard_purpose
    hitl    = st.session_state.onboard_hitl

    data_meta = _make_data_meta(path)
    guide     = PERSONA_GUIDE.get((role, purpose),
                                   f"{role} + {purpose} 목적으로 분석을 도와드릴게요.")

    st.session_state.session_id   = sid
    st.session_state.graph_config = {"configurable": {"thread_id": sid}}
    st.session_state.data_meta    = data_meta
    st.session_state.user_profile = {
        "role": role, "purpose": purpose,
        "hitl_level": hitl, "guide": guide,
    }
    st.session_state.agent_results = {}
    st.session_state.onboard_step  = 5

    # 시작 메시지
    rows = data_meta.get("row_count", "?")
    cols = data_meta.get("col_count", "?")
    fname = path.name
    _append_assistant(
        f"**{fname}** 파일을 불러왔어요. ({rows}행 · {cols}컬럼)\n\n"
        f"{guide}\n\n무엇부터 분석할까요?"
    )

    if user:
        save_session(user["id"], sid, role=role, purpose=purpose, hitl_level=hitl)


# ── HITL 렌더링 ───────────────────────────────────────────────────────

def _render_hitl() -> None:
    iv          = st.session_state.hitl_value or {}
    input_type  = iv.get("input_type", "selection")
    question    = iv.get("llm_question", "")
    message     = iv.get("message", "")
    options     = iv.get("options", ["승인", "수정", "재실행"])
    point       = iv.get("hitl_point", "")
    llm_choices = iv.get("llm_choices", [])

    with st.chat_message("assistant", avatar="🔭"):
        st.markdown(f"**에이전트 확인 단계**  \n`{point}`")

        with st.form("hitl_form"):
            free_text_answer = ""
            choice, modify   = options[0], ""

            if input_type == "free_text":
                if question:
                    st.markdown(f"> {question}")
                if llm_choices:
                    all_ch = llm_choices + ["직접 입력"]
                    sel    = st.radio("", all_ch, key="hitl_llm")
                    free_text_answer = (
                        st.text_area("", key="hitl_ft", height=80,
                                     label_visibility="collapsed")
                        if sel == "직접 입력" else sel
                    )
                else:
                    free_text_answer = st.text_area("", key="hitl_ft", height=80,
                                                    label_visibility="collapsed")
            else:
                if message:
                    st.markdown(message)
                choice = st.radio("", options, horizontal=True, key="hitl_sel")
                if choice == "수정":
                    modify = st.text_input("수정 내용", key="hitl_mod")

            submitted = st.form_submit_button("전달", use_container_width=True)

    if not submitted:
        return

    if input_type == "free_text":
        if not free_text_answer.strip():
            st.warning("답변을 입력해주세요.")
            return
        resume: dict = {"user_answer": free_text_answer.strip()}
    else:
        resume = {"response": choice, "user_answer": "", "modified_input": {}}
        if choice == "수정" and modify.strip():
            resume["user_answer"]    = modify.strip()
            resume["modified_input"] = {"user_input": modify.strip()}

    with st.spinner(""):
        result = asyncio.run(graph_step(
            Command(resume=resume),
            st.session_state.graph_config,
            st.session_state.agent_results,
        ))
    _apply_result(result)
    st.rerun()


def _run_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(""):
        result = asyncio.run(graph_step(
            build_turn_state(
                user_input=prompt,
                session_id=st.session_state.session_id,
                data_meta=st.session_state.data_meta,
                agent_results=st.session_state.agent_results,
                user_profile=st.session_state.user_profile,
            ),
            st.session_state.graph_config,
            st.session_state.agent_results,
        ))
    _apply_result(result)


# ── 사이드바 ──────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    user = get_current_user()

    with st.sidebar:
        # 로고
        st.markdown("""
        <div style="padding: 8px 0 16px;">
            <span style="font-family:Sora,sans-serif; font-size:1.4rem;
                         font-weight:700; background:linear-gradient(135deg,#06b6d4,#818cf8);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                E_LENS
            </span>
        </div>
        """, unsafe_allow_html=True)

        # 계정 정보
        st.markdown(f"""
        <div style="background:#1a2233; border:1px solid #1e2d45;
                    border-radius:10px; padding:12px 14px; margin-bottom:12px;">
            <div style="font-weight:600; font-size:0.95rem;">{user['name']}</div>
            <div style="color:#64748b; font-size:0.78rem; margin-top:2px;">{user['email']}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if col1.button("로그아웃", use_container_width=True):
            logout()
            st.rerun()
        if user.get("role") == "admin":
            col2.page_link("pages/admin.py", label="관리자", icon="⚙️")

        # 새 분석 시작
        st.divider()
        if st.button("➕  새 분석 시작", use_container_width=True, type="primary"):
            _reset_session()
            st.rerun()

        # 자율도 변경 (세션 중)
        if _session_ready():
            st.divider()
            cur = st.session_state.user_profile.get("hitl_level", 2)
            new = st.select_slider(
                "자율도",
                options=[0, 1, 2, 3, 4],
                value=cur,
                format_func=lambda v: f"L{v} {HITL_LEVELS[v][0]}",
            )
            if new != cur:
                st.session_state.user_profile["hitl_level"] = new
                _append_assistant(f"자율도 → **Level {new} ({HITL_LEVELS[new][0]})**")
                st.rerun()

        # 세션 기록
        st.divider()
        st.markdown("**세션 기록**")
        sessions = get_sessions_by_user(user["id"])
        if not sessions:
            st.caption("기록 없음")
        for s in sessions[:8]:
            created = s.get("created_at", "")[:16]
            status  = "✅" if s.get("status") == "completed" else "🔄"
            with st.expander(f"{status} {created}", expanded=False):
                if s.get("role"):    st.caption(f"직군: {s['role']}")
                if s.get("purpose"): st.caption(f"목적: {s['purpose']}")
                lvl = s.get("hitl_level", 2)
                st.caption(f"자율도: L{lvl} {HITL_LEVELS.get(lvl,('',))[0]}")
                if s.get("summary"): st.caption(s["summary"][:80])


def _reset_session() -> None:
    for k in ("session_id", "graph_config", "data_meta", "user_profile",
              "pending_hitl", "hitl_value",
              "onboard_file", "onboard_role", "onboard_purpose"):
        st.session_state[k] = None
    st.session_state.agent_results = {}
    st.session_state.messages      = []
    st.session_state.onboard_step  = 0
    st.session_state.onboard_hitl  = 2


# ── 메인 ─────────────────────────────────────────────────────────────

def main() -> None:
    _init()
    ensure_db()
    st.set_page_config(page_title="E_LENS", layout="wide",
                       page_icon="🔭", initial_sidebar_state="expanded")
    _inject_css()

    if not is_logged_in():
        render_login_page()
        return

    _render_sidebar()

    # 채팅 기록 렌더링
    for m in st.session_state.messages:
        with st.chat_message(m["role"],
                             avatar="🔭" if m["role"] == "assistant" else None):
            st.markdown(m["content"])
            for ip in m.get("image_paths") or []:
                p = Path(str(ip))
                if p.is_file():
                    st.image(str(p), caption=p.name)

    # HITL
    if st.session_state.pending_hitl:
        _render_hitl()
    elif not _session_ready():
        # 온보딩 플로우
        _render_onboarding()
    else:
        # 채팅 입력
        if prompt := st.chat_input("무엇을 분석할까요?"):
            _run_prompt(prompt)
            st.rerun()


main()