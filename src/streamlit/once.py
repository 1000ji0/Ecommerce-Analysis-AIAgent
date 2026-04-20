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
        --bg:       #070d19;
        --surface:  #0d1526;
        --surface2: #111c31;
        --border:   #243554;
        --cyan:     #06b6d4;
        --cyan-dim: rgba(6,182,212,0.12);
        --text:     #dbe7f5;
        --muted:    #8ea3bb;
        --main-bg:  radial-gradient(1200px 600px at 70% -5%, #14213d 0%, #0a1324 45%, #070d19 100%);
        --card:     #111b30;
        --card-soft:#132540;
        --card-bd:  #2b3f61;
        --ink:      #e7eef8;
        --ink-sub:  #a5b8cc;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Noto Sans KR', sans-serif !important;
    }

    /* 사이드바 — 다크 유지 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1526 0%, #0a1323 100%) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 12px 0 34px rgba(4, 10, 22, 0.32) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* 메인 영역 — 밝고 깨끗하게 */
    [data-testid="stMain"] {
        background: var(--main-bg) !important;
    }
    .main .block-container {
        background: transparent !important;
        max-width: 900px !important;
        padding: 2.25rem 2.5rem !important;
    }

    /* 채팅 입력 */
    [data-testid="stChatInput"] {
        background: var(--card) !important;
        border: 1.5px solid var(--card-bd) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(4,10,22,0.36) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: var(--ink) !important;
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8ea3bb !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 1px rgba(6,182,212,0.35), 0 12px 28px rgba(6,182,212,0.2) !important;
    }

    /* 사이드바 버튼 */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
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
        background: var(--card) !important;
        border: 1.5px solid var(--card-bd) !important;
        color: var(--ink) !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 6px 18px rgba(4,10,22,0.28) !important;
    }
    [data-testid="stMain"] .stButton > button:hover {
        border-color: #06b6d4 !important;
        box-shadow: 0 6px 18px rgba(6,182,212,0.15) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
        border-color: #0ea5c4 !important;
        color: #f8fbff !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 24px rgba(14,165,233,0.35) !important;
    }
    [data-testid="stMain"] .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0891b2, #2563eb) !important;
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
        font-size: 3.25rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #06b6d4, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.7px;
        margin: 0 !important;
        line-height: 1.05 !important;
    }
    .elens-sub {
        color: #a9bdd3 !important;
        font-size: 1.08rem !important;
        margin-top: 10px !important;
        letter-spacing: 0.3px;
        font-weight: 500 !important;
    }

    .hero-wrap {
        text-align: center;
        padding: 2.8rem 1rem 2.2rem;
    }
    .hero-card {
        max-width: 720px;
        margin: 0 auto;
        background: linear-gradient(170deg, rgba(17,27,48,0.96), rgba(13,21,38,0.92));
        border: 1px solid #2e466c;
        border-radius: 18px;
        padding: 1.7rem 1.8rem 1.5rem;
        box-shadow: 0 20px 46px rgba(4,10,22,0.45);
        backdrop-filter: blur(5px);
    }
    .hero-kicker {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(6,182,212,0.14);
        color: #8ee9ff;
        border: 1px solid rgba(6,182,212,0.35);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 0.7rem;
    }
    .hero-points {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .hero-chip {
        font-size: 0.78rem;
        font-weight: 600;
        color: #bdd8f5;
        background: #14233d;
        border: 1px solid #36517a;
        border-radius: 999px;
        padding: 0.3rem 0.62rem;
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
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }

    /* 업로드 존 */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.015) !important;
        border: 1.5px dashed #3b557a !important;
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
        background: var(--card) !important;
        border-radius: 14px !important;
        border: 1px solid var(--card-bd) !important;
        color: #0f172a !important;
        margin-bottom: 10px !important;
        box-shadow: 0 10px 22px rgba(4,10,22,0.32) !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #0f172a !important;
        line-height: 1.5 !important;
    }
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessageContent"] *,
    [data-testid="stChatMessage"] a {
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: var(--card-soft) !important;
        border-color: #386aa2 !important;
    }
    /* 메인 채팅 배경을 밝게 */
    [data-testid="stMain"] > div {
        background: var(--main-bg) !important;
    }
    .main .block-container {
        background: transparent !important;
        padding: 2.25rem 2.75rem !important;
    }

    /* expander */
    [data-testid="stExpander"] {
        background: #101a2f !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    .sidebar-brand {
        padding: 8px 0 16px;
        font-family: 'Sora', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.2px;
    }

    .sidebar-account {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .sidebar-account .name {
        font-weight: 600;
        font-size: 0.95rem;
    }
    .sidebar-account .email {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 2px;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--ink-sub) !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] label,
    [data-testid="stMain"] div {
        color: var(--ink) !important;
    }

    .chat-top-wrap {
        margin-bottom: 0.9rem;
    }
    .chat-top {
        border: 1px solid var(--card-bd);
        background: linear-gradient(160deg, rgba(18, 29, 51, 0.96), rgba(12, 21, 39, 0.92));
        border-radius: 16px;
        padding: 1rem 1.05rem 0.95rem;
        box-shadow: 0 10px 24px rgba(4, 10, 22, 0.26);
    }
    .chat-top-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: -0.2px;
        color: #e7eef8;
        margin-bottom: 0.18rem;
    }
    .chat-top-sub {
        font-size: 0.84rem;
        color: #9fb3ca;
        margin-bottom: 0.65rem;
    }
    .chat-top-chips {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
    }
    .chat-top-chip {
        font-size: 0.74rem;
        color: #bee7ff;
        background: rgba(6,182,212,0.12);
        border: 1px solid rgba(6,182,212,0.34);
        border-radius: 999px;
        padding: 0.18rem 0.56rem;
        font-weight: 600;
    }
    .starter-label {
        margin-top: 0.55rem;
        margin-bottom: 0.35rem;
        color: #9fb3ca;
        font-size: 0.8rem;
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


def _render_chat_top_header() -> None:
    profile = st.session_state.get("user_profile") or {}
    role = profile.get("role") or "분석 준비"
    purpose = profile.get("purpose") or "데이터 연결 후 맞춤 분석 시작"

    st.markdown(
        f"""
        <div class="chat-top-wrap">
          <div class="chat-top">
            <div class="chat-top-title">무엇을 도와드릴까요?</div>
            <div class="chat-top-sub">현재 모드: <b>{role}</b> · {purpose}</div>
            <div class="chat-top-chips">
              <span class="chat-top-chip">대화형 분석</span>
              <span class="chat-top-chip">자동 차트 생성</span>
              <span class="chat-top-chip">인사이트 요약</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_starter_prompts() -> None:
    if not _session_ready() or st.session_state.pending_hitl:
        return

    has_user_message = any(m.get("role") == "user" for m in st.session_state.messages)
    if has_user_message:
        return

    st.markdown('<div class="starter-label">빠른 시작</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    starters = [
        ("📈 성과 요약", "업로드된 데이터 기준 핵심 성과 지표를 5줄로 요약해줘"),
        ("🔎 EDA 시작", "데이터 품질 점검과 EDA를 먼저 진행해줘"),
        ("🧠 인사이트", "매출에 영향을 주는 핵심 요인과 액션 아이템을 알려줘"),
    ]

    if c1.button(starters[0][0], key="starter_1", use_container_width=True):
        _run_prompt(starters[0][1])
        st.rerun()
    if c2.button(starters[1][0], key="starter_2", use_container_width=True):
        _run_prompt(starters[1][1])
        st.rerun()
    if c3.button(starters[2][0], key="starter_3", use_container_width=True):
        _run_prompt(starters[2][1])
        st.rerun()


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
        <div class="hero-wrap">
            <div class="hero-card">
                <div class="hero-kicker">AI Commerce Intelligence</div>
                <p class="elens-title">E_LENS</p>
                <p class="elens-sub">이커머스 데이터 분석을 더 빠르고 선명하게 시작하세요</p>
                <div class="hero-points">
                    <span class="hero-chip">데이터 업로드 즉시 분석</span>
                    <span class="hero-chip">직군별 맞춤 인사이트</span>
                    <span class="hero-chip">대화형 리포트 생성</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🔭"):
            st.markdown("안녕하세요! **E_LENS**입니다.  \
            준비된 CSV를 올리면 바로 분석을 시작할 수 있어요.")

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
        answer_text = str(free_text_answer or "")
        if not answer_text.strip():
            st.warning("답변을 입력해주세요.")
            return
        resume: dict = {"user_answer": answer_text.strip()}
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
    if not user:
        return

    with st.sidebar:
        # 로고
        st.markdown("""
        <div class="sidebar-brand">E_LENS</div>
        """, unsafe_allow_html=True)

        # 계정 정보
        st.markdown(f"""
        <div class="sidebar-account">
            <div class="name">{user['name']}</div>
            <div class="email">@{user.get('login_id', user.get('email', ''))}</div>
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
    _render_chat_top_header()
    _render_starter_prompts()

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