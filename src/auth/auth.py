"""
src/auth/auth.py
Streamlit 로그인/인증 로직
st.session_state 기반 세션 관리
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from auth.auth_db import (
    init_db,
    get_user_by_login_id,
    verify_password,
    update_last_login,
    create_signup_request,
)


def ensure_db() -> None:
    """앱 시작 시 DB 초기화"""
    init_db()


def is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def get_current_user() -> dict | None:
    return st.session_state.get("auth_user")


def is_admin() -> bool:
    user = get_current_user()
    return bool(user and user.get("role") == "admin")


def login(login_id: str, password: str) -> tuple[bool, str]:
    """
    로그인 처리
    Returns: (성공 여부, 에러 메시지)
    """
    user = get_user_by_login_id(login_id)

    if not user:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다."

    if not user.get("is_active"):
        return False, "비활성화된 계정입니다. 관리자에게 문의하세요."

    if not verify_password(password, user["password"]):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다."

    # 로그인 성공
    update_last_login(user["id"])
    st.session_state.auth_user = {
        "id":       user["id"],
        "login_id": user.get("login_id", ""),
        "email":    user.get("email", ""),
        "name":     user["name"],
        "role":     user["role"],
    }
    return True, ""


def logout() -> None:
    st.session_state.auth_user = None
    # 세션 데이터도 초기화
    for k in ("session_id", "graph_config", "data_meta", "user_profile",
              "agent_results", "messages", "pending_hitl", "hitl_value"):
        st.session_state[k] = None if k not in ("agent_results", "messages") else {}
    st.session_state.messages      = []
    st.session_state.agent_results = {}


def render_login_page() -> None:
    """로그인 / 가입 요청 페이지 렌더링"""
    # CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@700&family=Noto+Sans+KR:wght@400;500&display=swap');
    .login-title {
        font-family: 'Sora', sans-serif;
        font-size: 3.1rem; font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 4px;
    }
    .login-sub {
        text-align: center; color: #64748b;
        font-size: 0.88rem; margin-bottom: 2rem;
        font-family: 'Noto Sans KR', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">E_LENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">이커머스 데이터분석 자동화의 모든 것</div>',
                unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        # 탭 — 로그인 / 가입 요청
        tab_login, tab_signup = st.tabs(["🔑  로그인", "✍️  가입 요청"])

        # ── 로그인 탭 ──────────────────────────────────────────────
        with tab_login:
            with st.form("login_form"):
                login_id = st.text_input("아이디", placeholder="your_id")
                password = st.text_input("비밀번호", type="password")
                submit   = st.form_submit_button("로그인", use_container_width=True)

            if submit:
                if not login_id or not password:
                    st.error("아이디와 비밀번호를 입력해주세요.")
                else:
                    ok, msg = login(login_id.strip().lower(), password)
                    if ok:
                        st.rerun()
                    else:
                        st.error(msg)

        # ── 가입 요청 탭 ────────────────────────────────────────────
        with tab_signup:
            st.caption("가입 요청 시 입력한 비밀번호로 관리자 승인 후 계정이 생성됩니다.")
            with st.form("signup_form"):
                req_name    = st.text_input("이름")
                req_login_id = st.text_input("아이디")
                req_pw      = st.text_input("비밀번호", type="password")
                req_pw2     = st.text_input("비밀번호 확인", type="password")
                req_message = st.text_area("요청 메시지 (선택)", height=80,
                                           placeholder="소속, 사용 목적 등을 간단히 적어주세요.")
                req_submit  = st.form_submit_button("가입 요청", use_container_width=True)

            if req_submit:
                if not req_name or not req_login_id or not req_pw:
                    st.error("이름, 아이디, 비밀번호를 입력해주세요.")
                elif len(req_login_id.strip()) < 3:
                    st.error("아이디는 3자 이상으로 입력해주세요.")
                elif req_pw != req_pw2:
                    st.error("비밀번호 확인이 일치하지 않습니다.")
                else:
                    ok = create_signup_request(
                        name=req_name.strip(),
                        login_id=req_login_id.strip().lower(),
                        password=req_pw,
                        message=req_message.strip(),
                    )
                    if ok:
                        st.success(
                            "가입 요청이 접수됐어요. "
                            "관리자 승인 후 같은 아이디/비밀번호로 로그인해주세요."
                        )
                    else:
                        st.warning("이미 사용 중이거나 요청된 아이디입니다.")