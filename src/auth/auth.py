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
    get_user_by_email,
    verify_password,
    update_last_login,
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


def login(email: str, password: str) -> tuple[bool, str]:
    """
    로그인 처리
    Returns: (성공 여부, 에러 메시지)
    """
    user = get_user_by_email(email.strip().lower())

    if not user:
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."

    if not user.get("is_active"):
        return False, "비활성화된 계정입니다. 관리자에게 문의하세요."

    if not verify_password(password, user["password"]):
        return False, "이메일 또는 비밀번호가 올바르지 않습니다."

    # 로그인 성공
    update_last_login(user["id"])
    st.session_state.auth_user = {
        "id":    user["id"],
        "email": user["email"],
        "name":  user["name"],
        "role":  user["role"],
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
    """로그인 페이지 렌더링"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="font-size: 2.2rem; font-weight: 600;">E_LENS</h1>
        <p style="color: #666; font-size: 1rem;">Ecommerce Analytics Agent</p>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        with st.form("login_form"):
            st.markdown("#### 로그인")
            email    = st.text_input("이메일", placeholder="admin@elens.com")
            password = st.text_input("비밀번호", type="password")
            submit   = st.form_submit_button("로그인", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("이메일과 비밀번호를 입력해주세요.")
            else:
                ok, msg = login(email, password)
                if ok:
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error(msg)