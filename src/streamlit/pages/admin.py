"""
src/streamlit/pages/admin.py
관리자 전용 페이지 — 사용자 관리 + 세션 현황
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from auth.auth import ensure_db, is_logged_in, is_admin, get_current_user, logout
from auth.auth_db import (
    get_all_users, create_user, update_user, delete_user,
    get_all_sessions,
)


def main() -> None:
    ensure_db()
    st.set_page_config(page_title="E_LENS 관리자", layout="wide")

    if not is_logged_in():
        st.warning("로그인이 필요합니다.")
        st.stop()

    if not is_admin():
        st.error("관리자만 접근할 수 있습니다.")
        st.stop()

    user = get_current_user()

    # 헤더
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("E_LENS 관리자")
        st.caption(f"로그인: {user['name']} ({user['email']})")
    with col2:
        if st.button("로그아웃"):
            logout()
            st.rerun()

    st.divider()

    tab1, tab2 = st.tabs(["👥 사용자 관리", "📋 세션 현황"])

    # ── 탭 1: 사용자 관리 ─────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("사용자 목록")
            users = get_all_users()
            if users:
                df = pd.DataFrame(users)[
                    ["id", "name", "email", "role", "is_active", "created_at", "last_login"]
                ]
                df.columns = ["ID", "이름", "이메일", "권한", "활성", "가입일", "최근 로그인"]
                df["활성"] = df["활성"].map({1: "✅", 0: "❌"})
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("사용자 없음")

        with col_right:
            # 사용자 추가
            st.subheader("사용자 추가")
            with st.form("add_user_form"):
                new_name  = st.text_input("이름")
                new_email = st.text_input("이메일")
                new_pw    = st.text_input("임시 비밀번호", type="password")
                new_role  = st.selectbox("권한", ["user", "admin"])
                if st.form_submit_button("추가", use_container_width=True):
                    if not all([new_name, new_email, new_pw]):
                        st.error("모든 항목을 입력해주세요.")
                    else:
                        try:
                            create_user(
                                email=new_email.strip().lower(),
                                password=new_pw,
                                name=new_name.strip(),
                                role=new_role,
                            )
                            st.success(f"{new_name} 계정 추가됨")
                            st.rerun()
                        except Exception as e:
                            st.error(f"추가 실패: {e}")

            st.divider()

            # 사용자 수정
            st.subheader("사용자 수정")
            users_for_edit = [u for u in users if u["email"] != user["email"]]
            if users_for_edit:
                target = st.selectbox(
                    "대상",
                    users_for_edit,
                    format_func=lambda u: f"{u['name']} ({u['email']})",
                    key="edit_target",
                )
                with st.form("edit_user_form"):
                    edit_name     = st.text_input("이름", value=target["name"])
                    edit_active   = st.toggle("활성", value=bool(target["is_active"]))
                    edit_pw       = st.text_input("새 비밀번호 (빈칸=유지)", type="password")
                    col_a, col_b  = st.columns(2)
                    save_btn      = col_a.form_submit_button("저장", use_container_width=True)
                    del_btn       = col_b.form_submit_button("삭제", use_container_width=True)

                if save_btn:
                    update_user(
                        user_id=target["id"],
                        name=edit_name or None,
                        is_active=edit_active,
                        password=edit_pw if edit_pw else None,
                    )
                    st.success("수정 완료")
                    st.rerun()

                if del_btn:
                    delete_user(target["id"])
                    st.success("삭제 완료")
                    st.rerun()

    # ── 탭 2: 세션 현황 ───────────────────────────────────────────────
    with tab2:
        st.subheader("전체 세션 현황")
        sessions = get_all_sessions()

        if sessions:
            # 요약 지표
            total     = len(sessions)
            completed = sum(1 for s in sessions if s.get("status") == "completed")
            running   = sum(1 for s in sessions if s.get("status") == "running")

            m1, m2, m3 = st.columns(3)
            m1.metric("전체 세션",   total)
            m2.metric("완료",        completed)
            m3.metric("진행 중",     running)

            st.divider()

            df = pd.DataFrame(sessions)
            display_cols = ["session_id", "user_name", "user_email",
                            "role", "purpose", "hitl_level",
                            "status", "created_at", "summary"]
            available = [c for c in display_cols if c in df.columns]
            df = df[available].copy()
            df.columns = [
                "세션 ID", "사용자", "이메일",
                "직군", "목적", "HITL 레벨",
                "상태", "생성일", "요약"
            ][:len(available)]

            # 필터
            col_f1, col_f2 = st.columns(2)
            status_filter = col_f1.selectbox("상태 필터", ["전체", "completed", "running"])
            if status_filter != "전체":
                df = df[df["상태"] == status_filter]

            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("세션 기록 없음")


main()