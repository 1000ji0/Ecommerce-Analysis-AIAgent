"""
src/streamlit/pages/admin.py
관리자 전용 페이지 — 사용자 관리 + 세션 현황
"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Streamlit Cloud / 로컬 모두 대응
_HERE = Path(__file__).resolve()
for _candidate in [
    _HERE.parents[3],           # root
    _HERE.parents[3] / "src",   # src
    _HERE.parents[2],           # src (로컬)
    _HERE.parents[1],           # streamlit
]:
    _p = str(_candidate)
    if _candidate.exists() and _p not in sys.path:
        sys.path.insert(0, _p)

from auth.auth import ensure_db, is_logged_in, is_admin, get_current_user, logout
from auth.auth_db import (
    get_all_users, create_user, update_user, delete_user,
    get_all_sessions, get_all_signup_requests,
    approve_signup_request, reject_signup_request,
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
        st.caption(f"로그인: {user['name']} ({user.get('login_id', '-')})")
    with col2:
        if st.button("로그아웃"):
            logout()
            st.rerun()

    st.divider()

    tab1, tab2, tab3 = st.tabs(["👥 사용자 관리", "📋 세션 현황", "📬 가입 요청"])

    # ── 탭 1: 사용자 관리 ─────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("사용자 목록")
            users = get_all_users()
            if users:
                df = pd.DataFrame(users)[
                    ["id", "login_id", "name", "role", "is_active", "created_at", "last_login"]
                ]
                df.columns = ["ID", "아이디", "이름", "권한", "활성", "가입일", "최근 로그인"]
                df["활성"] = df["활성"].map({1: "✅", 0: "❌"})
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("사용자 없음")

        with col_right:
            # 사용자 추가
            st.subheader("사용자 추가")
            with st.form("add_user_form"):
                new_name  = st.text_input("이름")
                new_login_id = st.text_input("아이디")
                new_pw    = st.text_input("비밀번호", type="password")
                new_role  = st.selectbox("권한", ["user", "admin"])
                if st.form_submit_button("추가", use_container_width=True):
                    if not all([new_name, new_login_id, new_pw]):
                        st.error("모든 항목을 입력해주세요.")
                    else:
                        try:
                            create_user(
                                login_id=new_login_id.strip().lower(),
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
            users_for_edit = [u for u in users if u.get("id") != user.get("id")]
            if users_for_edit:
                target = st.selectbox(
                    "대상",
                    users_for_edit,
                    format_func=lambda u: f"{u['name']} ({u.get('login_id', '-')})",
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
            display_cols = ["session_id", "user_name", "user_login_id",
                            "role", "purpose", "hitl_level",
                            "status", "created_at", "summary"]
            available = [c for c in display_cols if c in df.columns]
            df = df[available].copy()
            df.columns = [
                "세션 ID", "사용자", "아이디",
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

    # ── 탭 3: 가입 요청 ───────────────────────────────────────────────
    with tab3:
        requests = get_all_signup_requests()
        pending  = [r for r in requests if r["status"] == "pending"]

        if pending:
            st.info(f"대기 중인 가입 요청 {len(pending)}건")
        else:
            st.success("대기 중인 요청 없음")

        for req in requests:
            status_icon = {"pending": "⏳", "approved": "✅", "rejected": "❌"}.get(
                req["status"], "❓"
            )
            with st.expander(
                f"{status_icon}  {req['name']}  ({req.get('login_id', '-')})  —  {req['created_at'][:16]}",
                expanded=(req["status"] == "pending"),
            ):
                if req.get("message"):
                    st.markdown(f"**요청 메시지:** {req['message']}")
                st.caption(f"상태: {req['status']}")

                if req["status"] == "pending":
                    col_a, col_b = st.columns(2)
                    if col_a.button("✅ 승인", key=f"approve_{req['id']}",
                                    use_container_width=True):
                        created = approve_signup_request(req["id"])
                        if created:
                            name = req['name']
                            st.success(f"{name} 계정 승인 완료")
                            st.rerun()
                        else:
                            st.error("승인 실패: 요청 정보(아이디/비밀번호)를 확인해주세요.")

                    if col_b.button("❌ 거절", key=f"reject_{req['id']}",
                                    use_container_width=True):
                        reject_signup_request(req["id"])
                        st.warning(f"{req['name']} 요청 거절됨")
                        st.rerun()


main()