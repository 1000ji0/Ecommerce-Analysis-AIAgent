"""
T-14 Insight & Action Tool

t11(kpi_result), t12(eda_result), t13(feature_importance) 결과를 통합 분석해서
KPI·EDA·Feature 결과 통합 분석 → 핵심 인사이트 및 액션 아이템 제안
"""
import json
import re
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
 
from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.output.t20_trace_logger import log_tool_call
 
_llm = None
 
 
### 내부 함수: LLM 인스턴스 생성 (lazy initialization) ###
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm
 
 
###### main 함수: 인사이트 및 액션 아이템 생성 ######
def generate_insight(
    session_id: str,
    question: str,
    eda_result: dict | None = None,
    kpi_result: dict | None = None,
    feature_importance: dict | None = None,
) -> dict:
    """
    분석 결과 통합 → 인사이트 및 액션 아이템 생성
 
    Args:
        session_id:         세션 ID
        question:           원래 사용자 질문
        eda_result:         T-12 결과
                            {"summary": {shape, dtypes, missing, outlier_ratio,
                                         describe, target_corr_top5},
                             "chart_type": str, "chart_code": str}
        kpi_result:         T-11 결과
                            {"kpi_result": dict, "segment_result": dict}
        feature_importance: T-13 결과
                            {"task": str, "corr_ranking": dict,
                             "lgbm_ranking": dict, "final_ranking": dict,
                             "explanation": str, "valid_rows": int}
 
    Returns:
        {
            "insights":        [str, ...],  # 핵심 인사이트 (3개)
            "actions":         [str, ...],  # 실행 가능한 액션 아이템 (3개)
            "viz_suggestions": [str, ...],  # 시각화 구성 제안 (2개)
            "summary":         str,         # 전체 요약
        }
    """
    llm = _get_llm()
 
    has_eda = eda_result is not None
    has_kpi = kpi_result is not None
    has_fi  = feature_importance is not None
 
    context = _build_context(eda_result, kpi_result, feature_importance)
    system  = _build_system_prompt(has_eda, has_kpi, has_fi)
    msg     = f"사용자 질문: {question}\n\n분석 결과:\n{context}"
 
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    raw      = response.content.strip()
 
    result = _parse_response(raw, llm, msg, system)
    result = _validate_result(result)
 
    log_tool_call(session_id, "insight_action", {"question": question}, result)
    return result
 
 
### 내부 함수: 분석 결과 컨텍스트 구성 ###
def _build_context(
    eda_result: dict | None,
    kpi_result: dict | None,
    feature_importance: dict | None,
) -> str:
    """
    T-12 / T-13 / T-11 실제 출력 스키마 기준으로 핵심 수치 선별 요약
    - context 과도하게 길어지지 않도록 각 항목 제한
    - segment top 3만 포함
    - 긴 값은 truncate
    """
    parts = []
 
    # ── T-12 EDA 결과 ────────────────────────────────────────────────
    if eda_result:
        summary = eda_result.get("summary", {})
 
        # 데이터 크기
        shape = summary.get("shape", {})
        shape_str = f"{shape.get('rows', '?')}행 × {shape.get('cols', '?')}열"
 
        # 결측값 — 상위 3개만
        missing = summary.get("missing", {})
        top_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:3]
        missing_str = ", ".join(f"{col}({cnt})" for col, cnt in top_missing) or "없음"
 
        # 이상치 — 상위 3개만 (비율 높은 순)
        outlier = summary.get("outlier_ratio", {})
        top_outlier = sorted(outlier.items(), key=lambda x: x[1], reverse=True)[:3]
        outlier_str = ", ".join(f"{col}({v:.1%})" for col, v in top_outlier) or "없음"
 
        # 타겟 상관 top5
        corr_top5 = summary.get("target_corr_top5", {})
        corr_str = ", ".join(f"{col}={v}" for col, v in list(corr_top5.items())[:5]) or "없음"
 
        parts.append(
            "[EDA 분석 결과]\n"
            f"- 데이터 크기: {shape_str}\n"
            f"- 주요 결측 컬럼 (top3): {missing_str}\n"
            f"- 주요 이상치 컬럼 (top3): {outlier_str}\n"
            f"- 타겟 상관 top5: {corr_str}"
        )
 
    # ── T-11 KPI 결과 ────────────────────────────────────────────────
    if kpi_result:
        kpi = kpi_result.get("kpi_result", {})
        kpi_str = ", ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in kpi.items()
        )
        parts.append(f"[KPI 결과]\n{kpi_str}")
 
        # 세그먼트 — top 3만
        seg = kpi_result.get("segment_result", {})
        if seg:
            top_seg = dict(list(seg.items())[:3])
            seg_str = _truncate(top_seg, max_len=400)
            parts.append(f"[세그먼트별 KPI (top3)]\n{seg_str}")
 
    # ── T-13 Feature Importance 결과 ─────────────────────────────────
    if feature_importance:
        task        = feature_importance.get("task", "unknown")
        task_kor    = "분류" if task == "classification" else "회귀"
        final_rank  = feature_importance.get("final_ranking", {})
        explanation = feature_importance.get("explanation", "")
        valid_rows  = feature_importance.get("valid_rows", "?")
 
        rank_str = ", ".join(
            f"{i+1}위 {feat}({score})"
            for i, (feat, score) in enumerate(list(final_rank.items())[:5])
        )
 
        parts.append(
            "[변수 중요도 결과]\n"
            f"- 분석 유형: {task_kor} ({valid_rows}행 학습)\n"
            f"- 통합 순위 (Borda Count): {rank_str}\n"
            f"- 모델 해석: {explanation[:200]}"
        )
 
    return "\n\n".join(parts) if parts else "분석 결과 없음"
 
 
### 내부 함수: 동적 system prompt 생성 ###
def _build_system_prompt(has_eda: bool, has_kpi: bool, has_fi: bool) -> str:
    """가용한 분석 결과에 따라 system prompt 동적 생성"""
    available = []
    if has_eda: available.append("EDA/통계 분석")
    if has_kpi: available.append("KPI 지표")
    if has_fi:  available.append("변수 중요도")
 
    sources = "·".join(available) if available else "제한된 정보"
 
    return (
        "너는 데이터 분석 전문가다.\n"
        f"사용 가능한 분석 결과: {sources}\n\n"
        "아래 JSON 형식으로만 반환해라. 마크다운 금지.\n\n"
        "{\n"
        '  "insights": ["인사이트1", "인사이트2", "인사이트3"],\n'
        '  "actions": ["액션1", "액션2", "액션3"],\n'
        '  "viz_suggestions": ["시각화 제안1", "시각화 제안2"],\n'
        '  "summary": "전체 요약 2~3문장"\n'
        "}\n\n"
        "규칙:\n"
        "- 제공된 수치를 직접 인용해서 구체적으로 작성\n"
        "- insights: 데이터에서 발견한 핵심 패턴 (3개)\n"
        "- actions: 즉시 실행 가능한 구체적 액션 (3개)\n"
        "- viz_suggestions: 추가로 보면 좋을 시각화 구성 (2개)\n"
        "- 모든 항목 한국어"
    )
 
 
### 내부 함수: JSON 파싱 (실패 시 재시도 1회) ###
def _parse_response(raw: str, llm, msg: str, system: str) -> dict:
    """
    LLM 응답 JSON 파싱
    1차 실패 시 재시도 1회 — "JSON만 반환" 강조
    2차 실패 시 fallback dict 반환
    """
    def try_parse(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
 
    # 1차 시도
    result = try_parse(raw)
    if result is not None:
        return result
 
    # 2차 시도 — JSON 형식 강조 재요청
    retry_msg = (
        "반드시 JSON 형식으로만 반환해라. 설명, 마크다운 금지.\n\n"
        f"원래 질문:\n{msg}"
    )
    response2 = llm.invoke([SystemMessage(content=system), HumanMessage(content=retry_msg)])
    result = try_parse(response2.content.strip())
    if result is not None:
        return result
 
    # 최종 fallback
    return {
        "insights":        ["분석 결과를 파싱하지 못했습니다."],
        "actions":         ["데이터를 다시 확인해주세요."],
        "viz_suggestions": [],
        "summary":         "분석 결과 요약을 생성하지 못했습니다. 데이터를 확인 후 다시 시도해주세요.",
    }
 
 
### 내부 함수: 파싱 결과 스키마 검증 ###
def _validate_result(result: dict) -> dict:
    """
    파싱된 JSON 구조 검증 및 보정
    - 필수 키 존재 여부
    - 타입 검증 (list / str)
    - 최소 길이 보장
    """
    defaults = {
        "insights":        ["분석 결과를 확인해주세요."],
        "actions":         ["추가 분석이 필요합니다."],
        "viz_suggestions": [],
        "summary":         "분석 결과 요약을 생성하지 못했습니다.",
    }
 
    for key, default in defaults.items():
        val = result.get(key)
 
        # 키 없거나 타입 불일치
        if val is None or not isinstance(val, type(default)):
            result[key] = default
            continue
 
        # list면 최소 1개 이상 보장
        if isinstance(default, list) and len(val) == 0:
            result[key] = default
 
        # str이면 비어있으면 default
        if isinstance(default, str) and not str(val).strip():
            result[key] = default
 
    return result
 
 
### 유틸: dict/str truncate ###
def _truncate(obj, max_len: int = 400) -> str:
    """긴 dict를 JSON 문자열로 변환 후 max_len 초과 시 자름"""
    s = json.dumps(obj, ensure_ascii=False)
    return s[:max_len] + "..." if len(s) > max_len else s