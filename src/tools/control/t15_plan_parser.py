"""
T-15 분석 계획 파서
사용자 자연어 요청 + 데이터 메타정보 → 분석 실행 계획 생성
"""
import json
import re
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
 
from config import GEMINI_MODEL, GOOGLE_API_KEY
 
_llm = None
 
 # 유효한 Agent 목록 (AG-02~AG-05)
VALID_AGENTS = {"AG-02", "AG-03", "AG-04", "AG-05"}
 
 # 각 Agent별 기본값 (params에 명시적으로 다른 값이 없는 경우)
DEFAULT_PARAMS: dict[str, dict] = {
    "AG-02": {"outlier_method": "gaussian", "threshold": 0.8, "exec_tools": ["fpca", "nds", "timeseries", "stats"]},
    "AG-03": {"db_url": None},
    "AG-04": {"top_n": 5, "target_col": "TARGET"},
    "AG-05": {"format": "pdf"}
}
 
# 올바른 stage 실행 순서 (앞에 있을수록 먼저 실행)
STAGE_ORDER = ["AG-02", "AG-03", "AG-04", "AG-05"]
 
# AG별 params 스키마 (key: 허용 타입 또는 허용 값 목록)
PARAM_SCHEMA: dict[str, dict] = {
    "AG-02": {
        "outlier_method": ["gaussian", "iqr"],
        "threshold":      float,
        "exec_tools":     list,
    },
    "AG-03": {
        "db_url": str,
    },
    "AG-04": {
        "top_n":      int,
        "target_col": str,
    },
    "AG-05": {
        "format": ["pdf", "csv"],
    },
}
 
# AG-03을 포함해도 되는 키워드 (사용자 요청에 이 단어가 없으면 AG-03 제거)
SQL_KEYWORDS = [
    "sql", "db", "데이터베이스", "조회", "쿼리", "테이블",
    "kpi", "매출", "주문", "고객", "전환율", "roas",
]
 
SYSTEM_PROMPT = """
너는 데이터 분석 파이프라인 계획을 수립하는 AI다.
사용자 요청과 데이터 메타정보를 분석해서 아래 형식의 JSON만 반환해라.
마크다운 금지.
 
반환 형식:
{
  "stages": ["AG-02", "AG-04", "AG-05"],
  "params": {
    "AG-02": {"outlier_method": "gaussian", "threshold": 0.8},
    "AG-04": {"top_n": 5, "target_col": "TARGET"},
    "AG-05": {"format": "pdf"}
  },
  "description": "한국어로 계획 요약 2~3줄"
}
 
사용 가능한 Agent:
- AG-02: 데이터 전처리 + Feature Engineering (MCP T-01~07)
- AG-03: SQL 데이터 조회 및 KPI 분석 (SQL/DB 조회가 명시적으로 필요한 경우만)
- AG-04: EDA, 변수 중요도, 인사이트 도출
- AG-05: 보고서 생성 (PDF/CSV)
 
파라미터 가이드:
AG-02:
  - outlier_method: "gaussian" | "iqr"  (기본: "gaussian")
  - threshold: 0.0 ~ 1.0  (다중공선성 임계값, 기본: 0.8)
  - exec_tools: ["fpca", "nds", "timeseries", "stats"] 중 선택  (기본: 전체)
AG-03:
  - db_url: str  (DB 연결 URL, 사용자가 제공한 경우만 포함)
AG-04:
  - top_n: int  (중요 변수 수, 기본: 5)
  - target_col: str  (타겟 컬럼명, 기본: "TARGET")
AG-05:
  - format: "pdf" | "csv"  (기본: "pdf")
 
실행 순서 규칙:
- AG-02는 반드시 AG-04보다 앞에
- AG-05는 반드시 마지막
- AG-03은 SQL 조회가 명시적으로 필요한 경우만 포함
- params는 기본값과 다른 경우에만 포함
"""
 
 
### 내부 함수: LLM 인스턴스 생성 (lazy initialization) ###
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm
 
 
###### main 함수: 분석 실행 계획 생성 ######
def parse_plan(user_input: str, data_meta: dict) -> dict:
    """
    사용자 요청과 데이터 메타정보를 바탕으로 분석 실행 계획 생성
 
    Args:
        user_input: 사용자 자연어 요청
        data_meta:  T-08 upload_handler 반환값
                    {"path": str, "row_count": int, "col_count": int,
                     "preview": {"columns": list, "dtypes": dict, "sample": list}}
 
    Returns:
        {
            "stages":      list[str],  # 실행할 agent 순서
            "params":      dict,       # 각 stage 파라미터
            "description": str,        # 계획 요약 (HITL 표시용, 최소 10자)
        }
    """
    llm = _get_llm()
 
    meta_str = _summarize_meta(data_meta)
    user_msg = (
        f"사용자 요청: {user_input}\n\n"
        f"데이터 메타정보:\n{meta_str}"
    )
 
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
 
    raw  = response.content.strip()
    plan = _parse_response(raw, llm, user_msg)
    plan = _validate_plan(plan, user_input)
 
    return plan
 
 
### 내부 함수: data_meta 핵심 요약 ###
def _summarize_meta(data_meta: dict) -> str:
    """
    data_meta에서 LLM에 필요한 핵심 정보만 추출
    - 컬럼 타입 요약
    - sample 데이터 포함 (컬럼 성격 파악: 날짜/ID/타겟 후보 등)
    """
    preview       = data_meta.get("preview", {})
    columns       = preview.get("columns", [])
    dtypes        = preview.get("dtypes", {})
    sample        = preview.get("sample", [])
 
    # 컬럼 타입 분류
    numeric_cols  = [c for c, t in dtypes.items() if "int" in t or "float" in t]
    category_cols = [c for c, t in dtypes.items() if "object" in t or "category" in t]
 
    # sample 요약 — 첫 2행만, 컬럼별 값 확인용
    sample_str = "없음"
    if sample:
        sample_str = json.dumps(sample[:2], ensure_ascii=False)
 
    return (
        f"- 파일 경로: {data_meta.get('path', '없음')}\n"
        f"- 행 수: {data_meta.get('row_count', '?')}\n"
        f"- 컬럼 수: {data_meta.get('col_count', '?')}\n"
        f"- 전체 컬럼: {columns}\n"
        f"- 수치형 컬럼 ({len(numeric_cols)}개): {numeric_cols[:10]}\n"
        f"- 범주형 컬럼 ({len(category_cols)}개): {category_cols[:10]}\n"
        f"- 샘플 데이터 (2행): {sample_str}"
    )
 
 
### 내부 함수: JSON 파싱 (실패 시 재시도 1회) ###
def _parse_response(raw: str, llm, user_msg: str) -> dict:
    """
    LLM 응답 JSON 파싱
    1차 실패 시 재시도 1회
    2차 실패 시 기본 계획 반환
    """
    def try_parse(text: str) -> dict | None:
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
 
    result = try_parse(raw)
    if result is not None:
        return result
 
    retry_msg = (
        "반드시 JSON 형식으로만 반환해라. 설명, 마크다운 금지.\n\n"
        f"원래 요청:\n{user_msg}"
    )
    response2 = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=retry_msg),
    ])
    result = try_parse(response2.content.strip())
    if result is not None:
        return result
 
    return {
        "stages":      ["AG-02", "AG-04", "AG-05"],
        "params":      {},
        "description": "기본 분석 파이프라인을 실행합니다.",
    }
 
 
### 내부 함수: 계획 스키마 검증 및 보정 ###
def _validate_plan(plan: dict, user_input: str) -> dict:
    """
    파싱된 계획의 스키마 검증 및 보정
 
    1. stages: 유효 agent + 올바른 실행 순서 보장
    2. AG-03: user_input에 SQL 관련 키워드 없으면 제거
    3. params: agent별 세부값 타입 검증
    4. description: 최소 10자 이상
    """
    # 1. stages 유효 agent 필터링
    stages = plan.get("stages", [])
    if not isinstance(stages, list) or not stages:
        stages = ["AG-02", "AG-04", "AG-05"]
 
    valid_stages = [s for s in stages if s in VALID_AGENTS]
    if not valid_stages:
        valid_stages = ["AG-02", "AG-04", "AG-05"]
 
    # 2. AG-03 포함 여부 규칙 검증
    user_lower = user_input.lower()
    has_sql_intent = any(kw in user_lower for kw in SQL_KEYWORDS)
    if "AG-03" in valid_stages and not has_sql_intent:
        valid_stages.remove("AG-03")
 
    # 3. stage 실행 순서 보장 (STAGE_ORDER 기준으로 정렬)
    valid_stages = sorted(valid_stages, key=lambda x: STAGE_ORDER.index(x))
 
    # AG-05는 항상 마지막
    if "AG-05" in valid_stages:
        valid_stages.remove("AG-05")
        valid_stages.append("AG-05")
 
    plan["stages"] = valid_stages
 
    # 4. params 세부값 타입 검증
    params = plan.get("params", {})
    if not isinstance(params, dict):
        params = {}
 
    validated_params = {}
    for agent, agent_params in params.items():
        if agent not in VALID_AGENTS:
            continue
        if not isinstance(agent_params, dict):
            continue
 
        schema = PARAM_SCHEMA.get(agent, {})
        clean = {}
        for key, val in agent_params.items():
            expected = schema.get(key)
            if expected is None:
                continue  # 스키마에 없는 키 제거
            # 허용 값 목록인 경우
            if isinstance(expected, list):
                if str(val).lower() in [v.lower() for v in expected]:
                    clean[key] = str(val).lower()
            # 타입 검증
            elif expected == float:
                try:
                    clean[key] = float(val)
                except (ValueError, TypeError):
                    pass
            elif expected == int:
                try:
                    clean[key] = int(val)
                except (ValueError, TypeError):
                    pass
            elif expected == str:
                if isinstance(val, str) and val.strip():
                    clean[key] = val.strip()
            elif expected == list:
                if isinstance(val, list):
                    clean[key] = val
 
        if clean:
            validated_params[agent] = clean
 
    plan["params"] = validated_params
 
    # 5. description 최소 길이 검증 (10자 이상)
    desc = plan.get("description", "")
    if not isinstance(desc, str) or len(desc.strip()) < 10:
        stages_str = " → ".join(plan["stages"])
        plan["description"] = f"{stages_str} 순서로 분석 파이프라인을 실행합니다."
 
    return plan