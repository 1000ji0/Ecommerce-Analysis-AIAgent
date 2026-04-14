"""
T-22 Parameter Parser (production-ready v2)

사용자 자연어 입력 → 실행 파라미터 dict 변환 및 검증

특징:
- 룰 기반 추출 우선
- LLM 보조 추출
- 동의어 정규화
- JSON 블록 파싱 보완
- 검증 실패 이유 구조화 반환
- exec_tools 중복 제거
- 로깅 강화
- 예외 상황에서 안전하게 degrade

반환 구조:
{
    "params": dict,
    "rejected": dict[str, str],
    "warnings": list[str],
}
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None


# ============================================================
# Schema
# ============================================================

PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "outlier_method": {
        "choices": ["gaussian", "iqr"],
        "description": '"gaussian" | "iqr"',
    },
    "threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "float (0.0 ~ 1.0)",
    },
    "target_col": {
        "type": str,
        "description": "str (타겟 컬럼명)",
    },
    "n_new_samples": {
        "type": int,
        "min": 1,
        "description": "int (증강 샘플 수, 최소 1)",
    },
    "ncol": {
        "type": int,
        "min": 1,
        "description": "int (선택 변수 수, 최소 1)",
    },
    "criterion": {
        "choices": ["MSE", "MAE", "R2"],
        "description": '"MSE" | "MAE" | "R2"',
    },
    "exec_tools": {
        "multichoices": ["fpca", "nds", "timeseries", "stats"],
        "description": '["fpca", "nds", "timeseries", "stats"] 중 복수 선택',
    },
    "fold": {
        "type": int,
        "min": 1,
        "max": 20,
        "description": "int (k-fold 수, 1 ~ 20)",
    },
}


def _build_schema_desc(schema: dict[str, dict[str, Any]]) -> dict[str, str]:
    """LLM 프롬프트에 넣기 위한 설명 dict 자동 생성."""
    return {
        key: str(meta.get("description", ""))
        for key, meta in schema.items()
    }


SCHEMA_DESC = _build_schema_desc(PARAM_SCHEMA)

SYSTEM_PROMPT = (
    "사용자 입력에서 아래 파라미터 스키마에 맞는 값만 추출해라.\n"
    f"스키마: {json.dumps(SCHEMA_DESC, ensure_ascii=False)}\n"
    "규칙:\n"
    "- JSON만 반환, 마크다운 금지\n"
    "- 추출 불가 항목은 포함하지 않음\n"
    '- exec_tools는 반드시 list로 반환 (예: ["fpca", "stats"])\n'
    "- 숫자는 가능한 경우 숫자형으로 반환\n"
    "- 빈 결과면 {} 반환"
)


# ============================================================
# Synonyms / Aliases
# ============================================================
# 동의어·툴명 alias 정의
SYNONYMS: dict[str, str] = {
    # outlier_method
    "가우시안": "gaussian",
    "정규분포": "gaussian",
    "gauss": "gaussian",
    "아이큐알": "iqr",
    "사분위": "iqr",

    # criterion
    "평균제곱오차": "MSE",
    "mse": "MSE",
    "평균절대오차": "MAE",
    "mae": "MAE",
    "결정계수": "R2",
    "r2": "R2",
    "r²": "R2",

    # exec_tools
    "타임시리즈": "timeseries",
    "time_series": "timeseries",
    "시계열": "timeseries",
    "통계": "stats",
    "statistics": "stats",
}

TOOL_ALIASES: dict[str, str] = {
    "fpca": "fpca",
    "nds": "nds",
    "timeseries": "timeseries",
    "stats": "stats",
    "시계열": "timeseries",
    "통계": "stats",
    "타임시리즈": "timeseries",
    "time_series": "timeseries",
    "statistics": "stats",
}

TARGET_COL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"target_col\s*[=:]\s*([A-Za-z_가-힣][A-Za-z0-9_가-힣]*)", re.IGNORECASE),
    re.compile(r"target\s*col(?:umn)?\s*[=:]?\s*([A-Za-z_가-힣][A-Za-z0-9_가-힣]*)", re.IGNORECASE),
    re.compile(r"타겟\s*컬럼\s*(?:은|는|이|가|[:=])?\s*([A-Za-z_가-힣][A-Za-z0-9_가-힣]*)"),
    re.compile(r"목표\s*변수\s*(?:은|는|이|가|[:=])?\s*([A-Za-z_가-힣][A-Za-z0-9_가-힣]*)"),
    re.compile(r"타깃\s*컬럼\s*(?:은|는|이|가|[:=])?\s*([A-Za-z_가-힣][A-Za-z0-9_가-힣]*)"),
]


# ============================================================
# Rule-based extraction
# ============================================================
# 정규식 기반 명확 추출 패턴 정의
RuleConverter = Callable[[str], Any]
RULE_PATTERNS: list[tuple[re.Pattern[str], str, RuleConverter]] = [
    (re.compile(r"fold\s*[=:]?\s*(\d+)", re.IGNORECASE), "fold", int),
    (re.compile(r"(\d+)\s*fold", re.IGNORECASE), "fold", int),
    (re.compile(r"k[\s-]*fold\s*[=:]?\s*(\d+)", re.IGNORECASE), "fold", int),
    (re.compile(r"fold\s*는?\s*(\d+)", re.IGNORECASE), "fold", int),

    (re.compile(r"n_new_samples\s*[=:]?\s*(\d+)", re.IGNORECASE), "n_new_samples", int),
    (re.compile(r"샘플\s*수\s*[=:]?\s*(\d+)", re.IGNORECASE), "n_new_samples", int),
    (re.compile(r"증강\s*샘플\s*수\s*[=:]?\s*(\d+)", re.IGNORECASE), "n_new_samples", int),

    (re.compile(r"ncol\s*[=:]?\s*(\d+)", re.IGNORECASE), "ncol", int),
    (re.compile(r"변수\s*수\s*[=:]?\s*(\d+)", re.IGNORECASE), "ncol", int),
    (re.compile(r"(\d+)\s*개\s*변수", re.IGNORECASE), "ncol", int),

    (re.compile(r"threshold\s*[=:]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE), "threshold", float),
    (re.compile(r"임계값\s*[=:]?\s*([0-9]*\.?[0-9]+)"), "threshold", float),

    (re.compile(r"criterion\s*[=:]?\s*(MSE|MAE|R2|r²)", re.IGNORECASE), "criterion", str),
    (re.compile(r"평가\s*기준\s*[=:]?\s*(MSE|MAE|R2|r²)", re.IGNORECASE), "criterion", str),

    (re.compile(r"outlier_method\s*[=:]?\s*(gaussian|iqr)", re.IGNORECASE), "outlier_method", str),
    (re.compile(r"이상치\s*방법\s*[=:]?\s*(gaussian|iqr)", re.IGNORECASE), "outlier_method", str),
]


# ============================================================
# Public API
# ============================================================

def parse_params(user_input: str) -> dict[str, Any]:
    """
    자연어 입력에서 실행 파라미터 추출·검증

    Returns:
        {
            "params":   dict,
            "rejected": dict[str, str],
            "warnings": list[str],
        }
    """
    if not isinstance(user_input, str):
        raise TypeError(f"user_input must be str, got {type(user_input).__name__}")

    user_input = user_input.strip()
    logger.debug("[T-22] 입력: %s", user_input)

    if not user_input:
        return {
            "params": {},
            "rejected": {},
            "warnings": ["입력이 비어 있어 파라미터를 추출하지 못했습니다."],
        }

    normalized = _apply_synonyms(user_input)
    logger.debug("[T-22] 정규화 입력: %s", normalized)

    rule_params = _extract_by_rules(normalized)
    logger.debug("[T-22] 룰 추출: %s", rule_params)

    llm_params = _extract_by_llm(normalized)
    logger.debug("[T-22] LLM 추출: %s", llm_params)

    merged = _merge_params(rule_params=rule_params, llm_params=llm_params)
    logger.debug("[T-22] 병합 결과: %s", merged)

    params, rejected = _validate(merged)
    warnings = _build_warnings(rejected)

    logger.debug("[T-22] 최종 params=%s rejected=%s", params, rejected)

    return {
        "params": params,
        "rejected": rejected,
        "warnings": warnings,
    }


# ============================================================
# LLM
# ============================================================
# LLM 보조 추출: 애매한 표현 보완, JSON 파싱 실패 보완
def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def _extract_by_llm(text: str) -> dict[str, Any]:
    """
    애매한 표현은 LLM으로 보조 추출.
    실패 시 빈 dict 반환.
    """
    try:
        llm = _get_llm()
    except Exception as exc:
        logger.exception("[T-22] LLM 초기화 실패: %s", exc)
        return {}

    try:
        raw = _call_llm(llm, text)
        logger.debug("[T-22] LLM raw 응답(1차): %s", raw)
        params = _try_parse(raw)

        if params is not None:
            return params

        retry_prompt = f"반드시 JSON 객체만 반환해라. 마크다운 금지.\n입력: {text}"
        raw_retry = _call_llm(llm, retry_prompt)
        logger.debug("[T-22] LLM raw 응답(2차): %s", raw_retry)
        params = _try_parse(raw_retry)

        if params is not None:
            return params

        logger.warning("[T-22] LLM 응답 파싱 실패. 빈 dict 반환")
        return {}

    except Exception as exc:
        logger.exception("[T-22] LLM 추출 실패: %s", exc)
        return {}


def _call_llm(llm: ChatGoogleGenerativeAI, text: str) -> str:
    """LLM 호출 후 문자열 응답으로 정규화."""
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=text),
    ])

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ============================================================
# Normalize / Merge
# ============================================================
# 동의어 치환, exec_tools alias 정규화, 룰 기반 우선 병합
def _apply_synonyms(text: str) -> str:
    """
    한국어·약어를 내부 표현으로 치환.
    너무 공격적인 치환을 피하기 위해 정규식 경계 기반으로 적용.
    """
    result = text

    # 길이가 긴 키를 먼저 적용해서 부분 치환 충돌 줄임
    for src in sorted(SYNONYMS.keys(), key=len, reverse=True):
        dst = SYNONYMS[src]
        pattern = re.compile(rf"(?<![A-Za-z0-9_가-힣]){re.escape(src)}(?![A-Za-z0-9_가-힣])", re.IGNORECASE)
        result = pattern.sub(dst, result)

    return result


def _merge_params(rule_params: dict[str, Any], llm_params: dict[str, Any]) -> dict[str, Any]:
    """
    룰 기반 우선 병합.
    - rule_params가 명시적 패턴 추출이므로 우선
    - exec_tools는 합집합 병합
    """
    merged = dict(llm_params)

    for key, value in rule_params.items():
        if key == "exec_tools":
            merged_tools = _merge_exec_tools(
                llm_value=merged.get("exec_tools"),
                rule_value=value,
            )
            if merged_tools:
                merged["exec_tools"] = merged_tools
        else:
            merged[key] = value

    return merged


def _merge_exec_tools(llm_value: Any, rule_value: Any) -> list[str]:
    """exec_tools 전용 병합."""
    result: list[str] = []

    def _append_items(value: Any) -> None:
        if isinstance(value, list):
            items = value
        elif isinstance(value, str):
            items = re.split(r"[,\s/]+", value)
        else:
            return

        for item in items:
            if not item:
                continue
            normalized = TOOL_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
            if normalized not in result:
                result.append(normalized)

    _append_items(llm_value)
    _append_items(rule_value)
    return result


# ============================================================
# Rule extraction
# ============================================================
# 정규식 기반 명확 추출. 숫자·툴명·타겟 컬럼은 룰 기반이 더 안정적.
def _extract_by_rules(text: str) -> dict[str, Any]:
    """
    정규식 기반 명확 추출.
    숫자·툴명·타겟 컬럼은 룰 기반이 더 안정적.
    """
    result: dict[str, Any] = {}

    # 일반 패턴
    for pattern, key, converter in RULE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        try:
            result[key] = converter(match.group(1))
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("[T-22] 룰 추출 실패 key=%s pattern=%s err=%s", key, pattern.pattern, exc)

    # target_col
    target_col = _extract_target_col(text)
    if target_col:
        result["target_col"] = target_col

    # exec_tools
    exec_tools = _extract_exec_tools(text)
    if exec_tools:
        result["exec_tools"] = exec_tools

    return result


def _extract_target_col(text: str) -> str | None:
    """타겟 컬럼명 룰 기반 추출."""
    for pattern in TARGET_COL_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_exec_tools(text: str) -> list[str]:
    """텍스트 내 tool alias를 탐지해 exec_tools 추출."""
    found: list[str] = []
    lowered = text.lower()

    for alias, canonical in TOOL_ALIASES.items():
        pattern = re.compile(rf"(?<![a-z0-9_가-힣]){re.escape(alias.lower())}(?![a-z0-9_가-힣])")
        if pattern.search(lowered) and canonical not in found:
            found.append(canonical)

    return found


# ============================================================
# Parse helpers
# ============================================================

def _try_parse(raw: str) -> dict[str, Any] | None:
    """
    JSON 전체 파싱 시도
    실패 시 텍스트에서 JSON 블록만 추출해 재시도
    """
    if not isinstance(raw, str):
        raw = str(raw)

    cleaned = re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip()

    # 1차: 전체 파싱
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # 2차: 첫 JSON 객체 블록 추출
    block = _extract_first_json_object(cleaned)
    if block is None:
        return None

    try:
        parsed = json.loads(block)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    """
    문자열에서 첫 번째 JSON object 블록 { ... } 추출.
    중첩 brace를 단순 카운팅으로 처리.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


# ============================================================
# Validation
# ============================================================

def _validate(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """
    PARAM_SCHEMA 기준 검증
    Returns:
        (validated, rejected)
    """
    validated: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    for key, value in params.items():
        schema = PARAM_SCHEMA.get(key)
        if schema is None:
            rejected[key] = "지원하지 않는 파라미터"
            continue

        if "choices" in schema:
            _validate_choice(key, value, schema, validated, rejected)
            continue

        if "multichoices" in schema:
            _validate_multichoice(key, value, schema, validated, rejected)
            continue

        if "type" in schema:
            _validate_typed_value(key, value, schema, validated, rejected)
            continue

        rejected[key] = "알 수 없는 스키마 정의"

    return validated, rejected


def _validate_choice(
    key: str,
    value: Any,
    schema: dict[str, Any],
    validated: dict[str, Any],
    rejected: dict[str, str],
) -> None:
    choices = schema["choices"]
    choices_lower = [str(c).lower() for c in choices]
    value_norm = str(value).strip().lower()

    if value_norm in choices_lower:
        idx = choices_lower.index(value_norm)
        validated[key] = choices[idx]
    else:
        rejected[key] = f"허용값 아님: {value} (허용: {choices})"


def _validate_multichoice(
    key: str,
    value: Any,
    schema: dict[str, Any],
    validated: dict[str, Any],
    rejected: dict[str, str],
) -> None:
    allowed = schema["multichoices"]
    allowed_lower = [str(c).lower() for c in allowed]

    if isinstance(value, str):
        items = [v.strip() for v in re.split(r"[,\s/]+", value) if v.strip()]
    elif isinstance(value, list):
        items = value
    else:
        rejected[key] = "list 또는 comma-separated str 형식이어야 함"
        return

    clean: list[str] = []

    for item in items:
        item_str = str(item).strip()
        item_norm = TOOL_ALIASES.get(item_str.lower(), item_str.lower())

        if item_norm in allowed_lower:
            idx = allowed_lower.index(item_norm)
            canonical = allowed[idx]
            if canonical not in clean:
                clean.append(canonical)
        else:
            rejected[f"{key}[{item_str}]"] = f"허용값 아님 (허용: {allowed})"

    if clean:
        validated[key] = clean


def _validate_typed_value(
    key: str,
    value: Any,
    schema: dict[str, Any],
    validated: dict[str, Any],
    rejected: dict[str, str],
) -> None:
    expected_type = schema["type"]

    try:
        if expected_type is str:
            converted = str(value).strip()
        else:
            converted = expected_type(value)
    except (ValueError, TypeError):
        rejected[key] = f"타입 변환 실패: {value} → {expected_type.__name__}"
        return

    if expected_type is str and not converted:
        rejected[key] = "빈 문자열"
        return

    if "min" in schema and converted < schema["min"]:
        rejected[key] = f"최소값 미만: {converted} (최소 {schema['min']})"
        return

    if "max" in schema and converted > schema["max"]:
        rejected[key] = f"최대값 초과: {converted} (최대 {schema['max']})"
        return

    validated[key] = converted


# ============================================================
# Warning builder
# ============================================================

def _build_warnings(rejected: dict[str, str]) -> list[str]:
    """제거된 파라미터를 사용자 친화적 경고 메시지로 변환."""
    return [
        f"'{key}' 파라미터가 무시됐습니다: {reason}"
        for key, reason in rejected.items()
    ]