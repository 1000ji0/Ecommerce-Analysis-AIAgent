"""
T-21 피처 캐시 핸들러
Feature 생성 결과 캐싱 및 재사용 — 동일 데이터 재처리 방지
"""
import hashlib
import json
from pathlib import Path

from config import SESSION_DIR


def _cache_dir(session_id: str) -> Path:
    path = SESSION_DIR / session_id / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_key(data_path: str, exec_tools: list[str]) -> str:
    """data_path + exec_tools 조합으로 캐시 키 생성"""
    raw = f"{data_path}|{'_'.join(sorted(exec_tools))}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cache(session_id: str, data_path: str, exec_tools: list[str]) -> str | None:
    """
    캐시 적중 시 저장된 피처 파일 경로 반환, 없으면 None

    Returns:
        str: 캐시된 pickle 파일 경로
        None: 캐시 없음 → MCP T-01 호출 필요
    """
    key = _make_key(data_path, exec_tools)
    meta_file = _cache_dir(session_id) / f"{key}.json"

    if not meta_file.exists():
        return None

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    cached_path = Path(meta["feature_path"])

    # 캐시 파일이 실제로 존재하는지 확인
    if not cached_path.exists():
        meta_file.unlink(missing_ok=True)
        return None

    return str(cached_path)


def set_cache(
    session_id: str,
    data_path: str,
    exec_tools: list[str],
    feature_path: str,
) -> None:
    """
    피처 생성 결과를 캐시에 저장

    Args:
        feature_path: implement_fc 출력 pickle 파일 경로
    """
    key = _make_key(data_path, exec_tools)
    meta_file = _cache_dir(session_id) / f"{key}.json"

    meta = {
        "data_path": data_path,
        "exec_tools": exec_tools,
        "feature_path": feature_path,
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def clear_cache(session_id: str) -> None:
    """세션 캐시 전체 삭제"""
    cache_dir = _cache_dir(session_id)
    for f in cache_dir.glob("*.json"):
        f.unlink(missing_ok=True)
