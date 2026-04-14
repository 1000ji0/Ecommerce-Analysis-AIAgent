"""
T-10 Schema RAG (production-ready v2)

테이블·컬럼 메타데이터를 벡터 인덱싱하고,
자연어 질문으로 관련 스키마 컨텍스트를 검색한다.

주요 기능:
- schema dict / json 파일 인덱싱
- table / column 단위 문서화
- 자연어 기반 schema 검색
- SQL Tool용 context 생성
- 재인덱싱 지원
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

COLLECTION_NAME = "schema_metadata"
DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"

_client: chromadb.PersistentClient | None = None
_collection: Collection | None = None


# ============================================================
# Path / Chroma bootstrap
# ============================================================

def _project_root() -> Path:
    """
    src/tools/database/t10_schema_rag.py -> project root
    """
    return Path(__file__).resolve().parents[3]


def _chroma_dir() -> Path:
    """
    세션별이 아닌 공용 스키마 인덱스 저장 경로
    """
    path = _project_root() / "data" / "chroma_schema"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_collection() -> Collection:
    global _client, _collection

    if _collection is not None:
        return _collection

    try:
        _client = chromadb.PersistentClient(path=str(_chroma_dir()))
        ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name=DEFAULT_EMBEDDING_MODEL,
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        return _collection

    except Exception as exc:
        logger.exception("[T-10] Failed to initialize Chroma collection: %s", exc)
        raise RuntimeError(f"Chroma collection 초기화 실패: {exc}") from exc


# ============================================================
# Schema validation
# ============================================================

def _validate_schema(schema: dict[str, Any]) -> None:
    """
    최소 schema 형식 검증

    expected:
    {
        "table_name": {
            "description": "...",
            "columns": {
                "col_name": {"type": "...", "description": "..."}
            }
        }
    }
    """
    if not isinstance(schema, dict):
        raise ValueError("schema는 dict여야 합니다.")

    for table_name, table_info in schema.items():
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table 이름은 비어 있지 않은 문자열이어야 합니다.")

        if not isinstance(table_info, dict):
            raise ValueError(f"table '{table_name}' 정보는 dict여야 합니다.")

        columns = table_info.get("columns", {})
        if columns is None:
            columns = {}

        if not isinstance(columns, dict):
            raise ValueError(f"table '{table_name}'의 columns는 dict여야 합니다.")

        for col_name, col_info in columns.items():
            if not isinstance(col_name, str) or not col_name.strip():
                raise ValueError(f"table '{table_name}'의 column 이름은 문자열이어야 합니다.")

            if not isinstance(col_info, dict):
                raise ValueError(
                    f"table '{table_name}' / column '{col_name}' 정보는 dict여야 합니다."
                )


# ============================================================
# ID helpers
# ============================================================

def _table_doc_id(db_name: str, table: str) -> str:
    return f"table::{db_name}::{table}"


def _column_doc_id(db_name: str, table: str, column: str) -> str:
    return f"column::{db_name}::{table}::{column}"


def _collect_schema_ids(schema: dict[str, Any], db_name: str) -> list[str]:
    ids: list[str] = []

    for table, info in schema.items():
        ids.append(_table_doc_id(db_name, table))
        for col in info.get("columns", {}).keys():
            ids.append(_column_doc_id(db_name, table, col))

    return ids


# ============================================================
# Indexing
# ============================================================

def index_schema(
    schema: dict[str, Any],
    *,
    db_name: str = "default",
    dialect: str = "unknown",
    replace_existing: bool = True,
) -> dict[str, Any]:
    """
    DB 스키마를 벡터 DB에 임베딩·저장

    Args:
        schema:
            {
                "table_name": {
                    "description": "테이블 설명",
                    "columns": {
                        "col_name": {"type": "str", "description": "컬럼 설명"}
                    }
                }
            }
        db_name: DB 식별 이름
        dialect: mysql / sqlite / postgres ...
        replace_existing: 같은 db_name + table/column id가 있으면 삭제 후 재삽입

    Returns:
        {
            "indexed_count": int,
            "db_name": str,
            "dialect": str,
            "error": str | None,
        }
    """
    try:
        _validate_schema(schema)
        collection = _get_collection()

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for table, info in schema.items():
            table_desc = str(info.get("description", "")).strip()
            columns = info.get("columns", {}) or {}

            # 테이블 단위 문서
            table_doc = (
                f"DB: {db_name}\n"
                f"Dialect: {dialect}\n"
                f"테이블: {table}\n"
                f"설명: {table_desc}\n"
                f"컬럼: {', '.join(columns.keys())}"
            )
            documents.append(table_doc)
            metadatas.append({
                "db_name": db_name,
                "dialect": dialect,
                "table": table,
                "type": "table",
                "description": table_desc,
            })
            ids.append(_table_doc_id(db_name, table))

            # 컬럼 단위 문서
            for col, col_info in columns.items():
                col_type = str(col_info.get("type", "")).strip()
                col_desc = str(col_info.get("description", "")).strip()

                col_doc = (
                    f"DB: {db_name}\n"
                    f"Dialect: {dialect}\n"
                    f"테이블: {table}\n"
                    f"컬럼: {col}\n"
                    f"타입: {col_type}\n"
                    f"설명: {col_desc}"
                )
                documents.append(col_doc)
                metadatas.append({
                    "db_name": db_name,
                    "dialect": dialect,
                    "table": table,
                    "column": col,
                    "column_type": col_type,
                    "type": "column",
                    "description": col_desc,
                })
                ids.append(_column_doc_id(db_name, table, col))

        if not documents:
            return {
                "indexed_count": 0,
                "db_name": db_name,
                "dialect": dialect,
                "error": None,
            }

        if replace_existing:
            try:
                existing_ids = _collect_schema_ids(schema, db_name)
                collection.delete(ids=existing_ids)
            except Exception:
                # 없어도 문제는 아니므로 warning 수준
                logger.warning("[T-10] Existing schema delete skipped for db=%s", db_name)

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return {
            "indexed_count": len(ids),
            "db_name": db_name,
            "dialect": dialect,
            "error": None,
        }

    except Exception as exc:
        logger.exception("[T-10] index_schema failed: %s", exc)
        return {
            "indexed_count": 0,
            "db_name": db_name,
            "dialect": dialect,
            "error": str(exc),
        }


# ============================================================
# Search
# ============================================================

def search_schema(
    question: str,
    n_results: int = 5,
    *,
    db_name: str | None = None,
) -> dict[str, Any]:
    """
    자연어 질문으로 관련 테이블·컬럼 검색

    Args:
        question: 자연어 질문
        n_results: 검색 개수
        db_name: 특정 DB로 필터링하고 싶을 때 사용

    Returns:
        {
            "tables": ["table1", ...],
            "columns": [{"table": ..., "column": ...}, ...],
            "matches": [
                {
                    "type": "table" | "column",
                    "table": str,
                    "column": str | None,
                    "db_name": str,
                    "dialect": str,
                    "distance": float | None,
                    "document": str,
                }
            ],
            "context": str,
            "error": str | None,
        }
    """
    try:
        collection = _get_collection()

        try:
            total_count = collection.count()
        except Exception:
            total_count = 0

        if total_count == 0:
            return {
                "tables": [],
                "columns": [],
                "matches": [],
                "context": "스키마 없음",
                "error": None,
            }

        query_kwargs: dict[str, Any] = {
            "query_texts": [question],
            "n_results": n_results,
        }

        if db_name:
            query_kwargs["where"] = {"db_name": db_name}

        results = collection.query(**query_kwargs)

        metadatas = (results.get("metadatas") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        tables: set[str] = set()
        columns: list[dict[str, str]] = []
        matches: list[dict[str, Any]] = []
        context_parts: list[str] = []

        for idx, (meta, doc) in enumerate(zip(metadatas, documents)):
            distance = distances[idx] if idx < len(distances) else None

            table_name = meta.get("table", "")
            column_name = meta.get("column")
            item_type = meta.get("type", "")
            item_db_name = meta.get("db_name", "")
            item_dialect = meta.get("dialect", "")

            context_parts.append(doc)

            if table_name:
                tables.add(table_name)

            if item_type == "column" and column_name:
                columns.append({
                    "table": table_name,
                    "column": column_name,
                })

            matches.append({
                "type": item_type,
                "table": table_name,
                "column": column_name,
                "db_name": item_db_name,
                "dialect": item_dialect,
                "distance": distance,
                "document": doc,
            })

        return {
            "tables": sorted(list(tables)),
            "columns": columns,
            "matches": matches,
            "context": "\n\n".join(context_parts),
            "error": None,
        }

    except Exception as exc:
        logger.exception("[T-10] search_schema failed: %s", exc)
        return {
            "tables": [],
            "columns": [],
            "matches": [],
            "context": "",
            "error": str(exc),
        }


# ============================================================
# File loader
# ============================================================

def load_schema_from_file(
    schema_path: str | Path,
    *,
    db_name: str = "default",
    dialect: str = "unknown",
    replace_existing: bool = True,
) -> dict[str, Any]:
    """
    JSON 파일에서 스키마 로드 후 인덱싱
    """
    try:
        path = Path(schema_path)
        if not path.exists():
            return {
                "indexed_count": 0,
                "db_name": db_name,
                "dialect": dialect,
                "error": f"스키마 파일이 존재하지 않습니다: {path}",
            }

        schema = json.loads(path.read_text(encoding="utf-8"))
        return index_schema(
            schema,
            db_name=db_name,
            dialect=dialect,
            replace_existing=replace_existing,
        )

    except json.JSONDecodeError as exc:
        logger.exception("[T-10] invalid schema JSON: %s", exc)
        return {
            "indexed_count": 0,
            "db_name": db_name,
            "dialect": dialect,
            "error": f"JSON 파싱 실패: {exc}",
        }
    except Exception as exc:
        logger.exception("[T-10] load_schema_from_file failed: %s", exc)
        return {
            "indexed_count": 0,
            "db_name": db_name,
            "dialect": dialect,
            "error": str(exc),
        }


# ============================================================
# Optional utility
# ============================================================

def clear_schema_index(*, db_name: str | None = None) -> dict[str, Any]:
    """
    전체 인덱스 삭제 또는 특정 db_name만 삭제
    """
    try:
        collection = _get_collection()

        if db_name:
            results = collection.get(where={"db_name": db_name})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
            return {"deleted_count": len(ids), "db_name": db_name, "error": None}

        # 전체 삭제
        all_items = collection.get()
        ids = all_items.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return {"deleted_count": len(ids), "db_name": None, "error": None}

    except Exception as exc:
        logger.exception("[T-10] clear_schema_index failed: %s", exc)
        return {"deleted_count": 0, "db_name": db_name, "error": str(exc)}