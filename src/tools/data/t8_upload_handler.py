"""
T-08 데이터 업로드 핸들러
CSV/Excel 파일 수신·저장·미리보기 제공 및 메타데이터 State 저장

특징:
- _count_rows: 전체 행 수 별도 계산 (미리보기 5행과 분리)
- _read_file: CSV 인코딩 자동 감지 (utf-8 / utf-8-sig / euc-kr / cp949)
- 반환값에 encoding, size_mb 추가
"""
import shutil
import uuid
from pathlib import Path

import pandas as pd

from config import DATA_DIR

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB   = 200
CSV_ENCODINGS      = ["utf-8", "utf-8-sig", "euc-kr", "cp949"]


###### main 함수: 파일 업로드 처리 ######
def handle_upload(file_path: str | Path, original_filename: str) -> dict:
    """
    업로드된 파일을 DATA_DIR에 저장하고 메타정보 반환

    Args:
        file_path:         업로드된 임시 파일 경로
        original_filename: 원본 파일명

    Returns:
        {
            "path":      str,    # 저장된 파일 경로
            "filename":  str,    # 저장된 파일명
            "encoding":  str,    # 감지된 인코딩 (CSV만, Excel은 None)
            "preview":   dict,   # 컬럼명·타입·샘플값 미리보기
            "row_count": int,    # 전체 행 수
            "col_count": int,    # 컬럼 수
            "size_mb":   float,  # 파일 크기 (MB)
        }
    """
    file_path = Path(file_path)
    suffix    = Path(original_filename).suffix.lower()

    # 확장자 검증
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 파일 형식: {suffix}. 허용: {ALLOWED_EXTENSIONS}"
        )

    # 파일 크기 검증
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"파일 크기 초과: {size_mb:.1f}MB (최대 {MAX_FILE_SIZE_MB}MB)"
        )

    # DATA_DIR에 저장 (uuid prefix로 파일명 충돌 방지)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_name = f"{uuid.uuid4().hex[:8]}_{original_filename}"
    save_path = DATA_DIR / save_name
    shutil.copy2(file_path, save_path)

    # 미리보기 + 인코딩 감지
    df, detected_encoding = _read_file(save_path, suffix)
    preview   = _make_preview(df)

    # 전체 행 수 별도 계산 (미리보기는 5행만 읽으므로)
    row_count = _count_rows(save_path, suffix, detected_encoding)

    return {
        "path":      str(save_path),
        "filename":  save_name,
        "encoding":  detected_encoding,
        "preview":   preview,
        "row_count": row_count,
        "col_count": len(df.columns),
        "size_mb":   round(size_mb, 2),
    }


### 내부 함수: 파일 읽기 + 인코딩 감지 ###
def _read_file(path: Path, suffix: str) -> tuple[pd.DataFrame, str | None]:
    """
    미리보기용 5행 읽기
    CSV: 인코딩 자동 감지 (utf-8 → utf-8-sig → euc-kr → cp949 순)
    Excel: openpyxl 기본 처리

    Returns:
        (DataFrame, detected_encoding)
        Excel이면 encoding은 None
    """
    if suffix == ".csv":
        for encoding in CSV_ENCODINGS:
            try:
                df = pd.read_csv(path, nrows=5, encoding=encoding)
                return df, encoding
            except (UnicodeDecodeError, Exception):
                continue
        raise ValueError(
            f"CSV 인코딩 감지 실패. 시도한 인코딩: {CSV_ENCODINGS}"
        )
    else:
        df = pd.read_excel(path, nrows=5)
        return df, None


### 내부 함수: 전체 행 수 계산 ###
def _count_rows(path: Path, suffix: str, encoding: str | None) -> int:
    """
    전체 행 수 계산 (미리보기와 분리)
    CSV: 줄 수 세기 (header 제외, 빠름)
    Excel: 첫 컬럼만 읽어서 행 수 확인
    """
    try:
        if suffix == ".csv":
            enc = encoding or "utf-8"
            with open(path, "r", encoding=enc, errors="ignore") as f:
                # header 1줄 제외
                return max(0, sum(1 for _ in f) - 1)
        else:
            # 첫 컬럼만 읽어 행 수 계산 (전체 로드보다 빠름)
            df_full = pd.read_excel(path, usecols=[0])
            return len(df_full)
    except Exception:
        return -1  # 계산 실패 시 -1 반환 (미리보기는 정상 제공)


### 내부 함수: 미리보기 dict 생성 ###
def _make_preview(df: pd.DataFrame) -> dict:
    """컬럼명·타입·샘플값 미리보기 dict 생성"""
    return {
        "columns": list(df.columns),
        "dtypes":  {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample":  df.head(3).to_dict(orient="records"),
    }