"""
T-19 시각화 생성기
EDA & Viz Tool 출력 차트 코드 실행 → PNG/HTML 파일 저장

개선사항:
- _is_safe_code: exec 실행 전 위험 패턴 검사
- _detect_library: import문 기반 라이브러리 감지 (plotly/seaborn/matplotlib)
- fig None 체크: Plotly fig 없으면 명확한 에러
- 파일명 timestamp 추가: 덮어쓰기 방지
- matplotlib 상태 정리: 실행 전후 plt.close("all")
"""
import re
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서 렌더링
import matplotlib.pyplot as plt
import pandas as pd

from config import SESSION_DIR
from tools.output.t20_trace_logger import log_tool_call


# exec 실행 시 차단할 위험 패턴
FORBIDDEN_PATTERNS = [
    r"\bos\b", r"\bsubprocess\b", r"\beval\b",
    r"\bopen\b", r"\bshutil\b", r"__import__",
    r"\bexec\b", r"\bcompile\b", r"\bglobals\b", r"\blocals\b",
]


###### main 함수: 차트 코드 실행 → 파일 저장 ######
def generate_chart(
    session_id: str,
    chart_code: str,
    df: pd.DataFrame,
    chart_type: str = "chart",
    output_format: str = "png",
) -> dict:
    """
    차트 코드를 실행하고 이미지 파일로 저장

    Args:
        session_id:    세션 ID
        chart_code:    T-12에서 생성된 Plotly/Seaborn/Matplotlib 코드
        df:            데이터프레임 (코드 내에서 'df' 변수로 접근)
        chart_type:    차트 식별명 (파일명에 사용)
        output_format: "png" | "html"

    Returns:
        {
            "image_path": str,        # 저장된 파일 경로
            "success":    bool,
            "library":    str,        # 감지된 라이브러리
            "error":      str | None,
        }
    """
    chart_dir = SESSION_DIR / session_id / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 파일명에 timestamp 추가 — 덮어쓰기 방지
    timestamp   = datetime.now().strftime("%H%M%S")
    filename    = f"{chart_type}_{timestamp}.{output_format}"
    output_path = chart_dir / filename

    # 라이브러리 감지
    library = _detect_library(chart_code)

    try:
        # 코드 안전성 검사
        if not _is_safe_code(chart_code):
            raise ValueError("차트 코드에 허용되지 않는 패턴이 포함되어 있습니다.")

        # 라이브러리별 실행
        if library == "plotly":
            _run_plotly(chart_code, df, output_path, output_format)
        elif library == "seaborn":
            _run_seaborn(chart_code, df, output_path)
        else:
            _run_matplotlib(chart_code, df, output_path)

        result = {
            "image_path": str(output_path),
            "success":    True,
            "library":    library,
            "error":      None,
        }

    except Exception:
        result = {
            "image_path": "",
            "success":    False,
            "library":    library,
            "error":      traceback.format_exc(),
        }

    log_tool_call(session_id, "visualizer", {"chart_type": chart_type, "library": library}, result)
    return result


### 내부 함수: 라이브러리 감지 ###
def _detect_library(code: str) -> str:
    """
    import문 기반 차트 라이브러리 감지
    plotly > seaborn > matplotlib 순으로 우선 감지
    """
    if re.search(r"import plotly|from plotly|import px|from px", code):
        return "plotly"
    if re.search(r"import seaborn|from seaborn|import sns|from sns", code):
        return "seaborn"
    return "matplotlib"


### 내부 함수: 코드 안전성 검사 ###
def _is_safe_code(code: str) -> bool:
    """
    exec 실행 전 위험 패턴 차단
    os, subprocess, eval, open, shutil 등 시스템 접근 패턴 차단
    """
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return False
    return True


### 내부 함수: 실행 컨텍스트 생성 ###
def _make_exec_globals(df: pd.DataFrame) -> dict:
    """공통 실행 컨텍스트 — df, pd 주입"""
    return {
        "df":          df,
        "pd":          pd,
        "__builtins__": __builtins__,
    }


### 내부 함수: Plotly 실행 ###
def _run_plotly(code: str, df: pd.DataFrame, output_path: Path, output_format: str) -> None:
    """Plotly 코드 실행 후 PNG 또는 HTML로 저장"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly
    except ImportError:
        raise ImportError("plotly가 설치되어 있지 않습니다. uv add plotly를 실행하세요.")

    exec_globals = _make_exec_globals(df)
    exec_globals.update({"px": px, "go": go, "plotly": plotly})

    exec(code, exec_globals)  # noqa: S102

    fig = exec_globals.get("fig")
    if fig is None:
        raise ValueError("Plotly 코드 실행 후 'fig' 변수가 없습니다. 코드에 fig = px.xxx(...) 형태가 있어야 합니다.")

    if output_format == "html":
        fig.write_html(str(output_path))
    else:
        fig.write_image(str(output_path))


### 내부 함수: Seaborn 실행 ###
def _run_seaborn(code: str, df: pd.DataFrame, output_path: Path) -> None:
    """Seaborn 코드 실행 후 PNG로 저장"""
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("seaborn이 설치되어 있지 않습니다. uv add seaborn을 실행하세요.")

    plt.close("all")  # 실행 전 기존 figure 정리

    exec_globals = _make_exec_globals(df)
    exec_globals.update({"sns": sns, "plt": plt})

    exec(code, exec_globals)  # noqa: S102

    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close("all")  # 실행 후 figure 정리


### 내부 함수: Matplotlib 실행 ###
def _run_matplotlib(code: str, df: pd.DataFrame, output_path: Path) -> None:
    """Matplotlib 코드 실행 후 PNG로 저장"""
    plt.close("all")  # 실행 전 기존 figure 정리

    exec_globals = _make_exec_globals(df)
    exec_globals["plt"] = plt

    exec(code, exec_globals)  # noqa: S102

    # fig 변수가 있으면 fig.savefig, 없으면 plt.savefig
    fig = exec_globals.get("fig")
    if fig is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    else:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")

    plt.close("all")  # 실행 후 figure 정리