"""
T-18 보고서 생성기
분석 결과 전체를 PDF/CSV 보고서로 자동 생성

라이브러리: weasyprint (HTML → PDF 변환)
- 한국어 완벽 지원 (시스템 폰트 사용)
- HTML/CSS 기반이라 레이아웃 자유도 높음
- ReportLab 대비 폰트 설정 불필요

weasyprint 시스템 의존성:
  macOS:  brew install cairo pango
  Ubuntu: apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0
"""
import csv
import traceback
from datetime import datetime
from pathlib import Path

from config import SESSION_DIR
from tools.output.t20_trace_logger import log_tool_call


###### main 함수: 보고서 생성 ######
def generate_report(
    session_id: str,
    analysis_result: dict,
    output_format: str = "pdf",
) -> dict:
    """
    분석 결과를 보고서 파일로 생성

    Args:
        session_id:      세션 ID
        analysis_result: {
            "summary":         str,
            "insights":        list[str],
            "actions":         list[str],
            "viz_suggestions": list[str],
            "kpi_result":      dict,
            "feature_ranking": dict,
            "image_paths":     list[str],
        }
        output_format: "pdf" | "csv"

    Returns:
        {
            "report_path": str,
            "success":     bool,
            "error":       str | None,
        }
    """
    report_dir = SESSION_DIR / session_id / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename    = f"report_{timestamp}.{output_format}"
    report_path = report_dir / filename

    try:
        if output_format == "pdf":
            _build_pdf(report_path, analysis_result)
        else:
            _build_csv(report_path, analysis_result)

        result = {"report_path": str(report_path), "success": True, "error": None}

    except Exception:
        result = {"report_path": "", "success": False, "error": traceback.format_exc()}

    log_tool_call(session_id, "report_gen", {"format": output_format}, result)
    return result


### 내부 함수: HTML 템플릿 생성 ###
def _build_html(data: dict) -> str:
    """
    보고서 HTML 템플릿 생성
    weasyprint가 이 HTML을 PDF로 변환
    한국어: system-ui, Apple SD Gothic Neo, Malgun Gothic 등 시스템 폰트 사용
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 섹션별 HTML 조각 생성 ──────────────────────────────────────
    summary_html        = _section_summary(data.get("summary", ""))
    insights_html       = _section_list("핵심 인사이트", data.get("insights", []), "insight")
    actions_html        = _section_list("액션 아이템",   data.get("actions",  []), "action")
    viz_html            = _section_list("시각화 제안",   data.get("viz_suggestions", []), "viz")
    kpi_html            = _section_kpi(data.get("kpi_result", {}))
    feature_html        = _section_feature(data.get("feature_ranking", {}))
    images_html         = _section_images(data.get("image_paths", []))

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  /* ── 기본 설정 ── */
  body {{
    font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Nanum Gothic",
                 system-ui, sans-serif;
    font-size: 11pt;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
  }}

  /* ── 페이지 설정 (weasyprint) ── */
  @page {{
    size: A4;
    margin: 2cm 2cm 2.5cm 2cm;
    @bottom-center {{
      content: "DAISY 분석 보고서  |  " counter(page) " / " counter(pages);
      font-size: 8pt;
      color: #888;
    }}
  }}

  /* ── 헤더 ── */
  .header {{
    border-bottom: 2px solid #4A4A8A;
    padding-bottom: 12px;
    margin-bottom: 24px;
  }}
  .header h1 {{
    font-size: 20pt;
    color: #4A4A8A;
    margin: 0 0 4px 0;
  }}
  .header .meta {{
    font-size: 9pt;
    color: #666;
  }}

  /* ── 섹션 ── */
  .section {{
    margin-bottom: 20px;
  }}
  .section h2 {{
    font-size: 13pt;
    color: #4A4A8A;
    border-left: 4px solid #4A4A8A;
    padding-left: 8px;
    margin-bottom: 10px;
  }}
  .summary-box {{
    background: #F5F5FB;
    border-radius: 6px;
    padding: 12px 16px;
    line-height: 1.7;
  }}

  /* ── 리스트 아이템 ── */
  .item-list {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}
  .item-list li {{
    padding: 7px 10px 7px 30px;
    position: relative;
    border-bottom: 0.5px solid #eee;
    line-height: 1.5;
  }}
  .item-list li:last-child {{ border-bottom: none; }}
  .item-list li::before {{
    content: attr(data-num);
    position: absolute;
    left: 8px;
    color: #4A4A8A;
    font-weight: bold;
  }}

  /* ── 테이블 ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
  }}
  th {{
    background: #4A4A8A;
    color: white;
    padding: 7px 10px;
    text-align: left;
  }}
  td {{
    padding: 6px 10px;
    border-bottom: 0.5px solid #ddd;
  }}
  tr:nth-child(even) td {{ background: #F8F8FC; }}

  /* ── 이미지 ── */
  .chart-img {{
    width: 100%;
    max-height: 260px;
    object-fit: contain;
    margin: 8px 0;
    page-break-inside: avoid;
  }}

  /* ── 페이지 나누기 ── */
  .page-break {{ page-break-after: always; }}
</style>
</head>
<body>

<div class="header">
  <h1>DAISY 분석 보고서</h1>
  <div class="meta">생성일시: {generated_at}</div>
</div>

{summary_html}
{insights_html}
{actions_html}
{viz_html}
{kpi_html}
{feature_html}
{images_html}

</body>
</html>"""


### 섹션 생성 헬퍼 ###

def _section_summary(summary: str) -> str:
    if not summary:
        return ""
    return f"""
<div class="section">
  <h2>분석 요약</h2>
  <div class="summary-box">{summary}</div>
</div>"""


def _section_list(title: str, items: list, css_class: str) -> str:
    if not items:
        return ""
    li_tags = "".join(
        f'<li data-num="{i+1}.">{item}</li>'
        for i, item in enumerate(items)
    )
    return f"""
<div class="section">
  <h2>{title}</h2>
  <ul class="item-list {css_class}">{li_tags}</ul>
</div>"""


def _section_kpi(kpi: dict) -> str:
    if not kpi:
        return ""
    rows = "".join(
        f"<tr><td>{k}</td><td>{round(v, 4) if isinstance(v, float) else v}</td></tr>"
        for k, v in kpi.items()
    )
    return f"""
<div class="section">
  <h2>KPI 결과</h2>
  <table>
    <thead><tr><th>지표</th><th>값</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _section_feature(ranking: dict) -> str:
    if not ranking:
        return ""
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{feat}</td><td>{score}</td></tr>"
        for i, (feat, score) in enumerate(list(ranking.items())[:5])
    )
    return f"""
<div class="section">
  <h2>주요 변수 (상위 5)</h2>
  <table>
    <thead><tr><th>순위</th><th>변수명</th><th>중요도</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def _section_images(image_paths: list) -> str:
    if not image_paths:
        return ""
    imgs = ""
    for path in image_paths:
        p = Path(path)
        if p.exists() and p.suffix.lower() == ".png":
            imgs += f'<img class="chart-img" src="{p.resolve()}">'
    if not imgs:
        return ""
    return f"""
<div class="section">
  <h2>시각화</h2>
  {imgs}
</div>"""


### 내부 함수: PDF 생성 ###
def _build_pdf(output_path: Path, data: dict) -> None:
    """HTML 템플릿 → weasyprint → PDF"""
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "weasyprint가 설치되어 있지 않습니다.\n"
            "  uv add weasyprint\n"
            "  macOS:  brew install cairo pango\n"
            "  Ubuntu: apt-get install libcairo2 libpango-1.0-0 libpangocairo-1.0-0"
        )

    html_str = _build_html(data)
    HTML(string=html_str).write_pdf(str(output_path))


### 내부 함수: CSV 생성 ###
def _build_csv(output_path: Path, data: dict) -> None:
    """CSV 보고서 생성"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["항목", "내용"])
        writer.writerow(["생성일시", datetime.now().strftime("%Y-%m-%d %H:%M")])
        writer.writerow([])

        if summary := data.get("summary"):
            writer.writerow(["[요약]", summary])
            writer.writerow([])

        if insights := data.get("insights"):
            writer.writerow(["[인사이트]"])
            for i, ins in enumerate(insights, 1):
                writer.writerow([f"{i}.", ins])
            writer.writerow([])

        if actions := data.get("actions"):
            writer.writerow(["[액션 아이템]"])
            for i, act in enumerate(actions, 1):
                writer.writerow([f"{i}.", act])
            writer.writerow([])

        if kpi := data.get("kpi_result"):
            writer.writerow(["[KPI]"])
            writer.writerow(["지표", "값"])
            for k, v in kpi.items():
                writer.writerow([k, v])
            writer.writerow([])

        if ranking := data.get("feature_ranking"):
            writer.writerow(["[변수 중요도]"])
            writer.writerow(["순위", "변수명", "중요도"])
            for i, (feat, score) in enumerate(list(ranking.items())[:5], 1):
                writer.writerow([i, feat, score])