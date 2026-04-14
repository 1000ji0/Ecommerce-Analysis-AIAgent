from mcp.server.fastmcp import FastMCP
# from fastmcp import FastMCP
from agents import register_all
from agents.common.logging import setup_logging, log_registered_tools
from agents.common.tool_registry import dump_tools_from_mcp
from mcp.server.transport_security import TransportSecuritySettings
import logging
import os

print("### SERVER BOOT MARK: 2026-01-06 A ###", flush=True)
print("CWD:", os.getcwd())





transport_security = TransportSecuritySettings(
    allowed_hosts=[
        "fe-opt-mcp.co.kr",
        "www.fe-opt-mcp.co.kr",
        "127.0.0.1:8000",
        "localhost:8000",
    ],
    # 보통은 비워도 되는데, 필요하면 추가:
    allowed_origins=[
        "https://fe-opt-mcp.co.kr",
        "https://www.fe-opt-mcp.co.kr",
    ],
)

mcp = FastMCP(
    name="analytics_mcp",
    transport_security=transport_security,
    host="127.0.0.1",
    port=8000,
    instructions=(
        "규칙:\n"
        "1) mcp서버의 모든 툴들은 mcp서버 환경에서 수행한다."
        "2) 소스코드를 수정하지 않음"
        "3) 사용자의 의도에 맞는 tool을 실행한다.\n"
        "4) 입력할 파라미터가 없으면 기본값을 적극 활용한다!\n"
        "5) 모호하면 사용자가 툴을 선택할 수 있게 질문한다.\n"
        "6) 실행 이력이 있더라도 다시 수행\n"
        "7) 툴 실행 결과에서 'session_dir' 경로를 확인한다.\n"
        "8) session_dir에서 'metadata.json'을 먼저 읽어 요약을 파악한다.\n"
        "9) 반환된 데이터 경로의 파일(pkl, csv 등)을 읽어 상세 분석을 수행한다.\n"
        "10) 분석 결과를 사용자에게 명확하게 설명한다.\n"
        "11) 수행 시간을 마지막에 사용자에게 알려준다.\n"
    ),
)

register_all(mcp)
logger = setup_logging()

REGISTERED_TOOLS: list[str] = []
def track_tool(func):
    tool_func = mcp.tool()(func)
    REGISTERED_TOOLS.append(func.__name__)
    return tool_func

if __name__ == "__main__":
    mcp.run(transport="sse")


