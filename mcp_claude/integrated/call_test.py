import asyncio
import os

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    DEFAULT_FILE_PATH_FC,
    DEFAULT_FILE_PATH_TIME,
    DEFAULT_WU_TRAIN_FILE,
    DEFAULT_WU_TEST_FILE,
    DEFAULT_TARGET_COL,
    FC_OUTPUT_FILE,
    OH_OUTPUT_FILE,
    MC_OUTPUT_FILE,
    AUG_OUTPUT_FILE,
)
SSE_URL = "http://127.0.0.1:8000/sse"

# 파이프라인 순서: 출력은 모두 OUTPUT_DIR에 저장되므로 다음 단계 입력 = OUTPUT_DIR / 해당 출력 파일
def _out(path_dir, filename):
    return os.path.join(str(path_dir), filename)


async def main():
    # 1) SSE로 MCP 서버에 연결해서 read/write 스트림 확보
    async with sse_client(SSE_URL) as (read_stream, write_stream):
        # 2) 스트림으로 ClientSession 생성
        async with ClientSession(read_stream, write_stream) as session:
            # 3) MCP 프로토콜 초기화
            await session.initialize()

            # 4) 등록된 MCP 툴 목록 조회
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}

            print("=== TOOLS ON SERVER ===")
            for t in tools.tools:
                print("-", t.name)

            # 5) 각 MCP 툴별 테스트 케이스 (파이프라인 순서: 입력은 이전 단계 출력 경로)
            data_path_for_weight_update = str(OUTPUT_DIR) + os.sep

            tests = [
                {
                    "name": "implement_fc",
                    "arguments": {
                        "data_dir": DEFAULT_FILE_PATH_FC,
                        "exec_tools": ["fpca", "nds", "timeseries", "stats"],
                    },
                },
                {
                    "name": "delete_outlier",
                    "arguments": {
                        "data_path": _out(OUTPUT_DIR, FC_OUTPUT_FILE),
                        "target": DEFAULT_TARGET_COL,
                        "outlier_method": "gaussian",
                    },
                },
                {
                    "name": "smart_correlation",
                    "arguments": {
                        "data_path": _out(OUTPUT_DIR, OH_OUTPUT_FILE),
                        "target_col": DEFAULT_TARGET_COL,
                        "threshold": 0.8,
                        "method": "pearson",
                        "variables": None,
                        "missing_values": "raise",
                        "selection_method": "variance",
                        "estimator": None,
                    },
                },
                {
                    "name": "mrmr_selection",
                    "arguments": {
                        "data_path": _out(OUTPUT_DIR, OH_OUTPUT_FILE),
                        "target_col": DEFAULT_TARGET_COL,
                        "regression": True,
                        "scoring": "neg_mean_squared_error",
                        "method": "RFCQ",
                    },
                },
                {
                    "name": "gaussian_augmentation",
                    "arguments": {
                        "data_path": _out(OUTPUT_DIR, MC_OUTPUT_FILE),
                        "time_path": DEFAULT_FILE_PATH_TIME,
                        "ycol": DEFAULT_TARGET_COL,
                        "split_data_count": 3,
                        "split_method": "time",
                        "n_new_samples": 300,
                    },
                },
                {
                    "name": "rank_matrix",
                    "arguments": {
                        "data_path": _out(OUTPUT_DIR, AUG_OUTPUT_FILE),
                        "fold": 1,
                    },
                },
                {
                    "name": "select_best_model",
                    "arguments": {
                        "data_path": data_path_for_weight_update,
                        "train": DEFAULT_WU_TRAIN_FILE,
                        "test": DEFAULT_WU_TEST_FILE,
                        "ncol": 10,
                        "criterion": "MSE",
                    },
                },
            ]

            print("\n=== RUNNING TOOL TESTS ===")
            # 일부 툴만 테스트 시 test_tools 리스트를 수정하세요.
            test_tools = ["implement_fc", "delete_outlier", "smart_correlation", "mrmr_selection", "gaussian_augmentation", "rank_matrix", "select_best_model"]
            tests = [test for test in tests if test["name"] in test_tools]
            for test in tests:
                name = test["name"]
                args = test["arguments"]

                if name not in tool_names:
                    print(f"\n[SKIP] Tool '{name}' is not registered on server.")
                    continue

                print(f"\n[CALL] {name} with arguments: {args}")
                try:
                    result = await session.call_tool(name, arguments=args)
                    print("[RESULT]", result)
                except Exception as exc:
                    print(f"[ERROR] {name} raised exception: {exc}")


if __name__ == "__main__":
    asyncio.run(main())