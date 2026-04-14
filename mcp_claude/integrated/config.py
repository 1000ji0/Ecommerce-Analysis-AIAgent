from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent      # .../mcp_claude/integrated
BASE_DIR = PROJECT_ROOT.parent                     # .../mcp_claude
DATA_DIR = BASE_DIR / "data"                       # .../mcp_claude/data

# 파이프라인 출력 전용 디렉터리 (원본 데이터 덮어쓰기 방지)
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 기본 파일 경로 및 타겟 컬럼 설정
DEFAULT_WRITING_PATH = str(BASE_DIR)

# ---- 입력(원본) 데이터 (DATA_DIR에 두고 사용) ----
DEFAULT_FILE_PATH_FC = str(DATA_DIR / "pivoted_data_sample.csv")
DEFAULT_FILE_PATH_TIME = str(DATA_DIR / "time_info.csv")

# ---- 파이프라인 단계별 출력 파일명 (모두 OUTPUT_DIR에 저장) ----
# 1) Feature Creation
FC_OUTPUT_FILE = "feature_data_common.pickle"
# 2) Outlier Handling
OH_OUTPUT_FILE = "feature_data_outlier.pickle"
# 3) Multicollinearity (smart_correlation, mrmr_selection)
MC_OUTPUT_FILE = "feature_data_multi.pickle"
MC_GROUP_CSV = "group_dict.csv"
# 4) Augmentation
AUG_OUTPUT_FILE = "feature_data_augmentation.pickle"
# 5) Feature Selection (rank_matrix)
FS_RANK_MATRIX_FILE = "rank_matrix_smart_GAN_sample.pickle"
FS_COMMON_COLUMNS_FILE = "common_columns_sample.pickle"
# 6) Weight Update (select_best_model은 results/ 하위에 시각화 등 저장)

# ---- 툴 기본 입력 경로 (파이프라인 순서대로 다음 단계 입력 = 이전 단계 출력) ----
# Multicollinearity / Outlier 입력 등에 사용 (실제로는 OUTPUT_DIR 내 파일 경로 전달)
DEFAULT_FILE_PATH_MC = str(OUTPUT_DIR / FC_OUTPUT_FILE)
# Rank Matrix 입력 = Augmentation 출력
DEFAULT_FILE_PATH_RM = str(OUTPUT_DIR / AUG_OUTPUT_FILE)
# Weight Update용 train/test 파일명 (OUTPUT_DIR에 저장됨)
DEFAULT_WU_TRAIN_FILE = "wu_train_sample.pickle"
DEFAULT_WU_TEST_FILE = "wu_test_sample.pickle"

# 공통 타겟 컬럼 이름
DEFAULT_TARGET_COL = "TARGET"

# 최대 출력 행 수 제한 (임시)
MAX_ROW = 5