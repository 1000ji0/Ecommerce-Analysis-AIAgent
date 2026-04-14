# mcp_claude

MCP(Model Context Protocol) 서버 및 Feature Engineering 관련 에이전트 코드가 위치한 디렉터리입니다.

## 디렉터리 구조

```
mcp_claude/
├── README.md             # 본 문서
└── integrated/            # 최종 MCP 서버 (웹 UI 전달용)
    ├── server.py             # MCP 서버 진입점
    ├── config.py             # 경로·기본값 설정 (data/ 파일 경로)
    ├── call_test.py          # MCP 툴 로컬 호출 테스트 스크립트
    └── agents/               # MCP 툴 등록 및 로직
        ├── __init__.py       # register_all() — *tools 모듈 자동 스캔
        ├── common/           # session, logging, tool_registry, debug
        ├── feature/
        │   ├── feature_selection/
        │   │   └── tools.py  # rank_matrix MCP 툴
        │   ├── weight_update/
        │   │   └── tools.py  # select_best_model MCP 툴
        │   └── utils/        # feature 관련 SBF_*, fLasso, utils_model 등
        └── data_cleansing/
            ├── augmentation/
            │   └── tools.py  # gaussian_augmentation MCP 툴
            ├── feature_creation/
            │   └── tools.py  # implement_fc MCP 툴
            ├── multicollinearity/
            │   └── tools.py  # smart_correlation, mrmr_selection MCP 툴
            └── outlier_handling/
                └── tools.py  # delete_outlier MCP 툴
```

- **integrated**: 실제 MCP 서버. `agents/` 아래의 `*tools` 모듈에 `register(mcp)`가 있으면 자동으로 툴이 등록됩니다.

## 실행 방법

MCP 서버는 **integrated** 디렉터리를 작업 디렉터리로 두고 실행합니다.  
의존성은 프로젝트 루트의 `requirements-clean.txt`로 설치합니다.

### 1. 의존성 설치 (프로젝트 루트에서)

```bash
pip install -r requirements-clean.txt
```

### 2. MCP 서버 실행

프로젝트 가상환경을 활성화한 뒤, `integrated` 디렉터리에서 실행합니다.

```bash
cd mcp_claude/integrated
python server.py
```

- **전송 방식**: SSE (Server-Sent Events)
- **호스트/포트**: `127.0.0.1:8000` (server.py 내 설정)
- **허용 호스트**: `fe-opt-mcp.co.kr`, `www.fe-opt-mcp.co.kr`, `127.0.0.1:8000`, `localhost:8000`

프로젝트 루트에서 실행하려면:

```bash
python mcp_claude/integrated/server.py
```

이때 현재 작업 디렉터리가 `mcp_claude/integrated`가 아니므로, `config.py` 등에서 상대 경로를 쓰는 부분이 있다면 실행 디렉터리에 따라 경로가 달라질 수 있습니다. **일반적으로는 `cd mcp_claude/integrated` 후 `python server.py`로 실행하는 것을 권장합니다.**

### 3. MCP 툴 테스트 (call_test.py)

`integrated/call_test.py`는 **로컬에서 MCP 서버에 SSE로 연결해 등록된 툴 목록을 조회하고, 각 툴을 순서대로 호출하는 테스트 스크립트**입니다. `config.py`의 `DATA_DIR`, `DEFAULT_FILE_PATH_*` 등을 import 해서 툴 인자로 사용합니다.

**실행 방법**

1. MCP 서버가 `127.0.0.1:8000`에서 이미 실행 중이어야 합니다.
2. `mcp_claude/integrated` 디렉터리에서 실행합니다.

```bash
cd mcp_claude/integrated
python call_test.py
```

- 스크립트 내부에서 `test_tools` 리스트를 수정하면 **특정 툴만** 테스트할 수 있습니다.
- 툴별 인자(예: `data_path`, `train`, `test`)는 `config`에서 가져오므로, **데이터 경로를 바꾸려면 `config.py`만 수정**하면 됩니다.

## 2. 분석 파이프라인 (Pipeline Stages)

### 전체 분석 순서

```
Stage 1: 변수 생성
   ↓
Stage 2: 데이터 전처리 (순서 무관)
   ├─ 이상치 제거
   ├─ 다중공선성 제거
   └─ 데이터 증강
   ↓
Stage 3: 변수 다면평가 (여러 모델 기반 변수 중요도 도출)
   ↓
Stage 4: 최종 모델링 및 변수 선정
```

| Stage | MCP 툴 | 설명 |
|-------|--------|------|
| **1. 변수 생성** | `implement_fc` | 시계열/원시 데이터에서 FPCA, NDS, timeseries, stats 등으로 피처 생성. |
| **2. 데이터 전처리** | `delete_outlier` | 이상치 제거. |
| | `smart_correlation`, `mrmr_selection` | 다중공선성 제거·변수 선택. |
| | `gaussian_augmentation` | AE-GAN 기반 데이터 증강. |
| **3. 변수 다면평가** | `rank_matrix` | 선형/비선형/비모수 모델로 변수 중요도 계산, rank matrix 생성. |
| **4. 최종 모델링** | `select_best_model` | 최적 모델·변수 수 선택, 예측값·시각화 반환. |

## MCP 도구 설명

서버에 등록되는 주요 MCP 툴과 인자 정보는 다음과 같습니다.

| 구분 | 툴 이름 | 인자 | 설명 |
|------|---------|------|------|
| **Feature Selection** | `rank_matrix` | `data_path: str`, `fold: int = 1` | 입력 데이터에 대해 선형/비선형/비모수 모델로 변수 중요도를 계산하고, k-fold 평균으로 rank matrix를 생성합니다. |
| **Weight Update** | `select_best_model` | `data_path: str`, `train: str`, `test: str`, `ncol: int = 10`, `criterion: str = 'MSE'` | 여러 가중치·모델 학습 결과 중 최적 모델을 선택하고, 예측값·rank matrix 및 true vs predicted scatter plot을 반환합니다. |
| **Multicollinearity** | `smart_correlation` | `data_path: str`, `target_col: str = 'TARGET'`, `threshold: float = 0.8`, `method: str = 'pearson'`, `variables: list = None`, `missing_values: str = 'raise'`, `selection_method: str = 'variance'`, `estimator = None` | 상관성이 높은 변수들을 그룹화하고 대표 변수를 선택하여 다중공선성을 줄입니다. |
| **Multicollinearity** | `mrmr_selection` | `data_path: str`, `target_col: str = 'TARGET'`, `regression: bool = True`, `scoring: str = 'neg_mean_squared_error'`, `method: str = 'RFCQ'` | MRMR(Maximum Relevance Minimum Redundancy)을 사용하여 목표 변수와 상관성이 크고 중복성이 낮은 변수를 선택합니다. |
| **Feature Creation** | `implement_fc` | `data_dir: str`, `exec_tools: list = ['fpca','nds','timeseries','stats']` | FPCA, NDS, timeseries, stats 등 사용자가 선택한 방법으로 시계열·통계 피처를 생성합니다. |
| **Augmentation** | `gaussian_augmentation` | `data_path: str = None`, `time_path: str = None`, `ycol: str = 'TARGET'`, `split_data_count: int = 3`, `split_method: str = 'time'`, `n_new_samples: int = 300` | AE-GAN 기반 latent space에서 가우시안 분포로 데이터를 생성하고, wasserstein distance 기반 필터링을 통해 증강 데이터를 생성합니다. |
| **Outlier Handling** | `delete_outlier` | `data_path: str`, `target: str`, `outlier_method: str = 'gaussian'` | 지정 컬럼에 대해 gaussian 또는 IQR 기준으로 이상치를 탐지·제거한 데이터셋을 반환합니다. |

그 외 `agents/common/debug`에 등록되는 디버그용 툴이 있을 수 있습니다.

### MCP 도구별 사용 파일 (코드 기준)

모든 툴의 **출력은 `OUTPUT_DIR`(`data/output/`)에만 저장**되며, **원본 데이터(`DATA_DIR`)는 덮어쓰지 않습니다.**  
입력은 파이프라인 순서대로 이전 단계 출력 파일을 사용합니다.

| MCP 툴 | 입력(사용) 데이터 | 출력 파일 (저장 위치: `OUTPUT_DIR`) | config 상수 |
|--------|-------------------|--------------------------------------|-------------|
| **implement_fc** | `pivoted_data_sample.csv` (CSV 경로) | `feature_data_common.pickle` | `DEFAULT_FILE_PATH_FC` → `FC_OUTPUT_FILE` |
| **delete_outlier** | `feature_data_common.pickle` | `feature_data_outlier.pickle` | `OH_OUTPUT_FILE` |
| **smart_correlation** | `feature_data_outlier.pickle` | `feature_data_multi.pickle`, `group_dict.csv` | `MC_OUTPUT_FILE`, `MC_GROUP_CSV` |
| **mrmr_selection** | `feature_data_outlier.pickle` | `feature_data_multi.pickle` | `MC_OUTPUT_FILE` |
| **gaussian_augmentation** | `feature_data_multi.pickle` | `feature_data_augmentation.pickle`, `wu_train_sample.pickle`, `wu_test_sample.pickle` | `AUG_OUTPUT_FILE`, `DEFAULT_WU_*_FILE` |
| **rank_matrix** | `feature_data_augmentation.pickle` | `common_columns_sample.pickle`, `rank_matrix_smart_GAN_sample.pickle` | `FS_RANK_MATRIX_FILE`, `FS_COMMON_COLUMNS_FILE` |
| **select_best_model** | 디렉터리 경로 + `wu_train_sample.pickle`, `wu_test_sample.pickle`, `common_columns_sample.pickle`, `rank_matrix_smart_GAN_sample.pickle` | `results/{ncol}_scatter.png` 등 (동일 디렉터리 내 `results/`) | `OUTPUT_DIR`, `DEFAULT_WU_*_FILE` |

- **원본 데이터**: `DATA_DIR`에는 `pivoted_data_sample.csv`, `time_info.csv` 등만 두고, 툴 출력은 모두 `OUTPUT_DIR`로만 저장됩니다.
- **공통 타겟 컬럼**: `DEFAULT_TARGET_COL` (기본값 `TARGET`).
- **시간 정보**: Augmentation 입력 — `DEFAULT_FILE_PATH_TIME` (`time_info.csv`, `DATA_DIR`에 위치).

## 설정 (config.py)

`mcp_claude/integrated/config.py`에서 데이터 경로와 출력 파일명을 관리합니다.

| 변수 | 의미 | 예시 (기본값) |
|------|------|-------------------------------|
| `PROJECT_ROOT` | integrated 디렉터리 | `.../mcp_claude/integrated` |
| `BASE_DIR` | mcp_claude 디렉터리 | `.../mcp_claude` |
| `DATA_DIR` | 원본/입력 데이터 디렉터리 | `.../mcp_claude/data` |
| **`OUTPUT_DIR`** | **파이프라인 출력 전용 디렉터리 (원본 덮어쓰기 방지)** | `.../mcp_claude/data/output` |
| `DEFAULT_FILE_PATH_FC` | Feature Creation 입력 CSV | `DATA_DIR / "pivoted_data_sample.csv"` |
| `DEFAULT_FILE_PATH_TIME` | 시간 정보 CSV (Augmentation) | `DATA_DIR / "time_info.csv"` |
| `FC_OUTPUT_FILE` | Feature Creation 출력 파일명 | `"feature_data_common.pickle"` |
| `OH_OUTPUT_FILE` | Outlier Handling 출력 파일명 | `"feature_data_outlier.pickle"` |
| `MC_OUTPUT_FILE` | Multicollinearity 출력 파일명 | `"feature_data_multi.pickle"` |
| `MC_GROUP_CSV` | smart_correlation 그룹 정보 CSV | `"group_dict.csv"` |
| `AUG_OUTPUT_FILE` | Augmentation 피처 출력 파일명 | `"feature_data_augmentation.pickle"` |
| `FS_RANK_MATRIX_FILE` | Rank Matrix 출력 파일명 | `"rank_matrix_smart_GAN_sample.pickle"` |
| `FS_COMMON_COLUMNS_FILE` | Rank Matrix 공통 컬럼 파일명 | `"common_columns_sample.pickle"` |
| `DEFAULT_WU_TRAIN_FILE` | Weight Update 학습 데이터 파일명 | `"wu_train_sample.pickle"` |
| `DEFAULT_WU_TEST_FILE` | Weight Update 테스트 데이터 파일명 | `"wu_test_sample.pickle"` |
| `DEFAULT_TARGET_COL` | 공통 타겟 컬럼 이름 | `"TARGET"` |
| `MAX_ROW` | 최대 출력 행 수 (임시) | `5` |

- **call_test.py**는 위 상수와 `OUTPUT_DIR`를 사용해 파이프라인 순서대로 툴 인자를 채웁니다. 경로/파일명 변경은 **config.py만 수정**하면 됩니다.

## 툴 등록 방식

- `agents/` 아래에서 패키지 이름이 `*.tools`로 끝나는 모듈을 자동으로 찾습니다.
- 해당 모듈에 `register(mcp: FastMCP)` 함수가 있으면 호출되고, 그 안에서 `mcp.tool()(함수)`로 툴을 등록합니다.
- 새 도구를 넣으려면 해당 기능 폴더에 `tools.py`를 두고 `register(mcp)`를 구현하면 됩니다.
