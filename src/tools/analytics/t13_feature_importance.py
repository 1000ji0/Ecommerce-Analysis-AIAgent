"""
T-13 Feature Importance Tool
상관분석 + LightGBM 기반 변수 중요도 분석 및 자연어 설명
단계: 1) 상관분석, 2) 트리 모델 기반 중요도 분석 - LightGBM (없으면 RandomForest), 3) 정규화 점수 기반 통합 순위, 4) LLM 자연어 설명
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
 
from config import GEMINI_MODEL, GOOGLE_API_KEY
from tools.output.t20_trace_logger import log_tool_call
 
_llm = None
 
 
### 내부 함수: LLM 인스턴스 생성 (lazy initialization) ###
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm
 
 
###### main 함수: 변수 중요도 분석 ######
def analyze_importance(
    session_id: str,
    df: pd.DataFrame,
    target_col: str = "TARGET",
    top_n: int = 5,
) -> dict:
    """
    변수 중요도 분석 (5단계)
    1. 타겟 타입 자동 감지 (regression / classification)
    2. 상관분석 (Pearson + Spearman 최대값)
    3. 5-fold 교차검증 기반 Tree 중요도
    4. Borda Count 통합 순위
    5. LLM 자연어 설명
 
    Returns:
        {
            "task":          str,   # "regression" | "classification"
            "corr_ranking":  dict,  # 상관분석 기반 순위 (상위 top_n)
            "lgbm_ranking":  dict,  # Tree 기반 순위 (5-fold 평균, 상위 top_n)
            "final_ranking": dict,  # Borda Count 통합 순위 (상위 top_n)
            "explanation":   str,   # 자연어 설명
            "valid_rows":    int,   # 실제 학습에 사용된 행 수
        }
    """
    if target_col not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_col}'이 데이터에 없습니다.")
 
    # 범주형 인코딩 후 수치형 추출
    df_encoded = _encode_categoricals(df)
    X = df_encoded.drop(columns=[target_col]).select_dtypes(include="number")
    y = df_encoded[target_col]
 
    if X.empty:
        raise ValueError("수치형 피처가 없습니다. 범주형 인코딩 후에도 피처가 없습니다.")
 
    # NaN 제거
    valid_idx = y.notna() & X.notna().all(axis=1)
    X, y = X[valid_idx], y[valid_idx]
 
    if len(X) < 10:
        raise ValueError(f"유효 데이터가 너무 적습니다: {len(X)}행 (최소 10행 필요)")
 
    # 1단계: 타겟 타입 감지
    task = _detect_task(y)
 
    # 2단계: 상관분석 (Pearson + Spearman 최대값)
    corr_ranking = _corr_importance(X, y, top_n)
 
    # 3단계: 5-fold 교차검증 Tree 중요도
    lgbm_ranking = _tree_importance(X, y, top_n, task)
 
    # 4단계: Borda Count 통합 순위
    final_ranking = _borda_merge(corr_ranking, lgbm_ranking, top_n)
 
    # 5단계: LLM 자연어 설명
    explanation = _explain(final_ranking, corr_ranking, target_col, task)
 
    result = {
        "task":          task,
        "corr_ranking":  corr_ranking,
        "lgbm_ranking":  lgbm_ranking,
        "final_ranking": final_ranking,
        "explanation":   explanation,
        "valid_rows":    int(len(X)),
    }
    log_tool_call(
        session_id, "feature_importance",
        {"target_col": target_col, "top_n": top_n, "task": task}, result,
    )
    return result
 
 
### 내부 함수: 타겟 타입 자동 감지 ###
def _detect_task(y: pd.Series) -> str:
    """
    연속형이면 'regression', 분류형이면 'classification'
    고유값 10개 이하 또는 고유값 비율 5% 미만이면 분류로 판단
    """
    unique_ratio = y.nunique() / len(y)
    if y.nunique() <= 10 or unique_ratio < 0.05:
        return "classification"
    return "regression"
 
 
### 내부 함수: 범주형 변수 원-핫 인코딩 ###
def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    object/category 타입 컬럼을 원-핫 인코딩
    범주형 변수를 분석 대상에 포함하기 위함
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)
 
 
### 내부 함수: 상관분석 (Pearson + Spearman) ###
def _corr_importance(X: pd.DataFrame, y: pd.Series, top_n: int) -> dict:
    """
    Pearson (선형 관계) + Spearman (비선형 관계) 동시 계산
    각 피처별 두 상관계수 중 최대값 사용
    """
    pearson  = X.corrwith(y, method="pearson").abs()
    spearman = X.corrwith(y, method="spearman").abs()
 
    # 피처별 두 방법 중 더 높은 상관계수 선택
    corr = pd.concat([pearson, spearman], axis=1).max(axis=1)
 
    return (
        corr.sort_values(ascending=False)
        .head(top_n)
        .round(4)
        .to_dict()
    )
 
 
### 내부 함수: 5-fold 교차검증 Tree 중요도 ###
def _tree_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int,
    task: str,
) -> dict:
    """
    5-fold 교차검증으로 각 fold의 feature_importances_ 평균 계산
    task에 따라 Regressor / Classifier 자동 선택
    """
    model_fn = _get_model_fn(task)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    importance_list = []
 
    for train_idx, _ in kf.split(X):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        importance_list.append(model.feature_importances_)
 
    # 5-fold 평균 중요도
    avg_importance = np.mean(importance_list, axis=0)
 
    return (
        pd.Series(avg_importance, index=X.columns)
        .sort_values(ascending=False)
        .head(top_n)
        .round(4)
        .to_dict()
    )
 
 
def _get_model_fn(task: str):
    """task에 따라 모델 생성 함수 반환"""
    try:
        if task == "classification":
            from lightgbm import LGBMClassifier
            return lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        else:
            from lightgbm import LGBMRegressor
            return lambda: LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    except ImportError:
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return lambda: RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            return lambda: RandomForestRegressor(n_estimators=100, random_state=42)
 
 
### 내부 함수: Borda Count 통합 순위 ###
def _borda_merge(corr: dict, lgbm: dict, top_n: int) -> dict:
    """
    Borda Count 방식으로 두 순위 통합
    - 각 방법에서 순위별 점수 부여 (1위 = n점, 꼴찌 = 1점)
    - 두 방법의 점수 합산 → 최종 순위
    - 스케일 차이에 robust (상관계수 vs tree importance 단위 다름)
    """
    all_features = list(set(corr) | set(lgbm))
    n = len(all_features)
    scores = {feat: 0 for feat in all_features}
 
    # corr 순위 점수
    for i, feat in enumerate(sorted(corr, key=corr.get, reverse=True)):
        scores[feat] += (n - i)
 
    # lgbm 순위 점수
    for i, feat in enumerate(sorted(lgbm, key=lgbm.get, reverse=True)):
        scores[feat] += (n - i)
 
    sorted_feats = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    return {feat: scores[feat] for feat in sorted_feats}
 
 
### 내부 함수: LLM 자연어 설명 ###
def _explain(
    final_ranking: dict,
    corr_ranking: dict,
    target_col: str,
    task: str,
) -> str:
    """
    통합 순위 + 상관계수 + task 정보를 함께 전달해 구체적인 설명 생성
    """
    llm = _get_llm()
    task_kor = "분류" if task == "classification" else "회귀"
    system = (
        f"데이터 분석가로서 변수 중요도 결과를 한국어로 설명해라.\n"
        f"분석 유형은 {task_kor} 문제다.\n"
        "비전문가도 이해할 수 있게 2~3문장으로 핵심만 설명해라.\n"
        "상관계수 값을 구체적으로 언급해서 설명해라."
    )
    msg = (
        f"타겟 변수: {target_col}\n"
        f"통합 중요도 순위 (Borda Count): {final_ranking}\n"
        f"상관계수 참고 (Pearson/Spearman 최대값): {corr_ranking}"
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=msg)])
    return response.content.strip()
 