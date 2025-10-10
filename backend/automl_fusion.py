# === IMPORTS & CONFIG ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import os, json, warnings
import numpy as np
import pandas as pd
import optuna
import mlflow

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import f1_score, r2_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.utils.validation import check_is_fitted

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")


# === RESULT DATACLASS ===
@dataclass
class FusionResult:
    problem_type: str
    metric_name: str
    best_score: float
    leaderboard: List[Tuple[str, float]]
    blend_weights: Dict[str, float] | None
    model: Any  # final fitted ensemble (VotingClassifier/Regressor)


# === HELPERS ===
def _infer_type(y: pd.Series) -> str:
    """Heurystyka: mała liczba unikalnych wartości → klasyfikacja."""
    if pd.api.types.is_float_dtype(y) and y.nunique(dropna=True) > 50:
        return "regression"
    return "classification" if y.nunique(dropna=True) <= 50 else "regression"


def _splitter(y: pd.Series, seed: int):
    return (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        if _infer_type(y) == "classification"
        else KFold(n_splits=5, shuffle=True, random_state=seed)
    )


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    # Uwaga: w sklearn>=1.2 OneHotEncoder ma parametr sparse_output
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # kompatybilność
    return ColumnTransformer(
        [("num", StandardScaler(with_mean=False), num), ("cat", ohe, cat)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _metric(is_cls: bool):
    return (
        ("f1", make_scorer(f1_score, average="weighted"))
        if is_cls
        else ("r2", "r2")
    )


def _make_pipeline(is_cls: bool, pre: ColumnTransformer, model):
    # SMOTE tylko dla klasyfikacji – wewnątrz CV (bez przecieku)
    if is_cls:
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("model", model)])
    return Pipeline([("pre", pre), ("model", model)])


def _candidate_models(trial: optuna.Trial, is_cls: bool):
    models = []
    if is_cls:
        models += [
            (
                "XGB",
                XGBClassifier(
                    n_estimators=trial.suggest_int("xgb_n", 300, 900),
                    max_depth=trial.suggest_int("xgb_depth", 3, 10),
                    learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("xgb_sub", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("xgb_col", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("xgb_l2", 1e-3, 10.0, log=True),
                    tree_method="hist",
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
            (
                "LGBM",
                LGBMClassifier(
                    n_estimators=trial.suggest_int("lgbm_n", 300, 900),
                    num_leaves=trial.suggest_int("lgbm_leaves", 16, 128, log=True),
                    learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("lgbm_sub", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("lgbm_col", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("lgbm_l2", 1e-3, 10.0, log=True),
                    random_state=42,
                ),
            ),
            (
                "CAT",
                CatBoostClassifier(
                    iterations=trial.suggest_int("cat_n", 300, 900),
                    depth=trial.suggest_int("cat_depth", 4, 10),
                    learning_rate=trial.suggest_float("cat_lr", 0.01, 0.3, log=True),
                    l2_leaf_reg=trial.suggest_float("cat_l2", 1.0, 10.0),
                    verbose=False,
                    random_seed=42,
                ),
            ),
        ]
    else:
        models += [
            (
                "XGB",
                XGBRegressor(
                    n_estimators=trial.suggest_int("xgb_n", 300, 900),
                    max_depth=trial.suggest_int("xgb_depth", 3, 10),
                    learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("xgb_sub", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("xgb_col", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("xgb_l2", 1e-3, 10.0, log=True),
                    tree_method="hist",
                    random_state=42,
                ),
            ),
            (
                "LGBM",
                LGBMRegressor(
                    n_estimators=trial.suggest_int("lgbm_n", 300, 900),
                    num_leaves=trial.suggest_int("lgbm_leaves", 16, 128, log=True),
                    learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("lgbm_sub", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("lgbm_col", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("lgbm_l2", 1e-3, 10.0, log=True),
                    random_state=42,
                ),
            ),
            (
                "CAT",
                CatBoostRegressor(
                    iterations=trial.suggest_int("cat_n", 300, 900),
                    depth=trial.suggest_int("cat_depth", 4, 10),
                    learning_rate=trial.suggest_float("cat_lr", 0.01, 0.3, log=True),
                    l2_leaf_reg=trial.suggest_float("cat_l2", 1.0, 10.0),
                    verbose=False,
                    random_seed=42,
                ),
            ),
        ]
    return models


# === MAIN TRAIN ===
def train_fusion(df: pd.DataFrame, target: str, *, seed: int = 42, trials: int = 30) -> FusionResult:
    """
    Trenuje AutoML FUSION (CV + Optuna) i zwraca zblendowany model top-K.
    Artefakty i metryki są logowane do MLflow (jeśli MLFLOW_TRACKING_URI ustawione).
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame columns")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:mlruns"))

    y = df[target]
    X = df.drop(columns=[target])
    problem = _infer_type(y)
    is_cls = problem == "classification"
    pre = _build_preprocessor(X)
    cv = _splitter(y, seed)
    metric_name, scorer = _metric(is_cls)

    # Akumulacja wyników modeli w ramach wszystkich prób
    scores_map: Dict[str, List[float]] = {}

    def objective(trial: optuna.Trial):
        best_local = -1e9
        for name, model in _candidate_models(trial, is_cls):
            try:
                pipe = _make_pipeline(is_cls, pre, model)
                scores = cross_val_score(
                    pipe, X, y,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1,
                    error_score="raise"
                )
                score = float(np.mean(scores))
                trial.set_user_attr(name, score)
                scores_map.setdefault(name, []).append(score)
                best_local = max(best_local, score)
            except Exception as e:
                # Zanotuj niepowodzenie tego kandydata w tej próbie; nie przerywaj optymalizacji
                trial.set_user_attr(f"{name}_error", str(e))
        return best_local

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    if not scores_map:
        raise RuntimeError("No successful model evaluations; check data & dependencies")

    # Leaderboard – średnia po wszystkich próbach
    leaderboard = sorted(
        [(k, float(np.mean(v))) for k, v in scores_map.items() if len(v) > 0],
        key=lambda x: x[1],
        reverse=True
    )
    # Top-K do blendu
    topk = leaderboard[:3]
    names = [n for n, _ in topk]
    means = [m for _, m in topk]
    denom = float(sum(means)) + 1e-9
    weights = [float(m / denom) for m in means]

    # Złóż finalny ensemble
    est = []
    for name in names:
        if is_cls:
            if name == "XGB":
                m = XGBClassifier(n_estimators=600, tree_method="hist", random_state=seed)
            elif name == "LGBM":
                m = LGBMClassifier(n_estimators=700, random_state=seed)
            else:
                m = CatBoostClassifier(iterations=600, verbose=False, random_seed=seed)
        else:
            if name == "XGB":
                m = XGBRegressor(n_estimators=800, tree_method="hist", random_state=seed)
            elif name == "LGBM":
                m = LGBMRegressor(n_estimators=800, random_state=seed)
            else:
                m = CatBoostRegressor(iterations=700, verbose=False, random_seed=seed)
        est.append((name, _make_pipeline(is_cls, _build_preprocessor(X), m)))

    final = (
        VotingClassifier(estimators=est, voting="soft", weights=weights)
        if is_cls else
        VotingRegressor(estimators=est, weights=weights)
    )
    final.fit(X, y)

    # MLflow logging
    with mlflow.start_run(nested=True):
        mlflow.log_params({"problem": problem, "top_models": ",".join(names)})
        mlflow.log_metrics({metric_name: float(max(means))})
        # artefakty – leaderboard + wagi
        mlflow.log_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), artifact_file="leaderboard.json")
        mlflow.log_text(json.dumps({n: w for n, w in zip(names, weights)}, ensure_ascii=False, indent=2), artifact_file="blend_weights.json")

    return FusionResult(
        problem_type=problem,
        metric_name=metric_name,
        best_score=float(max(means)),
        leaderboard=leaderboard,
        blend_weights={n: w for n, w in zip(names, weights)},
        model=final,
    )
