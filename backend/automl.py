from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

@dataclass
class AutoMLResult:
    problem_type: str
    best_model_name: str
    metrics: Dict[str, float]
    feature_names: Any
    pipeline: Any
    y_pred: Any

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ]
    )
    return pre

def _metrics_cls(y_true, y_pred_proba, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba[:,1]) if y_pred_proba.shape[1] > 1 else np.nan)
    }

def _metrics_reg(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

def train_automl(df: pd.DataFrame, target: str, *, random_state: int=42) -> AutoMLResult:
    y = df[target]
    X = df.drop(columns=[target])
    problem_type = "classification" if (not pd.api.types.is_float_dtype(y) and y.nunique() <= 20) else "regression"

    pre = _build_preprocessor(X)

    if problem_type == "classification":
        candidates = {
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", eval_metric="logloss", random_state=random_state
            ),
            "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9)
        }
    else:
        candidates = {
            "RandomForest": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
            "XGBoost": XGBRegressor(
                n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", random_state=random_state
            ),
            "LightGBM": LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9)
        }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if problem_type=="classification" else None)

    best_score = -np.inf
    best = None
    best_name = ""
    y_pred_store = None

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        if problem_type == "classification":
            y_pred_proba = pipe.predict_proba(X_te)
            y_pred = (y_pred_proba[:,1] >= 0.5).astype(int) if y_pred_proba.shape[1] == 2 else pipe.predict(X_te)
            metrics = _metrics_cls(y_te, y_pred_proba, y_pred)
            score = metrics["f1"]
        else:
            y_pred = pipe.predict(X_te)
            metrics = _metrics_reg(y_te, y_pred)
            score = metrics["r2"]
        if score > best_score:
            best_score = score
            best = pipe
            best_name = name
            y_pred_store = y_pred

    return AutoMLResult(problem_type, best_name, metrics, X.columns, best, y_pred_store)
