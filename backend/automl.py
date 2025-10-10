# === AUTOML_BASELINE_PRO+++ ===
# Kontekst biznesowy:
# Szybki, stabilny baseline AutoML dla klasyfikacji (binary/multiclass) i regresji.
# - Preprocessing: ColumnTransformer (StandardScaler + OneHotEncoder(handle_unknown="ignore"))
# - Kandydaci: RandomForest + (XGBoost, LightGBM), z konfiguracją rozsądną na start
# - Wynik: najlepszy pipeline + metryki + (dla binarki) optymalny próg i y_pred_proba
# - Defensywność: kompatybilność z różnymi wersjami OneHotEncoder (sparse_output), bez twardych zależności
# - Gotowy do integracji z Twoim UI i raportowaniem

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Lightweight boosted trees (opcjonalnie – jeśli brak, moduł działa z RF)
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore
    LGBMRegressor = None  # type: ignore

# (opcjonalnie) loguru
try:
    from loguru import logger
except Exception:  # pragma: no cover
    class _L:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def error(self, *a, **k): ...
        def debug(self, *a, **k): ...
    logger = _L()  # type: ignore


# === DANE WYNIKOWE ===
@dataclass
class AutoMLResult:
    problem_type: str                 # "classification" | "regression"
    best_model_name: str              # nazwa zwycięzcy
    metrics: Dict[str, float]         # metryki ewaluacji (na teście)
    feature_names: List[str]          # cechy po preprocesingu (get_feature_names_out, jeśli dostępne)
    pipeline: Any                     # gotowy pipeline (fit)
    y_pred: np.ndarray                # predykcje na teście (po optymalnym progu dla binarki)
    y_pred_proba: Optional[np.ndarray]  # proby (binarka/multiclass), jeśli dostępne
    best_threshold: Optional[float]   # tylko binarka – optymalny próg
    label_encoder: Optional[LabelEncoder]  # dla klasyfikacji – enkoder etykiet


# === PREPROCESSOR ===
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Kompatybilność OneHotEncoder: sparse_output (>=1.2) vs sparse (starsze)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn <1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
    return pre


# === METRYKI ===
def _metrics_cls_binary(y_true, y_proba, y_pred) -> Dict[str, float]:
    # y_proba: (n,2) lub (n,) – normalizujemy
    if y_proba is None:
        auc = np.nan
        ap = np.nan
    else:
        p1 = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            auc = float(roc_auc_score(y_true, p1))
        except Exception:
            auc = np.nan
        try:
            ap = float(average_precision_score(y_true, p1))
        except Exception:
            ap = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "roc_auc": float(auc),
        "pr_auc": float(ap),
    }


def _metrics_cls_multiclass(y_true, y_proba, y_pred) -> Dict[str, float]:
    # Multiclass: F1 weighted, AUC macro ovo/ovr jeśli proby dostępne
    try:
        auc_ovr = float(roc_auc_score(y_true, y_proba, multi_class="ovr")) if y_proba is not None else np.nan
    except Exception:
        auc_ovr = np.nan
    try:
        ap_macro = float(average_precision_score(pd.get_dummies(y_true), y_proba, average="macro")) if y_proba is not None else np.nan
    except Exception:
        ap_macro = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc_ovr": float(auc_ovr),
        "pr_auc_macro": float(ap_macro),
    }


def _metrics_reg(y_true, y_pred) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# === PROG BINARNY (OPTYMALIZACJA) ===
def _optimal_threshold(y_true: np.ndarray, y_proba_pos: np.ndarray, *, metric: str = "f1") -> Tuple[float, float]:
    """
    Skanuje progi (percentyle prob) i wybiera najlepszy dla zadanej metryki: 'f1' lub 'accuracy' lub 'youden' (J dla TPR-FPR).
    Zwraca: (best_thr, best_score)
    """
    if y_proba_pos.ndim != 1:
        y_proba_pos = y_proba_pos.ravel()

    # candidaty progów – percentyle (stabilne dla dużych zbiorów)
    qs = np.linspace(0.05, 0.95, 37)  # 37 punktów
    thrs = np.quantile(y_proba_pos, qs)
    best_thr, best_score = 0.5, -np.inf

    for t in np.unique(thrs):
        pred = (y_proba_pos >= t).astype(int)
        if metric == "accuracy":
            score = accuracy_score(y_true, pred)
        elif metric == "youden":
            # Youden's J = TPR - FPR (przydatny przy nierównych klasach)
            tp = np.sum((y_true == 1) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            tn = np.sum((y_true == 0) & (pred == 0))
            fp = np.sum((y_true == 0) & (pred == 1))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            score = tpr - fpr
        else:
            score = f1_score(y_true, pred)
        if score > best_score:
            best_thr, best_score = float(t), float(score)

    return best_thr, best_score


# === GŁÓWNA FUNKCJA ===
def train_automl(
    df: pd.DataFrame,
    target: str,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    calibrate: bool = True,              # dla binarki kalibracja Platt/Isotonic (auto)
    optimize_threshold_by: str = "f1",   # 'f1' | 'accuracy' | 'youden' (tylko binarka)
) -> AutoMLResult:
    """
    Trenuje szybki baseline AutoML i zwraca najlepszy pipeline z metrykami.
    - df: pełny DataFrame z targetem
    - target: nazwa kolumny celu
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in DataFrame")

    y_raw = df[target]
    X = df.drop(columns=[target]).copy()

    # heurystyka typu problemu
    n_unique = y_raw.nunique(dropna=True)
    problem_type = "classification" if (not pd.api.types.is_float_dtype(y_raw) and n_unique <= 50) else "regression"

    # Label encoding dla klasyfikacji (pozwala pracować z etykietami tekstowymi)
    label_encoder: Optional[LabelEncoder] = None
    y = y_raw
    multiclass = False
    if problem_type == "classification":
        label_encoder = LabelEncoder().fit(y_raw.astype(str))
        y = label_encoder.transform(y_raw.astype(str))
        multiclass = len(label_encoder.classes_) > 2

    pre = _build_preprocessor(X)

    # Kandydaci
    candidates: Dict[str, Any] = {}
    if problem_type == "classification":
        candidates["RandomForest"] = RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, class_weight=None if multiclass else "balanced_subsample"
        )
        if XGBClassifier is not None:
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", eval_metric="logloss", random_state=random_state
            )
        if LGBMClassifier is not None:
            candidates["LightGBM"] = LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, random_state=random_state
            )
    else:
        candidates["RandomForest"] = RandomForestRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1
        )
        if XGBRegressor is not None:
            candidates["XGBoost"] = XGBRegressor(
                n_estimators=600, learning_rate=0.05, max_depth=8,
                subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                random_state=random_state
            )
        if LGBMRegressor is not None:
            candidates["LightGBM"] = LGBMRegressor(
                n_estimators=700, learning_rate=0.05, num_leaves=64,
                subsample=0.9, colsample_bytree=0.9, random_state=random_state
            )

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if problem_type == "classification" else None
    )

    best_score = -np.inf
    best_name = ""
    best_pipe: Optional[Pipeline] = None
    best_y_pred = None
    best_y_proba = None
    best_threshold: Optional[float] = None
    last_metrics: Dict[str, float] = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        # Fit
        pipe.fit(X_tr, y_tr)

        # Pred
        if problem_type == "classification":
            # Proby
            y_proba = None
            # jeżeli chcemy kalibrować – owiń model kalibratorem na train
            if calibrate and hasattr(model, "predict_proba"):
                # dopasuj kalibrowany model na danych treningowych (z pipelinem)
                try:
                    # Kalibracja wymaga estymatora; uderzamy w końcówkę pipeline
                    base_est = model
                    cal = CalibratedClassifierCV(base_est, method="sigmoid" if len(np.unique(y_tr)) > 2 else "isotonic", cv=3)
                    pipe_cal = Pipeline(steps=[("pre", pre), ("model", cal)])
                    pipe_cal.fit(X_tr, y_tr)
                    pipe = pipe_cal
                except Exception as e:
                    logger.warning(f"Calibration skipped for {name}: {e}")

            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_te)
            # Optymalizacja progu dla binarki
            if not multiclass and y_proba is not None:
                p1 = y_proba[:, 1]
                thr, _ = _optimal_threshold(y_te, p1, metric=optimize_threshold_by)
                y_pred = (p1 >= thr).astype(int)
                best_thr_candidate = float(thr)
            else:
                y_pred = pipe.predict(X_te)
                best_thr_candidate = None

            # Metryki
            if multiclass:
                metrics = _metrics_cls_multiclass(y_te, y_proba, y_pred)
                score = metrics["f1_weighted"]
            else:
                metrics = _metrics_cls_binary(y_te, y_proba, y_pred)
                score = metrics["f1"]
        else:
            y_pred = pipe.predict(X_te)
            y_proba = None
            metrics = _metrics_reg(y_te, y_pred)
            score = metrics["r2"]
            best_thr_candidate = None

        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe
            best_y_pred = y_pred
            best_y_proba = y_proba
            best_threshold = best_thr_candidate
            last_metrics = metrics

        logger.info(f"[AutoML] {name} score={score:.5f} metrics={metrics}")  # type: ignore[attr-defined]

    # Feature names out (po OHE) – jeśli dostępne
    feature_names: List[str] = []
    try:
        pre_best = best_pipe.named_steps["pre"]  # type: ignore
        feature_names = list(pre_best.get_feature_names_out())
    except Exception:
        # fallback – oryginalne kolumny
        feature_names = list(X.columns)

    # Dekoduj etykiety do oryginału (dla y_pred) – tylko klasyfikacja
    if problem_type == "classification" and label_encoder is not None:
        try:
            best_y_pred_decoded = label_encoder.inverse_transform(best_y_pred)  # type: ignore
        except Exception:
            best_y_pred_decoded = best_y_pred
    else:
        best_y_pred_decoded = best_y_pred

    return AutoMLResult(
        problem_type=problem_type,
        best_model_name=best_name,
        metrics=last_metrics,
        feature_names=feature_names,
        pipeline=best_pipe,
        y_pred=np.asarray(best_y_pred_decoded),
        y_pred_proba=(np.asarray(best_y_proba) if best_y_proba is not None else None),
        best_threshold=best_threshold,
        label_encoder=label_encoder
    )
