# src/ml_models/automl_pipeline.py — TURBO PRO (back-compat API)
from __future__ import annotations
import pathlib
import json
import time
import warnings
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib

# =========================
# Logger
# =========================
LOGGER = logging.getLogger("automl")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.propagate = False

# =========================
# Ścieżki i rejestr
# =========================
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = MODELS_DIR / "registry.json"

def _save_registry(entry: dict):
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else []
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []
    data.append(entry)
    REGISTRY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# =========================
# Utils
# =========================
def _validate_df(df: pd.DataFrame, target: str) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame):
        return "Nieprawidłowy obiekt danych (brak DataFrame)."
    if target not in df.columns:
        return f"Brak kolumny celu: {target!r}."
    if len(df) < 30:
        return f"Zbyt mało wierszy do AutoML (len={len(df)} < 30)."
    if df[target].isna().all():
        return "Kolumna celu zawiera wyłącznie braki (NaN)."
    return None

def _is_classification(y: pd.Series) -> bool:
    if y is None or len(y) == 0:
        return False
    nunique = y.nunique(dropna=True)
    if nunique <= max(20, int(0.05 * len(y))):
        return True
    if (y.dtype == "object") or pd.api.types.is_categorical_dtype(y):
        return True
    return False

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _split_xy(df: pd.DataFrame, target: str, is_classif: bool, random_state: int):
    # Odfiltruj brakujące targety
    df2 = df.dropna(subset=[target]).copy()
    y = df2[target]
    X = df2.drop(columns=[target])
    stratify = y if is_classif and y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify
    )
    return X_tr, X_te, y_tr, y_te

def _column_roles(X: pd.DataFrame, max_onehot: int = 12) -> Dict[str, List[str]]:
    num_cols = list(X.select_dtypes(include=[np.number, "number", "float", "int"]).columns)
    obj_cols = list(X.select_dtypes(include=["object"]).columns)
    cat_low, cat_high = [], []
    for c in obj_cols:
        nun = X[c].nunique(dropna=True)
        if nun <= 1:
            continue
        if nun <= max_onehot:
            cat_low.append(c)
        else:
            cat_high.append(c)
    dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    bool_cols = list(X.select_dtypes(include=["bool"]).columns)
    # Usuń duplikaty między grupami
    for c in dt_cols + bool_cols:
        if c in cat_low: cat_low.remove(c)
        if c in cat_high: cat_high.remove(c)
    return {"num": num_cols, "cat_low": cat_low, "cat_high": cat_high, "bool": bool_cols, "dt": dt_cols}

def _make_ohe() -> OneHotEncoder:
    # Zgodność wsteczna (sklearn < 1.2 nie zna sparse_output)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# =========================
# Wrapper: dekodowanie etykiet (+obsługa eval_set)
# =========================
@dataclass
class LabelDecodingClassifier(BaseEstimator, ClassifierMixin):
    """
    Opakowuje dowolny estimator klasyfikacyjny, aby encode/decode etykiety Y
    oraz poprawnie przekazywać eval_set (enkoduje y_eval).
    """
    base_estimator: BaseEstimator
    classes_: Optional[np.ndarray] = None  # po fit
    _fitted: bool = False

    def fit(self, X, y, **fit_params):
        y_arr = pd.Series(y)
        # Czy wymaga enkodowania?
        needs_encoding = True
        if pd.api.types.is_integer_dtype(y_arr) or pd.api.types.is_bool_dtype(y_arr):
            vals = sorted(pd.unique(y_arr))
            needs_encoding = not (len(vals) > 0 and vals[0] == 0 and vals[-1] == len(vals) - 1)

        if needs_encoding:
            classes = pd.Index(pd.unique(y_arr)).sort_values()
            mapping = {cls: i for i, cls in enumerate(classes)}
            y_enc = y_arr.map(mapping).astype(int).values
            self.classes_ = classes.to_numpy()
            self._mapping = mapping
            self._inv_mapping = {v: k for k, v in mapping.items()}
        else:
            y_enc = y_arr.astype(int).values
            self.classes_ = np.array(sorted(pd.unique(y_arr)))
            self._mapping = None
            self._inv_mapping = None

        # Obsługa eval_set: przemapuj etykiety eval na encoded
        if "eval_set" in fit_params and fit_params["eval_set"]:
            eval_set = fit_params.pop("eval_set")
            new_eval = []
            for Xv, yv in eval_set:
                yv_ser = pd.Series(yv)
                if self._mapping is not None:
                    yv_enc = yv_ser.map(self._mapping).astype(int).values
                else:
                    yv_enc = yv_ser.astype(int).values
                new_eval.append((Xv, yv_enc))
            fit_params["eval_set"] = new_eval

        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y_enc, **fit_params)
        self._fitted = True

        # propaguj feature_importances_ jeśli dostępne
        if hasattr(self.base_estimator_, "feature_importances_"):
            self.feature_importances_ = getattr(self.base_estimator_, "feature_importances_")  # type: ignore[attr-defined]
        return self

    def predict(self, X):
        y_pred_enc = self.base_estimator_.predict(X)
        if getattr(self, "_inv_mapping", None) is None:
            return y_pred_enc
        if isinstance(y_pred_enc, (pd.Series, pd.Index)):
            y_pred_enc = y_pred_enc.to_numpy()
        return np.array([self._inv_mapping.get(int(v), v) for v in y_pred_enc])

    def predict_proba(self, X):
        if hasattr(self.base_estimator_, "predict_proba"):
            return self.base_estimator_.predict_proba(X)
        raise AttributeError("Base estimator nie wspiera predict_proba")

# =========================
# Główny AutoML
# =========================
def train_automl(df: pd.DataFrame, target: str, random_state: int = 42) -> Tuple[Any, Dict[str, float], str]:
    """
    Trenuje kilku kandydatów (LGBM / XGB / RandomForest) i wybiera najlepszy na walidacji.
    Zwraca:
      - model (sklearn Pipeline),
      - metrics (dict),
      - problem_type ("classification"|"regression")
    """
    # Ciszej dla ostrzeżeń z LGBM/XGB
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    err = _validate_df(df, target)
    if err:
        raise AssertionError(err)

    # Typ problemu + split
    y_series = df[target]
    is_classif = _is_classification(y_series)
    X_train, X_test, y_train, y_test = _split_xy(df, target, is_classif, random_state)

    # Role kolumn + preprocessing
    roles = _column_roles(X_train)
    transformers = []

    if roles["num"]:
        transformers.append(("num", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median"))
        ]), roles["num"]))

    if roles["cat_low"]:
        transformers.append(("cat_low", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe())
        ]), roles["cat_low"]))

    if roles["cat_high"]:
        transformers.append(("cat_high", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), roles["cat_high"]))

    if roles["bool"]:
        transformers.append(("bool", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), roles["bool"]))

    # Datetime kolumny – zakładamy, że FE już wyprodukował *_year, *_month itd.

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    # Przygotuj *eval_set* w tej samej przestrzeni cech (ważne dla early stopping)
    # Używamy klona preprocesora dopasowanego na train:
    pre_for_eval = clone(pre).fit(X_train, y_train)
    X_val_trans = pre_for_eval.transform(X_test)

    # Kandydaci
    candidates: List[Tuple[str, Any, Dict[str, Any]]] = []

    if is_classif:
        # wsparcie nierównowagi (binarka)
        n_classes = int(pd.Series(y_train).nunique())
        if n_classes == 2:
            cls_counts = pd.Series(y_train).value_counts()
            n_pos, n_neg = float(cls_counts.min()), float(cls_counts.max())
            scale_pos_weight = (n_neg / max(n_pos, 1.0))
        else:
            scale_pos_weight = 1.0

        candidates = [
            ("lgbm", LabelDecodingClassifier(LGBMClassifier(
                random_state=random_state,
                n_estimators=1200,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                class_weight="balanced" if n_classes > 2 else None,
            )), {"model__eval_set": [(X_val_trans, y_test)], "model__early_stopping_rounds": 80}),
            ("xgb", LabelDecodingClassifier(XGBClassifier(
                random_state=random_state,
                n_estimators=2500,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight if n_classes == 2 else 1.0,
            )), {"model__eval_set": [(X_val_trans, y_test)], "model__early_stopping_rounds": 120}),
            ("rf", LabelDecodingClassifier(RandomForestClassifier(
                random_state=random_state,
                n_estimators=700,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )), {}),
        ]
        problem_type = "classification"
    else:
        candidates = [
            ("lgbm", LGBMRegressor(
                random_state=random_state,
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
            ), {"model__eval_set": [(X_val_trans, y_test)], "model__early_stopping_rounds": 80}),
            ("xgb", XGBRegressor(
                random_state=random_state,
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
            ), {"model__eval_set": [(X_val_trans, y_test)], "model__early_stopping_rounds": 120}),
            ("rf", RandomForestRegressor(
                random_state=random_state,
                n_estimators=700,
                max_depth=None,
                n_jobs=-1,
            ), {}),
        ]
        problem_type = "regression"

    # Fit kandydatów (walidacja na X_test/y_test dla wyboru najlepszego)
    best_name, best_pipe, best_score = None, None, -np.inf
    holdout_metrics: Dict[str, float] = {}

    for name, est, fit_params in candidates:
        pipe = Pipeline(steps=[
            ("pre", pre),
            ("model", est),
        ])

        fit_kwargs = {}
        if fit_params:
            # przekazujemy eval_set z *transformowanym* X (X_val_trans)
            fit_kwargs.update(fit_params)

        try:
            pipe.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            # jeżeli fit z early stoppingiem się nie uda, spróbuj bez
            LOGGER.warning("fit '%s' z early stopping nie powiódł się (%s) – próbuję bez", name, e)
            pipe = Pipeline(steps=[("pre", pre), ("model", est)])
            pipe.fit(X_train, y_train)

        # Predykcje na holdoucie
        y_pred = pipe.predict(X_test)

        if problem_type == "classification":
            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))
            f1w = float(f1_score(y_test, y_pred, average="weighted"))
            score = f1w  # metryka wyboru

            # AUC dla binarki (jeśli dostępne proby)
            try:
                y_proba = pipe.predict_proba(X_test)
                proba = y_proba[:, 1] if (hasattr(y_proba, "shape") and y_proba.ndim == 2 and y_proba.shape[1] == 2) else None
                if proba is not None:
                    y_true_bin = pd.Series(y_test)
                    if not pd.api.types.is_integer_dtype(y_true_bin):
                        classes = pd.Index(pd.unique(y_true_bin)).sort_values()
                        mapping = {cls: i for i, cls in enumerate(classes)}
                        y_true_bin = y_true_bin.map(mapping).astype(int)
                    auc = float(roc_auc_score(y_true_bin, proba))
                else:
                    auc = float("nan")
            except Exception:
                auc = float("nan")

            metrics = {"accuracy": acc, "balanced_accuracy": bacc, "f1_weighted": f1w, "roc_auc": auc}

        else:
            y_true = np.asarray(y_test).astype(float)
            y_hat = np.asarray(y_pred).astype(float)
            rmse = float(mean_squared_error(y_true, y_hat, squared=False))
            mae = float(mean_absolute_error(y_true, y_hat))
            r2 = float(r2_score(y_true, y_hat))
            mape = _safe_mape(y_true, y_hat)
            score = -rmse
            metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

        LOGGER.info("candidate '%s' -> score=%.5f metrics=%s", name, score, metrics)

        if score > best_score:
            best_name, best_pipe, best_score = name, pipe, score
            holdout_metrics = metrics

    assert best_pipe is not None, "Nie udało się wytrenować żadnego modelu."

    # Doklej feature_importances_ (dla zgodności z UI)
    try:
        base = best_pipe.named_steps["model"]
        if hasattr(base, "feature_importances_"):
            setattr(best_pipe, "feature_importances_", getattr(base, "feature_importances_"))
        elif hasattr(base, "base_estimator_") and hasattr(base.base_estimator_, "feature_importances_"):
            setattr(best_pipe, "feature_importances_", getattr(base.base_estimator_, "feature_importances_"))
    except Exception:
        pass

    # Zapis modelu i rejestru
    model_id = f"{problem_type}_{best_name}_{int(time.time())}_{random_state}"
    model_path = MODELS_DIR / f"model_{model_id}.joblib"

    payload = {
        "model": best_pipe,
        "target": target,
        "problem_type": problem_type,
        "columns": list(df.drop(columns=[target]).columns),
        "preprocessor": "sklearn.ColumnTransformer",
        "best_estimator": best_name,
        "metrics": holdout_metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": random_state,
        "versions": {
            "sklearn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
        },
    }
    try:
        joblib.dump(payload, model_path)
    except Exception:
        # minimalny fallback: zapisz sam pipeline
        joblib.dump(best_pipe, model_path)

    _save_registry({
        "path": model_path.name,
        "target": target,
        "problem_type": problem_type,
        "best_estimator": best_name,
        "metrics": holdout_metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return best_pipe, holdout_metrics, problem_type
