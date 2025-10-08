"""
AutoML Pipeline Engine - Zaawansowany automatyczny trening modeli ML.

Funkcjonalności:
- Automatyczna detekcja typu problemu (classification/regression)
- Multi-model competition (LGBM, XGBoost, RandomForest)
- Inteligentny preprocessing (numerical, categorical, datetime, boolean)
- Early stopping z eval_set
- Label encoding dla klasyfikacji
- Class imbalance handling
- Feature importance tracking
- Model registry i wersjonowanie
- Comprehensive metrics
- Robust error handling
"""

from __future__ import annotations

import pathlib
import json
import time
import warnings
import logging
from typing import Tuple, Dict, Any, List, Optional, Literal, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Ścieżki
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = MODELS_DIR / "registry.json"

# Limity bezpieczeństwa
MIN_ROWS_FOR_TRAINING = 30
MIN_ROWS_FOR_VALIDATION = 10
MAX_ONEHOT_CARDINALITY = 12
MAX_CLASSES_FOR_CLASSIFICATION = 100
TEST_SIZE = 0.2

# Domyślne parametry
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1

# Typy problemów
ProblemType = Literal["classification", "regression"]
ModelName = Literal["lgbm", "xgb", "rf"]

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "automl_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger bez duplikatów handlerów.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class ColumnRoles:
    """役割ごとに分類された列."""
    numeric: List[str] = field(default_factory=list)
    categorical_low: List[str] = field(default_factory=list)
    categorical_high: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Konwertuje do słownika."""
        return {
            "numeric": self.numeric,
            "categorical_low": self.categorical_low,
            "categorical_high": self.categorical_high,
            "boolean": self.boolean,
            "datetime": self.datetime
        }


@dataclass
class TrainingMetrics:
    """Metryki z treningu modelu."""
    # Common
    problem_type: str
    best_model: str
    training_time: float
    n_samples_train: int
    n_samples_test: int
    n_features: int
    
    # Classification specific
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression specific
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika, pomijając None."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ModelPayload:
    """Pełny payload zapisywanego modelu."""
    model: Any
    target: str
    problem_type: str
    columns: List[str]
    column_roles: Dict[str, List[str]]
    preprocessor: str
    best_estimator: str
    metrics: Dict[str, Any]
    created_at: str
    random_state: int
    versions: Dict[str, str]
    feature_names: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika (bez modelu - dla rejestru)."""
        data = asdict(self)
        data.pop("model", None)  # Model nie idzie do rejestru
        return data


# ========================================================================================
# VALIDATION
# ========================================================================================

def _validate_dataframe(df: pd.DataFrame, target: str) -> Optional[str]:
    """
    Waliduje DataFrame i kolumnę celu.
    
    Args:
        df: DataFrame do walidacji
        target: Nazwa kolumny celu
        
    Returns:
        Komunikat błędu lub None jeśli OK
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return "Input nie jest prawidłowym DataFrame"
    
    if df.empty:
        return "DataFrame jest pusty"
    
    if len(df) < MIN_ROWS_FOR_TRAINING:
        return f"Za mało wierszy: {len(df)} < {MIN_ROWS_FOR_TRAINING}"
    
    if target not in df.columns:
        return f"Kolumna celu '{target}' nie istnieje w DataFrame"
    
    if df[target].isna().all():
        return "Kolumna celu zawiera wyłącznie wartości NaN"
    
    # Sprawdź czy jest przynajmniej jedna cecha
    if len(df.columns) <= 1:
        return "Brak cech (feature columns) - tylko kolumna celu"
    
    return None


def _infer_problem_type(y: pd.Series) -> ProblemType:
    """
    Automatycznie wykrywa typ problemu.
    
    Args:
        y: Serie z wartościami celu
        
    Returns:
        "classification" lub "regression"
    """
    # Usuń NaN
    y_clean = y.dropna()
    
    if len(y_clean) == 0:
        LOGGER.warning("Brak wartości do wykrycia typu problemu, zakładam regression")
        return "regression"
    
    # Liczba unikalnych wartości
    n_unique = y_clean.nunique()
    
    # Heurystyka: jeśli mało unikalnych wartości względem rozmiaru - klasyfikacja
    classification_threshold = max(20, int(0.05 * len(y_clean)))
    
    if n_unique <= classification_threshold:
        LOGGER.debug(f"Wykryto klasyfikację: {n_unique} unikalnych wartości")
        return "classification"
    
    # Jeśli dtype to object/category - klasyfikacja
    if y_clean.dtype == "object" or pd.api.types.is_categorical_dtype(y_clean):
        LOGGER.debug(f"Wykryto klasyfikację: typ {y_clean.dtype}")
        return "classification"
    
    # Jeśli bool - klasyfikacja
    if pd.api.types.is_bool_dtype(y_clean):
        LOGGER.debug("Wykryto klasyfikację: typ boolean")
        return "classification"
    
    # W przeciwnym razie - regresja
    LOGGER.debug(f"Wykryto regresję: {n_unique} unikalnych wartości, typ {y_clean.dtype}")
    return "regression"


# ========================================================================================
# DATA SPLITTING
# ========================================================================================

def _split_data(
    df: pd.DataFrame,
    target: str,
    is_classification: bool,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Dzieli dane na train/test z obsługą stratyfikacji.
    
    Args:
        df: DataFrame źródłowy
        target: Nazwa kolumny celu
        is_classification: Czy problem klasyfikacyjny
        test_size: Rozmiar zbioru testowego
        random_state: Random state
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    # Usuń wiersze z brakami w targecie
    df_clean = df.dropna(subset=[target]).copy()
    
    if len(df_clean) < MIN_ROWS_FOR_TRAINING:
        raise ValueError(
            f"Po usunięciu braków w targecie pozostało {len(df_clean)} wierszy "
            f"(minimum: {MIN_ROWS_FOR_TRAINING})"
        )
    
    y = df_clean[target]
    X = df_clean.drop(columns=[target])
    
    # Stratyfikacja dla klasyfikacji (jeśli możliwa)
    stratify = None
    if is_classification:
        n_unique = y.nunique()
        if n_unique > 1 and n_unique < len(y) * test_size:
            stratify = y
            LOGGER.debug("Używam stratyfikacji dla podziału danych")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
    except ValueError as e:
        # Fallback bez stratyfikacji
        LOGGER.warning(f"Stratyfikacja nieudana: {e}, próbuję bez stratyfikacji")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    
    LOGGER.info(
        f"Podział danych: train={len(X_train)}, test={len(X_test)} "
        f"({len(X_test)/len(df_clean)*100:.1f}%)"
    )
    
    return X_train, X_test, y_train, y_test


# ========================================================================================
# COLUMN ROLES
# ========================================================================================

def _identify_column_roles(
    X: pd.DataFrame,
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY
) -> ColumnRoles:
    """
    Identyfikuje role poszczególnych kolumn.
    
    Args:
        X: DataFrame z cechami
        max_onehot_cardinality: Max unikalnych wartości dla OneHotEncoder
        
    Returns:
        ColumnRoles object
    """
    numeric_cols: List[str] = []
    cat_low: List[str] = []
    cat_high: List[str] = []
    bool_cols: List[str] = []
    dt_cols: List[str] = []
    
    for col in X.columns:
        series = X[col]
        
        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_cols.append(col)
            continue
        
        # Boolean
        if pd.api.types.is_bool_dtype(series):
            bool_cols.append(col)
            continue
        
        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            # Sprawdź czy to nie ukryta kategoryczna (np. 0/1/2)
            n_unique = series.nunique(dropna=True)
            if n_unique <= 1:
                # Kolumna stała - skip
                LOGGER.debug(f"Pomijam kolumnę stałą: {col}")
                continue
            elif n_unique <= max_onehot_cardinality and n_unique < len(series) * 0.05:
                # Prawdopodobnie kategoryczna
                cat_low.append(col)
            else:
                numeric_cols.append(col)
            continue
        
        # Categorical/Object
        n_unique = series.nunique(dropna=True)
        
        if n_unique <= 1:
            LOGGER.debug(f"Pomijam kolumnę stałą: {col}")
            continue
        
        if n_unique <= max_onehot_cardinality:
            cat_low.append(col)
        else:
            cat_high.append(col)
    
    roles = ColumnRoles(
        numeric=numeric_cols,
        categorical_low=cat_low,
        categorical_high=cat_high,
        boolean=bool_cols,
        datetime=dt_cols
    )
    
    LOGGER.debug(f"Column roles: {roles.to_dict()}")
    return roles


# ========================================================================================
# PREPROCESSING
# ========================================================================================

def _create_onehot_encoder() -> OneHotEncoder:
    """
    Tworzy OneHotEncoder z backward compatibility.
    
    Returns:
        Skonfigurowany OneHotEncoder
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=None
        )
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def _build_preprocessor(roles: ColumnRoles) -> ColumnTransformer:
    """
    Buduje preprocessor na podstawie ról kolumn.
    
    Args:
        roles: ColumnRoles object
        
    Returns:
        Skonfigurowany ColumnTransformer
    """
    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    
    # Numeric columns
    if roles.numeric:
        numeric_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            # Opcjonalnie: StandardScaler dla niektórych modeli
        ])
        transformers.append(("numeric", numeric_pipeline, roles.numeric))
    
    # Low cardinality categorical (OneHot)
    if roles.categorical_low:
        cat_low_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", _create_onehot_encoder())
        ])
        transformers.append(("cat_low", cat_low_pipeline, roles.categorical_low))
    
    # High cardinality categorical (Ordinal)
    if roles.categorical_high:
        cat_high_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])
        transformers.append(("cat_high", cat_high_pipeline, roles.categorical_high))
    
    # Boolean columns
    if roles.boolean:
        bool_pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])
        transformers.append(("boolean", bool_pipeline, roles.boolean))
    
    # Datetime - zakładamy że feature engineering już je przekonwertował
    # (np. na _year, _month, _day_of_week itd.)
    
    if not transformers:
        raise ValueError("Brak kolumn do przetworzenia - wszystkie kolumny są stałe lub nieprawidłowe")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    LOGGER.debug(f"Utworzono preprocessor z {len(transformers)} transformerami")
    return preprocessor


# ========================================================================================
# LABEL ENCODING WRAPPER
# ========================================================================================

@dataclass
class LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper dla klasyfikatora z automatycznym encoding/decoding etykiet.
    
    Obsługuje:
    - Encoding non-integer labels do 0, 1, 2, ...
    - Decoding predictions z powrotem do oryginalnych labels
    - Przekazywanie eval_set z encodingiem
    - Propagacja feature_importances_
    """
    base_estimator: BaseEstimator
    classes_: Optional[np.ndarray] = None
    _mapping: Optional[Dict[Any, int]] = None
    _inv_mapping: Optional[Dict[int, Any]] = None
    _fitted: bool = False
    
    def fit(self, X, y, **fit_params):
        """Fit z encoding etykiet i eval_set."""
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        # Sprawdź czy potrzebne encoding
        needs_encoding = True
        
        if pd.api.types.is_integer_dtype(y_series) or pd.api.types.is_bool_dtype(y_series):
            unique_vals = sorted(y_series.dropna().unique())
            # Jeśli już są 0, 1, 2, ... to skip encoding
            if len(unique_vals) > 0:
                is_sequential = all(
                    unique_vals[i] == i 
                    for i in range(len(unique_vals))
                )
                if is_sequential:
                    needs_encoding = False
        
        # Encoding
        if needs_encoding:
            classes = pd.Index(y_series.dropna().unique()).sort_values()
            self._mapping = {cls: i for i, cls in enumerate(classes)}
            self._inv_mapping = {i: cls for cls, i in self._mapping.items()}
            self.classes_ = classes.to_numpy()
            
            y_encoded = y_series.map(self._mapping).astype(int).values
            LOGGER.debug(f"Encoded {len(self._mapping)} classes")
        else:
            y_encoded = y_series.astype(int).values
            self.classes_ = np.array(sorted(y_series.dropna().unique()))
            self._mapping = None
            self._inv_mapping = None
        
        # Obsługa eval_set
        if "eval_set" in fit_params and fit_params["eval_set"]:
            eval_set = fit_params.pop("eval_set")
            encoded_eval_set = []
            
            for X_val, y_val in eval_set:
                y_val_series = pd.Series(y_val) if not isinstance(y_val, pd.Series) else y_val
                
                if self._mapping is not None:
                    y_val_encoded = y_val_series.map(self._mapping).astype(int).values
                else:
                    y_val_encoded = y_val_series.astype(int).values
                
                encoded_eval_set.append((X_val, y_val_encoded))
            
            fit_params["eval_set"] = encoded_eval_set
        
        # Clone i fit base estimator
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y_encoded, **fit_params)
        
        # Propagacja feature_importances_
        if hasattr(self.base_estimator_, "feature_importances_"):
            self.feature_importances_ = self.base_estimator_.feature_importances_  # type: ignore
        
        self._fitted = True
        return self
    
    def predict(self, X):
        """Predict z decoding etykiet."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        y_pred_encoded = self.base_estimator_.predict(X)
        
        # Decode jeśli było encoding
        if self._inv_mapping is None:
            return y_pred_encoded
        
        # Convert to numpy if needed
        if isinstance(y_pred_encoded, (pd.Series, pd.Index)):
            y_pred_encoded = y_pred_encoded.to_numpy()
        
        return np.array([
            self._inv_mapping.get(int(val), val)
            for val in y_pred_encoded
        ])
    
    def predict_proba(self, X):
        """Predict proba (pass-through)."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not hasattr(self.base_estimator_, "predict_proba"):
            raise AttributeError("Base estimator nie wspiera predict_proba")
        
        return self.base_estimator_.predict_proba(X)


# ========================================================================================
# MODEL CANDIDATES
# ========================================================================================

def _get_classification_candidates(
    random_state: int,
    n_classes: int,
    y_train: pd.Series
) -> List[Tuple[ModelName, BaseEstimator, Dict[str, Any]]]:
    """
    Tworzy listę kandydatów dla klasyfikacji.
    
    Args:
        random_state: Random state
        n_classes: Liczba klas
        y_train: Series z targetem (dla class weights)
        
    Returns:
        Lista tuple (name, estimator, fit_params)
    """
    # Oblicz scale_pos_weight dla binary classification
    scale_pos_weight = 1.0
    if n_classes == 2:
        class_counts = y_train.value_counts()
        if len(class_counts) == 2:
            n_neg = float(class_counts.max())
            n_pos = float(class_counts.min())
            scale_pos_weight = n_neg / max(n_pos, 1.0)
            LOGGER.debug(f"Binary classification: scale_pos_weight={scale_pos_weight:.2f}")
    
    candidates: List[Tuple[ModelName, BaseEstimator, Dict[str, Any]]] = [
        (
            "lgbm",
            LabelEncodingClassifier(base_estimator=LGBMClassifier(
                random_state=random_state,
                n_estimators=1200,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=DEFAULT_N_JOBS,
                class_weight="balanced" if n_classes > 2 else None,
                verbosity=-1
            )),
            {"early_stopping_rounds": 80}
        ),
        (
            "xgb",
            LabelEncodingClassifier(base_estimator=XGBClassifier(
                random_state=random_state,
                n_estimators=2500,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=DEFAULT_N_JOBS,
                scale_pos_weight=scale_pos_weight if n_classes == 2 else 1.0,
                verbosity=0
            )),
            {"early_stopping_rounds": 120}
        ),
        (
            "rf",
            LabelEncodingClassifier(base_estimator=RandomForestClassifier(
                random_state=random_state,
                n_estimators=700,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=DEFAULT_N_JOBS,
                class_weight="balanced_subsample"
            )),
            {}
        ),
    ]
    
    return candidates


def _get_regression_candidates(
    random_state: int
) -> List[Tuple[ModelName, BaseEstimator, Dict[str, Any]]]:
    """
    Tworzy listę kandydatów dla regresji.
    
    Args:
        random_state: Random state
        
    Returns:
        Lista tuple (name, estimator, fit_params)
    """
    candidates: List[Tuple[ModelName, BaseEstimator, Dict[str, Any]]] = [
        (
            "lgbm",
            LGBMRegressor(
                random_state=random_state,
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=DEFAULT_N_JOBS,
                verbosity=-1
            ),
            {"early_stopping_rounds": 80}
        ),
        (
            "xgb",
            XGBRegressor(
                random_state=random_state,
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=DEFAULT_N_JOBS,
                verbosity=0
            ),
            {"early_stopping_rounds": 120}
        ),
        (
            "rf",
            RandomForestRegressor(
                random_state=random_state,
                n_estimators=700,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=DEFAULT_N_JOBS
            ),
            {}
        ),
    ]
    
    return candidates


# ========================================================================================
# METRICS COMPUTATION
# ========================================================================================

def _compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipeline: Pipeline
) -> Dict[str, float]:
    """
    Oblicza metryki dla klasyfikacji.
    
    Args:
        y_true: Prawdziwe etykiety
        y_pred: Predykcje
        pipeline: Pipeline (dla predict_proba)
        
    Returns:
        Słownik z metrykami
    """
    metrics: Dict[str, float] = {}
    
    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # Weighted metrics
    try:
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception as e:
        LOGGER.warning(f"Nie udało się obliczyć weighted metrics: {e}")
    
    # ROC AUC dla binary
    try:
        y_proba = pipeline.predict_proba(y_true.index)
        
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            # Binary classification
            proba_positive = y_proba[:, 1]
            
            # Encode y_true if needed
            y_true_binary = y_true.copy()
            if not pd.api.types.is_integer_dtype(y_true_binary):
                classes = sorted(y_true_binary.unique())
                mapping = {cls: i for i, cls in enumerate(classes)}
                y_true_binary = y_true_binary.map(mapping)
            
            metrics["roc_auc"] = float(roc_auc_score(y_true_binary, proba_positive))
    except Exception as e:
        LOGGER.warning(f"Nie udało się obliczyć ROC AUC: {e}")
        metrics["roc_auc"] = float("nan")
    
    return metrics


def _compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Oblicza metryki dla regresji.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        
    Returns:
        Słownik z metrykami
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    metrics: Dict[str, float] = {}
    
    metrics["rmse"] = float(mean_squared_error(y_true_arr, y_pred_arr, squared=False))
    metrics["mae"] = float(mean_absolute_error(y_true_arr, y_pred_arr))
    metrics["r2"] = float(r2_score(y_true_arr, y_pred_arr))
    
    # MAPE (z obsługą zeros)
    try:
        mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100.0)
        metrics["mape"] = mape
    except Exception:
        # Fallback do safe MAPE
        eps = 1e-8
        denom = np.maximum(np.abs(y_true_arr), eps)
        mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)
        metrics["mape"] = mape
    
    return metrics


def _get_selection_score(metrics: Dict[str, float], problem_type: str) -> float:
    """
    Zwraca score do wyboru najlepszego modelu.
    
    Args:
        metrics: Słownik z metrykami
        problem_type: Typ problemu
    Returns:
        Score (wyższy = lepszy)
    """
    if problem_type == "classification":
        # Preferuj F1 weighted, fallback na balanced accuracy
        return metrics.get("f1_weighted", metrics.get("balanced_accuracy", 0.0))
    else:
        # Dla regresji: ujemny RMSE (żeby wyższy był lepszy)
        return -metrics.get("rmse", float("inf"))


# ========================================================================================
# MODEL TRAINING
# ========================================================================================

def _train_candidate(
    name: str,
    estimator: BaseEstimator,
    fit_params: Dict[str, Any],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str
) -> Tuple[Pipeline, Dict[str, float], bool]:
    """
    Trenuje pojedynczego kandydata.
    
    Args:
        name: Nazwa modelu
        estimator: Estimator do wytrenowania
        fit_params: Parametry fit (np. eval_set)
        preprocessor: Preprocessor
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        problem_type: Typ problemu
        
    Returns:
        Tuple (pipeline, metrics, used_early_stopping)
    """
    # Utwórz pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    
    # Przygotuj eval_set jeśli potrzebne
    fit_kwargs = {}
    used_early_stopping = False
    
    if fit_params and "early_stopping_rounds" in fit_params:
        # Transform X_test używając preprocessora (clone żeby nie leakować)
        try:
            preprocessor_for_eval = clone(preprocessor).fit(X_train, y_train)
            X_test_transformed = preprocessor_for_eval.transform(X_test)
            
            fit_kwargs["model__eval_set"] = [(X_test_transformed, y_test)]
            fit_kwargs["model__early_stopping_rounds"] = fit_params["early_stopping_rounds"]
            
            # LGBM/XGB verbose
            if name in ("lgbm", "xgb"):
                fit_kwargs["model__verbose"] = False
            
            used_early_stopping = True
            LOGGER.debug(f"{name}: używam early stopping z eval_set")
            
        except Exception as e:
            LOGGER.warning(f"{name}: nie udało się przygotować eval_set: {e}")
    
    # Fit
    try:
        pipeline.fit(X_train, y_train, **fit_kwargs)
        LOGGER.debug(f"{name}: fit zakończony sukcesem")
    except Exception as e:
        if used_early_stopping:
            # Retry bez early stopping
            LOGGER.warning(f"{name}: fit z early stopping nie powiódł się, próbuję bez: {e}")
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])
            pipeline.fit(X_train, y_train)
            used_early_stopping = False
        else:
            raise
    
    # Predykcje
    y_pred = pipeline.predict(X_test)
    
    # Metryki
    if problem_type == "classification":
        metrics = _compute_classification_metrics(y_test, y_pred, pipeline)
    else:
        metrics = _compute_regression_metrics(y_test, y_pred)
    
    return pipeline, metrics, used_early_stopping


def _train_all_candidates(
    candidates: List[Tuple[str, BaseEstimator, Dict[str, Any]]],
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str
) -> Tuple[str, Pipeline, Dict[str, float]]:
    """
    Trenuje wszystkich kandydatów i wybiera najlepszego.
    
    Args:
        candidates: Lista kandydatów
        preprocessor: Preprocessor
        X_train, y_train: Dane treningowe
        X_test, y_test: Dane testowe
        problem_type: Typ problemu
        
    Returns:
        Tuple (best_name, best_pipeline, best_metrics)
    """
    best_name: Optional[str] = None
    best_pipeline: Optional[Pipeline] = None
    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    
    for name, estimator, fit_params in candidates:
        LOGGER.info(f"Trenuję kandydata: {name}")
        
        try:
            pipeline, metrics, used_es = _train_candidate(
                name=name,
                estimator=estimator,
                fit_params=fit_params,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                problem_type=problem_type
            )
            
            # Score do selekcji
            score = _get_selection_score(metrics, problem_type)
            
            LOGGER.info(
                f"{name}: score={score:.5f}, metrics={metrics}, "
                f"early_stopping={used_es}"
            )
            
            # Sprawdź czy najlepszy
            if score > best_score:
                best_name = name
                best_pipeline = pipeline
                best_score = score
                best_metrics = metrics
                LOGGER.info(f"Nowy najlepszy model: {name}")
        
        except Exception as e:
            LOGGER.exception(f"Błąd treningu kandydata {name}: {e}")
            continue
    
    if best_pipeline is None:
        raise RuntimeError("Nie udało się wytrenować żadnego modelu")
    
    return best_name, best_pipeline, best_metrics


# ========================================================================================
# FEATURE IMPORTANCE
# ========================================================================================

def _extract_feature_importances(pipeline: Pipeline) -> Optional[np.ndarray]:
    """
    Wyciąga feature importances z pipeline.
    
    Args:
        pipeline: Wytrenowany pipeline
        
    Returns:
        Array z importances lub None
    """
    try:
        model = pipeline.named_steps["model"]
        
        # Direct access
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        
        # LabelEncodingClassifier wrapper
        if hasattr(model, "base_estimator_"):
            base = model.base_estimator_
            if hasattr(base, "feature_importances_"):
                return base.feature_importances_
        
    except Exception as e:
        LOGGER.warning(f"Nie udało się wyciągnąć feature importances: {e}")
    
    return None


def _attach_feature_importances(pipeline: Pipeline) -> None:
    """
    Przyczepia feature_importances_ do pipeline (dla UI).
    
    Args:
        pipeline: Pipeline do modyfikacji
    """
    importances = _extract_feature_importances(pipeline)
    
    if importances is not None:
        pipeline.feature_importances_ = importances  # type: ignore
        LOGGER.debug(f"Przyczepiono feature_importances_ ({len(importances)} features)")


# ========================================================================================
# MODEL SAVING
# ========================================================================================

def _save_model(
    pipeline: Pipeline,
    target: str,
    problem_type: str,
    best_name: str,
    metrics: Dict[str, float],
    columns: List[str],
    column_roles: ColumnRoles,
    random_state: int
) -> Tuple[pathlib.Path, str]:
    """
    Zapisuje model i aktualizuje rejestr.
    
    Args:
        pipeline: Wytrenowany pipeline
        target: Nazwa kolumny celu
        problem_type: Typ problemu
        best_name: Nazwa najlepszego modelu
        metrics: Metryki
        columns: Lista kolumn cech
        column_roles: Role kolumn
        random_state: Random state
        
    Returns:
        Tuple (model_path, model_id)
    """
    # Generuj ID
    timestamp = int(time.time())
    model_id = f"{problem_type}_{best_name}_{timestamp}_{random_state}"
    model_path = MODELS_DIR / f"model_{model_id}.joblib"
    
    # Utworz payload
    payload = ModelPayload(
        model=pipeline,
        target=target,
        problem_type=problem_type,
        columns=columns,
        column_roles=column_roles.to_dict(),
        preprocessor="sklearn.ColumnTransformer",
        best_estimator=best_name,
        metrics=metrics,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        random_state=random_state,
        versions={
            "sklearn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
    )
    
    # Zapisz model
    try:
        joblib.dump(payload, model_path, compress=3)
        LOGGER.info(f"Model zapisany: {model_path}")
    except Exception as e:
        LOGGER.warning(f"Nie udało się zapisać pełnego payload: {e}, zapisuję sam pipeline")
        joblib.dump(pipeline, model_path, compress=3)
    
    # Aktualizuj rejestr
    _update_registry(
        model_path=model_path,
        payload=payload
    )
    
    return model_path, model_id


def _update_registry(model_path: pathlib.Path, payload: ModelPayload) -> None:
    """
    Aktualizuje rejestr modeli.
    
    Args:
        model_path: Ścieżka do pliku modelu
        payload: Payload modelu
    """
    # Wczytaj istniejący rejestr
    try:
        if REGISTRY_FILE.exists():
            registry_data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            if not isinstance(registry_data, list):
                registry_data = []
        else:
            registry_data = []
    except Exception as e:
        LOGGER.warning(f"Nie udało się wczytać rejestru: {e}, tworzę nowy")
        registry_data = []
    
    # Dodaj nowy wpis
    registry_entry = {
        "path": model_path.name,
        "full_path": str(model_path),
        "target": payload.target,
        "problem_type": payload.problem_type,
        "best_estimator": payload.best_estimator,
        "metrics": payload.metrics,
        "created_at": payload.created_at,
        "random_state": payload.random_state,
        "n_features": len(payload.columns),
        "versions": payload.versions
    }
    
    registry_data.append(registry_entry)
    
    # Zapisz rejestr
    try:
        REGISTRY_FILE.write_text(
            json.dumps(registry_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        LOGGER.info("Rejestr zaktualizowany")
    except Exception as e:
        LOGGER.error(f"Nie udało się zapisać rejestru: {e}")


# ========================================================================================
# MAIN API
# ========================================================================================

def train_automl(
    df: pd.DataFrame,
    target: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = TEST_SIZE,
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY
) -> Tuple[Pipeline, Dict[str, float], str]:
    """
    Automatyczny trening modelu ML z konkurencją między algorytmami.
    
    Pipeline:
    1. Walidacja danych i kolumny celu
    2. Automatyczna detekcja typu problemu (classification/regression)
    3. Podział na train/test z stratyfikacją
    4. Identyfikacja ról kolumn (numeric, categorical, boolean, datetime)
    5. Budowa preprocessora (imputation, encoding, scaling)
    6. Trening kandydatów: LGBM, XGBoost, RandomForest
    7. Wybór najlepszego na podstawie metryk walidacyjnych
    8. Zapis modelu i aktualizacja rejestru
    
    Args:
        df: DataFrame z danymi (features + target)
        target: Nazwa kolumny celu
        random_state: Random state dla reprodukowalności (default: 42)
        test_size: Rozmiar zbioru testowego (default: 0.2)
        max_onehot_cardinality: Max unikalnych wartości dla OneHot (default: 12)
        
    Returns:
        Tuple zawierający:
        - pipeline: Wytrenowany sklearn Pipeline (preprocessor + model)
        - metrics: Słownik z metrykami na zbiorze testowym
        - problem_type: "classification" lub "regression"
        
    Raises:
        ValueError: Jeśli dane są nieprawidłowe
        RuntimeError: Jeśli wszystkie modele zawiodły
        
    Example:
        >>> pipeline, metrics, ptype = train_automl(df, target="price")
        >>> print(f"Problem type: {ptype}")
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
        >>> predictions = pipeline.predict(new_data)
        
    Notes:
        - Model jest automatycznie zapisywany w `models/trained_models/`
        - Rejestr modeli aktualizowany w `models/trained_models/registry.json`
        - Pipeline zawiera `feature_importances_` jeśli dostępne
        - Dla klasyfikacji obsługuje encoding nietypowych labels
        - Early stopping używany dla LGBM i XGBoost
    """
    start_time = time.time()
    
    LOGGER.info("="*80)
    LOGGER.info("AutoML Pipeline - START")
    LOGGER.info("="*80)
    
    # 1. Walidacja
    LOGGER.info("Etap 1/8: Walidacja danych")
    validation_error = _validate_dataframe(df, target)
    if validation_error:
        raise ValueError(f"Walidacja nie powiodła się: {validation_error}")
    
    LOGGER.info(f"✓ DataFrame: {len(df)} wierszy × {len(df.columns)} kolumn")
    LOGGER.info(f"✓ Kolumna celu: '{target}'")
    
    # 2. Detekcja typu problemu
    LOGGER.info("Etap 2/8: Detekcja typu problemu")
    y_series = df[target]
    problem_type = _infer_problem_type(y_series)
    
    if problem_type == "classification":
        n_classes = y_series.nunique(dropna=True)
        if n_classes > MAX_CLASSES_FOR_CLASSIFICATION:
            raise ValueError(
                f"Za dużo klas: {n_classes} > {MAX_CLASSES_FOR_CLASSIFICATION}. "
                "Rozważ regresję lub zmniejszenie liczby klas."
            )
        LOGGER.info(f"✓ Typ problemu: CLASSIFICATION ({n_classes} klas)")
    else:
        LOGGER.info("✓ Typ problemu: REGRESSION")
    
    is_classification = (problem_type == "classification")
    
    # 3. Podział danych
    LOGGER.info("Etap 3/8: Podział na train/test")
    X_train, X_test, y_train, y_test = _split_data(
        df=df,
        target=target,
        is_classification=is_classification,
        test_size=test_size,
        random_state=random_state
    )
    
    LOGGER.info(f"✓ Train: {len(X_train)} wierszy, Test: {len(X_test)} wierszy")
    
    # 4. Identyfikacja ról kolumn
    LOGGER.info("Etap 4/8: Identyfikacja ról kolumn")
    column_roles = _identify_column_roles(X_train, max_onehot_cardinality)
    
    total_features = sum([
        len(column_roles.numeric),
        len(column_roles.categorical_low),
        len(column_roles.categorical_high),
        len(column_roles.boolean)
    ])
    
    LOGGER.info(f"✓ Numeric: {len(column_roles.numeric)}")
    LOGGER.info(f"✓ Categorical (low card): {len(column_roles.categorical_low)}")
    LOGGER.info(f"✓ Categorical (high card): {len(column_roles.categorical_high)}")
    LOGGER.info(f"✓ Boolean: {len(column_roles.boolean)}")
    LOGGER.info(f"✓ Datetime: {len(column_roles.datetime)} (pomijane - zakładamy FE)")
    LOGGER.info(f"✓ Total features: {total_features}")
    
    if total_features == 0:
        raise ValueError("Brak użytecznych cech po identyfikacji ról")
    
    # 5. Budowa preprocessora
    LOGGER.info("Etap 5/8: Budowa preprocessora")
    preprocessor = _build_preprocessor(column_roles)
    LOGGER.info("✓ Preprocessor utworzony")
    
    # 6. Przygotowanie kandydatów
    LOGGER.info("Etap 6/8: Przygotowanie kandydatów")
    if is_classification:
        n_classes = int(y_train.nunique())
        candidates = _get_classification_candidates(
            random_state=random_state,
            n_classes=n_classes,
            y_train=y_train
        )
    else:
        candidates = _get_regression_candidates(random_state=random_state)
    
    LOGGER.info(f"✓ Kandydaci: {[name for name, _, _ in candidates]}")
    
    # 7. Trening i selekcja
    LOGGER.info("Etap 7/8: Trening kandydatów i selekcja najlepszego")
    best_name, best_pipeline, best_metrics = _train_all_candidates(
        candidates=candidates,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type
    )
    
    LOGGER.info(f"✓ Najlepszy model: {best_name}")
    LOGGER.info(f"✓ Metryki: {best_metrics}")
    
    # Attach feature importances
    _attach_feature_importances(best_pipeline)
    
    # 8. Zapis modelu
    LOGGER.info("Etap 8/8: Zapis modelu")
    model_path, model_id = _save_model(
        pipeline=best_pipeline,
        target=target,
        problem_type=problem_type,
        best_name=best_name,
        metrics=best_metrics,
        columns=list(X_train.columns),
        column_roles=column_roles,
        random_state=random_state
    )
    
    LOGGER.info(f"✓ Model ID: {model_id}")
    LOGGER.info(f"✓ Ścieżka: {model_path}")
    
    # Podsumowanie
    elapsed = time.time() - start_time
    LOGGER.info("="*80)
    LOGGER.info(f"AutoML Pipeline - KONIEC (czas: {elapsed:.2f}s)")
    LOGGER.info("="*80)
    
    return best_pipeline, best_metrics, problem_type


# ========================================================================================
# UTILITIES
# ========================================================================================

def load_model(model_path: Union[str, pathlib.Path]) -> Pipeline:
    """
    Wczytuje zapisany model.
    
    Args:
        model_path: Ścieżka do pliku modelu
        
    Returns:
        Pipeline lub pełny payload
    """
    path = pathlib.Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model nie istnieje: {path}")
    
    try:
        payload = joblib.load(path)
        
        # Jeśli to ModelPayload, zwróć sam model
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"]
        
        # W przeciwnym razie zakładamy że to pipeline
        return payload
        
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać modelu: {e}")


def list_models() -> List[Dict[str, Any]]:
    """
    Zwraca listę wszystkich zapisanych modeli z rejestru.
    
    Returns:
        Lista słowników z informacjami o modelach
    """
    if not REGISTRY_FILE.exists():
        return []
    
    try:
        registry_data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        
        if not isinstance(registry_data, list):
            return []
        
        return registry_data
        
    except Exception as e:
        LOGGER.error(f"Nie udało się wczytać rejestru: {e}")
        return []


def get_model_info(model_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Zwraca informacje o zapisanym modelu.
    
    Args:
        model_path: Ścieżka do pliku modelu
        
    Returns:
        Słownik z metadanymi modelu
    """
    path = pathlib.Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model nie istnieje: {path}")
    
    try:
        payload = joblib.load(path)
        
        if isinstance(payload, dict):
            # Zwróć wszystko oprócz samego modelu
            info = {k: v for k, v in payload.items() if k != "model"}
            return info
        else:
            return {"error": "Model nie zawiera metadanych"}
            
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać informacji o modelu: {e}")