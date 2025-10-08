"""
Anomaly Detection Engine - Zaawansowana detekcja anomalii w danych.

Funkcjonalności:
- 4 algorytmy: IsolationForest, LOF, EllipticEnvelope, OneClassSVM
- Automatyczna imputacja i skalowanie
- Inteligentny fallback przy błędach
- Feature importance (Spearman correlation)
- Comprehensive metadata tracking
- Robust error handling
- Parameter validation and auto-correction
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Tuple, Literal, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Limity bezpieczeństwa
MIN_ROWS_FOR_DETECTION = 5
MAX_SAMPLE_FOR_CORRELATION = 50_000
DEFAULT_CONTAMINATION = 0.05
MIN_CONTAMINATION = 0.001
MAX_CONTAMINATION = 0.499

# Domyślne parametry modeli
DEFAULT_N_NEIGHBORS = 20
DEFAULT_NU = 0.1
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42

# Typy metod
AnomalyMethod = Literal["iforest", "lof", "elliptic", "ocsvm"]

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "anomaly_detector", level: int = logging.INFO) -> logging.Logger:
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
class AnomalyParams:
    """Parametry dla detekcji anomalii."""
    contamination: float | str
    method: AnomalyMethod
    scale: bool
    random_state: int
    n_neighbors: int
    assume_centered: bool
    nu: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "contamination": self.contamination,
            "method": self.method,
            "scale": self.scale,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors,
            "assume_centered": self.assume_centered,
            "nu": self.nu
        }


@dataclass
class AnomalyMetadata:
    """Metadane detekcji anomalii."""
    method: str
    contamination: float | str
    n_rows: int
    n_num_cols: int
    dropped_constant_cols: list[str]
    n_anomalies: int
    anomaly_percentage: float
    feature_influence: Dict[str, float]
    scaled: bool
    imputer: str
    random_state: int
    score_range: Tuple[float, float]
    params: Dict[str, Any]
    warnings: Optional[list[str]]
    fallback_used: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "method": self.method,
            "contamination": self.contamination,
            "n_rows": self.n_rows,
            "n_num_cols": self.n_num_cols,
            "dropped_constant_cols": self.dropped_constant_cols,
            "n_anomalies": self.n_anomalies,
            "anomaly_percentage": self.anomaly_percentage,
            "feature_influence": self.feature_influence,
            "scaled": self.scaled,
            "imputer": self.imputer,
            "random_state": self.random_state,
            "score_range": self.score_range,
            "params": self.params,
            "warnings": self.warnings,
            "fallback_used": self.fallback_used
        }


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _make_empty_result(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    """
    Tworzy pusty wynik z metadanymi błędu.
    
    Args:
        df: DataFrame bazowy
        reason: Powód błędu
        
    Returns:
        DataFrame z pustymi kolumnami anomalii
    """
    result = df.copy()
    result["_anomaly_score"] = np.nan
    result["_is_anomaly"] = pd.Series(
        np.zeros(len(df), dtype="int8"), 
        index=df.index
    ).astype("Int8")
    
    result.attrs["anomaly_meta"] = {
        "error": reason,
        "n_rows": len(df),
        "method": "none",
        "n_anomalies": 0
    }
    
    LOGGER.warning(f"Anomaly detection failed: {reason}")
    return result


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalizuje array do zakresu [0, 1].
    
    Args:
        arr: Array do normalizacji
        
    Returns:
        Znormalizowany array
    """
    arr_float = np.asarray(arr, dtype=float)
    
    if arr_float.size == 0:
        return arr_float
    
    arr_min = np.nanmin(arr_float)
    arr_max = np.nanmax(arr_float)
    arr_range = arr_max - arr_min
    
    if arr_range <= 1e-12:
        return np.zeros_like(arr_float)
    
    return (arr_float - arr_min) / arr_range


def _validate_contamination(
    contamination: float | str,
    method: str
) -> Tuple[float | str, Optional[str]]:
    """
    Waliduje parametr contamination.
    
    Args:
        contamination: Wartość do walidacji
        method: Metoda detekcji
        
    Returns:
        Tuple (validated_value, warning_message)
    """
    # 'auto' dla IsolationForest
    if contamination == "auto":
        if method == "iforest":
            return "auto", None
        else:
            # EllipticEnvelope i inne nie wspierają 'auto'
            warning = f"{method} nie wspiera contamination='auto', używam {DEFAULT_CONTAMINATION}"
            return DEFAULT_CONTAMINATION, warning
    
    # Walidacja float
    if isinstance(contamination, (int, float)):
        contam_float = float(contamination)
        
        if contam_float <= MIN_CONTAMINATION or contam_float >= MAX_CONTAMINATION:
            warning = (
                f"contamination={contam_float} poza zakresem "
                f"({MIN_CONTAMINATION}, {MAX_CONTAMINATION}), "
                f"skorygowano do {DEFAULT_CONTAMINATION}"
            )
            return DEFAULT_CONTAMINATION, warning
        
        return contam_float, None
    
    # Nieobsługiwany typ
    warning = f"Nieprawidłowy typ contamination: {type(contamination)}, używam {DEFAULT_CONTAMINATION}"
    return DEFAULT_CONTAMINATION, warning


def _validate_n_neighbors(
    n_neighbors: int,
    n_rows: int,
    method: str
) -> Tuple[int, Optional[str]]:
    """
    Waliduje parametr n_neighbors dla LOF.
    
    Args:
        n_neighbors: Liczba sąsiadów
        n_rows: Liczba wierszy w danych
        method: Metoda detekcji
        
    Returns:
        Tuple (validated_value, warning_message)
    """
    if method != "lof":
        return n_neighbors, None
    
    # Minimum 2, maximum n_rows - 1
    min_neighbors = 2
    max_neighbors = max(2, n_rows - 1)
    
    if n_neighbors < min_neighbors or n_neighbors > max_neighbors:
        validated = np.clip(n_neighbors, min_neighbors, max_neighbors)
        warning = (
            f"n_neighbors={n_neighbors} poza zakresem [{min_neighbors}, {max_neighbors}], "
            f"skorygowano do {validated}"
        )
        return int(validated), warning
    
    return n_neighbors, None


def _validate_nu(nu: float, method: str) -> Tuple[float, Optional[str]]:
    """
    Waliduje parametr nu dla OneClassSVM.
    
    Args:
        nu: Parametr nu
        method: Metoda detekcji
        
    Returns:
        Tuple (validated_value, warning_message)
    """
    if method != "ocsvm":
        return nu, None
    
    if nu <= 0.0 or nu >= 1.0:
        warning = f"nu={nu} poza zakresem (0, 1), skorygowano do {DEFAULT_NU}"
        return DEFAULT_NU, warning
    
    return nu, None


def _sanitize_parameters(
    params: AnomalyParams,
    n_rows: int
) -> Tuple[AnomalyParams, list[str]]:
    """
    Sanityzuje wszystkie parametry i zbiera ostrzeżenia.
    
    Args:
        params: Parametry wejściowe
        n_rows: Liczba wierszy w danych
        
    Returns:
        Tuple (sanitized_params, warnings)
    """
    warnings_list: list[str] = []
    method = params.method.lower().strip()
    
    # Walidacja contamination
    contam_valid, contam_warn = _validate_contamination(params.contamination, method)
    if contam_warn:
        warnings_list.append(contam_warn)
    
    # Walidacja n_neighbors
    nn_valid, nn_warn = _validate_n_neighbors(params.n_neighbors, n_rows, method)
    if nn_warn:
        warnings_list.append(nn_warn)
    
    # Walidacja nu
    nu_valid, nu_warn = _validate_nu(params.nu, method)
    if nu_warn:
        warnings_list.append(nu_warn)
    
    # Nowe parametry
    sanitized = AnomalyParams(
        contamination=contam_valid,
        method=method,  # type: ignore
        scale=params.scale,
        random_state=params.random_state,
        n_neighbors=nn_valid,
        assume_centered=params.assume_centered,
        nu=nu_valid
    )
    
    return sanitized, warnings_list


# ========================================================================================
# PREPROCESSING
# ========================================================================================

def _prepare_numeric_data(
    df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], list[str], Optional[str]]:
    """
    Przygotowuje dane numeryczne do detekcji.
    
    Args:
        df: DataFrame wejściowy
        
    Returns:
        Tuple (numeric_df, dropped_constant_cols, error_reason)
    """
    # Wybierz kolumny numeryczne
    numeric_df = df.select_dtypes(include=np.number).copy()
    
    if numeric_df.empty:
        return None, [], "no_numeric_columns"
    
    # Zamień Inf na NaN
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    
    # Usuń kolumny stałe
    constant_cols = [
        col for col in numeric_df.columns
        if numeric_df[col].nunique(dropna=True) <= 1
    ]
    
    if constant_cols:
        LOGGER.debug(f"Usuwam kolumny stałe: {constant_cols}")
        numeric_df = numeric_df.drop(columns=constant_cols)
    
    if numeric_df.empty:
        return None, constant_cols, "only_constant_columns"
    
    return numeric_df, constant_cols, None


def _impute_and_scale(
    X: pd.DataFrame,
    scale: bool
) -> Tuple[np.ndarray, Optional[RobustScaler]]:
    """
    Imputuje braki i opcjonalnie skaluje dane.
    
    Args:
        X: DataFrame numeryczny
        scale: Czy skalować dane
        
    Returns:
        Tuple (transformed_array, scaler_or_none)
    """
    # Imputacja
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    # Skalowanie
    scaler = None
    if scale:
        scaler = RobustScaler()
        X_imputed = scaler.fit_transform(X_imputed)
        LOGGER.debug("Dane przeskalowane używając RobustScaler")
    
    return X_imputed, scaler


# ========================================================================================
# MODEL TRAINING
# ========================================================================================

def _fit_isolation_forest(
    X: np.ndarray,
    contamination: float | str,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trenuje IsolationForest.
    
    Args:
        X: Dane do trenowania
        contamination: Parametr zanieczyszczenia
        random_state: Random state
        
    Returns:
        Tuple (decision_scores, is_anomaly)
    """
    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_samples="auto",
        n_jobs=-1,
        bootstrap=False,
        warm_start=False
    )
    
    clf.fit(X)
    
    # Wyższy score = bardziej normalny
    scores = clf.decision_function(X)
    predictions = clf.predict(X)  # 1 = normal, -1 = anomaly
    
    is_anomaly = (predictions == -1).astype(np.int8)
    
    return scores, is_anomaly


def _fit_local_outlier_factor(
    X: np.ndarray,
    contamination: float | str,
    n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trenuje LocalOutlierFactor.
    
    Args:
        X: Dane do trenowania
        contamination: Parametr zanieczyszczenia
        n_neighbors: Liczba sąsiadów
        
    Returns:
        Tuple (decision_scores, is_anomaly)
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination if isinstance(contamination, float) else "auto",
        novelty=False,
        n_jobs=-1,
        algorithm="auto"
    )
    
    predictions = lof.fit_predict(X)  # 1 = normal, -1 = anomaly
    is_anomaly = (predictions == -1).astype(np.int8)
    
    # Negative outlier factor: bardziej ujemny = anomalia
    # Normalizujemy do [0, 1] gdzie wyższy = normalny
    negative_outlier_factors = lof.negative_outlier_factor_
    scores = _minmax_normalize(negative_outlier_factors)
    
    return scores, is_anomaly


def _fit_elliptic_envelope(
    X: np.ndarray,
    contamination: float,
    assume_centered: bool,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trenuje EllipticEnvelope.
    
    Args:
        X: Dane do trenowania
        contamination: Parametr zanieczyszczenia (tylko float)
        assume_centered: Czy zakładać wyśrodkowane dane
        random_state: Random state
        
    Returns:
        Tuple (decision_scores, is_anomaly)
    """
    ee = EllipticEnvelope(
        contamination=contamination,
        support_fraction=None,
        assume_centered=assume_centered,
        random_state=random_state
    )
    
    ee.fit(X)
    
    scores = ee.decision_function(X)
    predictions = ee.predict(X)  # 1 = normal, -1 = anomaly
    
    is_anomaly = (predictions == -1).astype(np.int8)
    
    return scores, is_anomaly


def _fit_one_class_svm(
    X: np.ndarray,
    nu: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trenuje OneClassSVM.
    
    Args:
        X: Dane do trenowania
        nu: Parametr nu
        
    Returns:
        Tuple (decision_scores, is_anomaly)
    """
    ocsvm = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma="scale"
    )
    
    ocsvm.fit(X)
    
    scores = ocsvm.decision_function(X)
    predictions = ocsvm.predict(X)  # 1 = normal, -1 = anomaly
    
    is_anomaly = (predictions == -1).astype(np.int8)
    
    return scores, is_anomaly


def _train_model(
    X: np.ndarray,
    params: AnomalyParams
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Trenuje wybrany model z automatycznym fallbackiem.
    
    Args:
        X: Dane do trenowania
        params: Parametry modelu
        
    Returns:
        Tuple (scores, is_anomaly, fallback_method)
    """
    method = params.method
    fallback = None
    
    try:
        if method == "iforest":
            scores, is_anom = _fit_isolation_forest(
                X, params.contamination, params.random_state
            )
            
        elif method == "lof":
            scores, is_anom = _fit_local_outlier_factor(
                X, params.contamination, params.n_neighbors
            )
            
        elif method == "elliptic":
            # EllipticEnvelope wymaga float contamination
            contam = params.contamination if isinstance(params.contamination, float) else DEFAULT_CONTAMINATION
            scores, is_anom = _fit_elliptic_envelope(
                X, contam, params.assume_centered, params.random_state
            )
            
        elif method == "ocsvm":
            scores, is_anom = _fit_one_class_svm(X, params.nu)
            
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        
        return scores, is_anom, fallback
        
    except Exception as e:
        LOGGER.exception(f"Metoda '{method}' nie powiodła się, fallback do IsolationForest")
        
        # Fallback do IsolationForest
        try:
            scores, is_anom = _fit_isolation_forest(
                X, DEFAULT_CONTAMINATION, params.random_state
            )
            fallback = f"{method}→iforest"
            return scores, is_anom, fallback
            
        except Exception as fallback_error:
            LOGGER.exception("Fallback do IsolationForest również się nie powiódł")
            raise RuntimeError(f"Wszystkie metody zawiodły: {e}, fallback: {fallback_error}")


# ========================================================================================
# FEATURE IMPORTANCE
# ========================================================================================

def _compute_feature_influence(
    numeric_df: pd.DataFrame,
    scores: np.ndarray
) -> Dict[str, float]:
    """
    Oblicza wpływ cech na scores używając korelacji Spearmana.
    
    Args:
        numeric_df: DataFrame z danymi numerycznymi
        scores: Scores z detekcji
        
    Returns:
        Słownik {feature: abs_correlation}
    """
    try:
        # Przygotuj dane
        df_with_scores = numeric_df.copy()
        df_with_scores["_score"] = scores
        
        # Sampeluj jeśli za dużo
        if len(df_with_scores) > MAX_SAMPLE_FOR_CORRELATION:
            df_with_scores = df_with_scores.sample(
                MAX_SAMPLE_FOR_CORRELATION,
                random_state=DEFAULT_RANDOM_STATE
            )
        
        # Korelacja Spearmana
        correlations = df_with_scores.corr(method="spearman")["_score"]
        correlations = correlations.drop("_score", errors="ignore")
        
        # Absolute values, sorted
        abs_correlations = correlations.abs().sort_values(ascending=False)
        
        return abs_correlations.to_dict()
        
    except Exception as e:
        LOGGER.warning(f"Nie udało się obliczyć feature influence: {e}")
        return {}


# ========================================================================================
# MAIN API
# ========================================================================================

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float | str = DEFAULT_CONTAMINATION,
    method: AnomalyMethod = "iforest",
    scale: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    assume_centered: bool = False,
    nu: float = DEFAULT_NU,
) -> pd.DataFrame:
    """
    Wykrywa anomalie w danych używając wybranego algorytmu.
    
    Obsługiwane metody:
    - **iforest**: IsolationForest (najszybszy, dobry dla wielowymiarowych danych)
    - **lof**: LocalOutlierFactor (dobry dla gęstości lokalnej)
    - **elliptic**: EllipticEnvelope (zakłada rozkład Gaussowski)
    - **ocsvm**: OneClassSVM (kernel methods, wolniejszy)
    
    Args:
        df: DataFrame z danymi
        contamination: Oczekiwany procent anomalii (0.001-0.499) lub "auto"
        method: Metoda detekcji ("iforest" | "lof" | "elliptic" | "ocsvm")
        scale: Czy skalować dane (zalecane dla lof/ocsvm)
        random_state: Random state dla reprodukowalności
        n_neighbors: Liczba sąsiadów dla LOF (domyślnie 20)
        assume_centered: Czy zakładać wyśrodkowane dane dla EllipticEnvelope
        nu: Parametr nu dla OneClassSVM (0-1, domyślnie 0.1)
        
    Returns:
        DataFrame z dwiema dodatkowymi kolumnami:
        - `_anomaly_score`: Score anomalii (wyższy = bardziej normalny)
        - `_is_anomaly`: Binary flag (0 = normal, 1 = anomaly)
        
        Dodatkowo metadane w `df.attrs["anomaly_meta"]`
        
    Raises:
        ValueError: Jeśli parametry są nieprawidłowe
        RuntimeError: Jeśli wszystkie metody zawiodły
        
    Example:
        >>> result = detect_anomalies(df, contamination=0.1, method="iforest")
        >>> anomalies = result[result["_is_anomaly"] == 1]
        >>> print(result.attrs["anomaly_meta"])
    """
    # Walidacja wejścia
    if df is None or not isinstance(df, pd.DataFrame):
        return _make_empty_result(pd.DataFrame(), "invalid_dataframe")
    
    if df.empty:
        return _make_empty_result(df, "empty_dataframe")
    
    # Przygotowanie danych numerycznych
    numeric_df, constant_cols, error = _prepare_numeric_data(df)
    
    if error:
        return _make_empty_result(df, error)
    
    assert numeric_df is not None  # dla type checker
    
    # Sprawdź minimalną liczbę wierszy
    n_rows = len(numeric_df)
    if n_rows < MIN_ROWS_FOR_DETECTION:
        return _make_empty_result(df, f"too_few_rows({n_rows}<{MIN_ROWS_FOR_DETECTION})")
    
    # Sanityzacja parametrów
    params = AnomalyParams(
        contamination=contamination,
        method=method,
        scale=scale,
        random_state=random_state,
        n_neighbors=n_neighbors,
        assume_centered=assume_centered,
        nu=nu
    )
    
    sanitized_params, param_warnings = _sanitize_parameters(params, n_rows)
    
    # Auto-scale dla LOF i OCSVM jeśli nie wyspecyfikowano
    effective_scale = sanitized_params.scale or sanitized_params.method in ("lof", "ocsvm")
    if effective_scale and not sanitized_params.scale:
        param_warnings.append(f"Auto-skalowanie włączone dla {sanitized_params.method}")
    
    # Imputacja i skalowanie
    try:
        X, scaler = _impute_and_scale(numeric_df, effective_scale)
    except Exception as e:
        LOGGER.exception("Błąd podczas preprocessingu danych")
        return _make_empty_result(df, f"preprocessing_failed: {e}")
    
    # Trenowanie modelu
    try:
        scores, is_anomaly, fallback = _train_model(X, sanitized_params)
    except Exception as e:
        LOGGER.exception("Wszystkie metody detekcji zawiodły")
        return _make_empty_result(df, f"model_training_failed: {e}")
    
    # Feature influence
    feature_influence = _compute_feature_influence(numeric_df, scores)
    
    # Budowanie wyniku
    result = df.copy()
    result["_anomaly_score"] = pd.Series(scores, index=numeric_df.index)
    result["_is_anomaly"] = pd.Series(is_anomaly, index=numeric_df.index).astype("Int8")
    
    # Metadane
    n_anomalies = int(result["_is_anomaly"].sum())
    anomaly_pct = (n_anomalies / n_rows * 100.0) if n_rows > 0 else 0.0
    
    score_min = float(np.nanmin(scores)) if scores.size > 0 else np.nan
    score_max = float(np.nanmax(scores)) if scores.size > 0 else np.nan
    
    method_name = sanitized_params.method
    if fallback:
        method_name = f"{sanitized_params.method} (fallback→iforest)"
        param_warnings.append(f"Fallback: {fallback}")
    
    metadata = AnomalyMetadata(
        method=method_name,
        contamination=sanitized_params.contamination,
        n_rows=len(df),
        n_num_cols=numeric_df.shape[1],
        dropped_constant_cols=constant_cols,
        n_anomalies=n_anomalies,
        anomaly_percentage=round(anomaly_pct, 2),
        feature_influence=feature_influence,
        scaled=effective_scale,
        imputer="median",
        random_state=sanitized_params.random_state,
        score_range=(score_min, score_max),
        params={
            "n_neighbors": sanitized_params.n_neighbors if method == "lof" else None,
            "assume_centered": sanitized_params.assume_centered if method == "elliptic" else None,
            "nu": sanitized_params.nu if method == "ocsvm" else None,
        },
        warnings=param_warnings if param_warnings else None,
        fallback_used=fallback
    )
    
    result.attrs["anomaly_meta"] = metadata.to_dict()
    
    LOGGER.info(
        f"Detekcja zakończona: {n_anomalies}/{n_rows} anomalii "
        f"({anomaly_pct:.1f}%) metodą {method_name}"
    )
    
    return result


# ========================================================================================
# UTILITIES
# ========================================================================================

def get_anomaly_summary(result: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca podsumowanie wykrytych anomalii.
    
    Args:
        result: DataFrame zwrócony z detect_anomalies
        
    Returns:
        Słownik z podsumowaniem
    """
    if "_is_anomaly" not in result.columns:
        return {"error": "Brak kolumny _is_anomaly w DataFrame"}
    
    meta = result.attrs.get("anomaly_meta", {})
    
    anomalies = result[result["_is_anomaly"] == 1]
    
    return {
        "total_rows": len(result),
        "n_anomalies": len(anomalies),
        "anomaly_percentage": meta.get("anomaly_percentage", 0.0),
        "method": meta.get("method", "unknown"),
        "score_range": meta.get("score_range", (np.nan, np.nan)),
        "top_features": dict(list(meta.get("feature_influence", {}).items())[:5])
    }


def filter_anomalies(
    result: pd.DataFrame,
    include_anomalies: bool = True
) -> pd.DataFrame:
    """
    Filtruje DataFrame do anomalii lub normalnych obserwacji.
    
    Args:
        result: DataFrame zwrócony z detect_anomalies
        include_anomalies: True = tylko anomalie, False = tylko normalne
        
    Returns:
        Przefiltrowany DataFrame
    """
    if "_is_anomaly" not in result.columns:
        LOGGER.warning("Brak kolumny _is_anomaly w DataFrame")
        return result
    
    if include_anomalies:
        return result[result["_is_anomaly"] == 1].copy()
    else:
        return result[result["_is_anomaly"] == 0].copy()