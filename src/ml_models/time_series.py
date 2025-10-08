"""
Time Series Forecasting Engine v2 - Zaawansowane prognozowanie z Prophet.

Funkcjonalności:
- Prophet z pełną konfiguracją
- Automatyczna detekcja kolumny czasu i częstotliwości
- External regressors (auto-select lub manual)
- Multiple growth types (linear, logistic, flat)
- Outlier detection i clipping
- Comprehensive metrics (RMSE, MAE, R², sMAPE, MASE)
- Rolling backtest dla walidacji
- Country holidays support
- Robust preprocessing
- Detailed metadata tracking
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal, Callable
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress Prophet's verbose output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")

# Optional helper import
try:
    from ..utils.helpers import ensure_datetime_index
    HAS_HELPER = True
except ImportError:
    HAS_HELPER = False

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Hinty dla kolumn z datami
DATE_COLUMN_HINTS = (
    "date", "time", "timestamp", "datetime", "data", "czas",
    "dt", "day", "month", "year", "period", "fecha"
)

# Limity bezpieczeństwa
MIN_TIME_POINTS = 10
MAX_TIME_POINTS = 100_000
MAX_HORIZON_STEPS = 10_000
MIN_VALIDATION_SIZE = 5
DEFAULT_HORIZON = 12

# Parametry Prophet
DEFAULT_CHANGEPOINT_PRIOR_SCALE = 0.1
DEFAULT_SEASONALITY_MODE = "additive"
DEFAULT_GROWTH = "linear"

# External regressors
MAX_AUTO_REGRESSORS = 15
MIN_REGRESSOR_UNIQUE = 3

# Outlier detection
OUTLIER_METHODS = ("iqr", "zscore")
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 4.0

# Frequency aliases
FREQ_ALIASES = {
    "Y": "Y", "A": "Y", "YE": "Y", "AS": "Y",
    "Q": "Q", "QS": "QS", "QE": "Q",
    "M": "MS", "MS": "MS", "ME": "MS",
    "W": "W",
    "D": "D",
    "H": "H",
    "T": "min", "MIN": "min",
    "S": "S",
}

# Seasonal periods for MASE
SEASONAL_PERIODS = {
    "Y": 1, "Q": 4, "MS": 12, "M": 12,
    "W": 52, "D": 7, "H": 24, "MIN": 60, "S": 60
}

# Types
GrowthType = Literal["linear", "logistic", "flat"]
SeasonalityMode = Literal["additive", "multiplicative"]
OutlierMethod = Literal["iqr", "zscore"]

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "forecasting_v2", level: int = logging.INFO) -> logging.Logger:
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
class ForecastMetrics:
    """Metryki prognozy."""
    rmse: float
    mae: float
    r2: float
    smape: float
    mase: float
    
    def to_dict(self) -> Dict[str, float]:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass(frozen=True)
class BacktestFold:
    """Wyniki pojedynczego foldu backtestingu."""
    fold: int
    train_end: str
    test_len: int
    metrics: ForecastMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "fold": self.fold,
            "train_end": self.train_end,
            "test_len": self.test_len,
            **self.metrics.to_dict()
        }


@dataclass
class ForecastOptions:
    """Opcje dla forecasting."""
    seasonality_mode: SeasonalityMode = DEFAULT_SEASONALITY_MODE
    changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR_SCALE
    growth: GrowthType = DEFAULT_GROWTH
    cap: Optional[float] = None
    floor: Optional[float] = None
    extra_regressors: Union[None, str, List[str]] = None
    holidays: Optional[pd.DataFrame] = None
    country_holidays: Optional[str] = None
    include_history: bool = False
    freq: Optional[str] = None
    outlier_clip: Optional[OutlierMethod] = None
    backtest_folds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "growth": self.growth,
            "cap": self.cap,
            "floor": self.floor,
            "extra_regressors": self.extra_regressors,
            "has_holidays": self.holidays is not None,
            "country_holidays": self.country_holidays,
            "include_history": self.include_history,
            "freq": self.freq,
            "outlier_clip": self.outlier_clip,
            "backtest_folds": self.backtest_folds
        }


# ========================================================================================
# DATETIME DETECTION
# ========================================================================================

def _detect_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback dla detekcji indeksu czasowego.
    
    Args:
        df: DataFrame do przetworzenia
        
    Returns:
        DataFrame z DatetimeIndex
    """
    df_copy = df.copy()
    
    # Już jest DatetimeIndex
    if isinstance(df_copy.index, pd.DatetimeIndex):
        return df_copy.sort_index()
    
    # Szukaj po hintach
    candidate = None
    for col in df_copy.columns:
        col_lower = str(col).lower()
        if any(hint in col_lower for hint in DATE_COLUMN_HINTS):
            candidate = col
            break
    
    # Próba konwersji kandydata
    if candidate:
        try:
            datetime_series = pd.to_datetime(
                df_copy[candidate],
                errors="coerce",
                infer_datetime_format=True
            )
            
            if datetime_series.notna().mean() > 0.6:
                df_copy[candidate] = datetime_series
                df_copy = df_copy.set_index(candidate).sort_index()
                LOGGER.debug(f"Ustawiono DatetimeIndex z kolumny: {candidate}")
                return df_copy
        except Exception as e:
            LOGGER.debug(f"Nie udało się skonwertować {candidate}: {e}")
    
    # Fallback: pierwsza kolumna
    if len(df_copy.columns) > 0:
        first_col = df_copy.columns[0]
        try:
            datetime_series = pd.to_datetime(
                df_copy[first_col],
                errors="coerce",
                infer_datetime_format=True
            )
            
            if datetime_series.notna().mean() > 0.6:
                df_copy[first_col] = datetime_series
                df_copy = df_copy.set_index(first_col).sort_index()
                LOGGER.debug(f"Ustawiono DatetimeIndex z pierwszej kolumny: {first_col}")
                return df_copy
        except Exception as e:
            LOGGER.debug(f"Nie udało się skonwertować pierwszej kolumny: {e}")
    
    return df


# ========================================================================================
# FREQUENCY INFERENCE
# ========================================================================================

def _infer_frequency(idx: pd.DatetimeIndex) -> str:
    """
    Wykrywa częstotliwość szeregu czasowego.
    
    Args:
        idx: DatetimeIndex
        
    Returns:
        Kod częstotliwości
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return "D"
    
    # Próba automatycznego wykrycia
    try:
        freq = pd.infer_freq(idx)
        if freq:
            normalized = _normalize_frequency(freq)
            LOGGER.debug(f"Wykryto częstotliwość: {freq} → {normalized}")
            return normalized
    except Exception as e:
        LOGGER.debug(f"pd.infer_freq nie powiodło się: {e}")
    
    # Fallback: heurystyka
    return _infer_from_median_diff(idx)


def _normalize_frequency(freq: str) -> str:
    """Normalizuje alias częstotliwości."""
    freq_upper = freq.upper()
    
    # Prefixy
    if freq_upper.startswith(("A", "Y")):
        return "Y"
    if freq_upper.startswith("Q"):
        return "Q"
    if freq_upper.startswith("M"):
        return "MS"
    if freq_upper.startswith("W"):
        return "W"
    if freq_upper.startswith("D") or freq_upper.startswith("B"):
        return "D"
    if freq_upper.startswith("H"):
        return "H"
    if freq_upper.startswith("T"):
        return "MIN"
    if freq_upper.startswith("S"):
        return "S"
    
    return FREQ_ALIASES.get(freq_upper, "D")


def _infer_from_median_diff(idx: pd.DatetimeIndex) -> str:
    """Wykrywa częstotliwość z mediany różnic."""
    if len(idx) < 2:
        return "D"
    
    diffs = np.diff(idx.view("i8"))
    median_seconds = float(np.median(diffs) / 1e9)
    
    # Progi
    if median_seconds < 60:
        return "S"
    elif median_seconds < 3600:
        return "MIN"
    elif median_seconds < 86400:
        return "H"
    elif median_seconds < 7 * 86400:
        return "D"
    elif median_seconds < 28 * 86400:
        return "W"
    elif median_seconds < 92 * 86400:
        return "MS"
    elif median_seconds < 366 * 86400:
        return "Q"
    else:
        return "Y"


def _get_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    """Określa które seasonalities mają sens."""
    freq_upper = freq.upper()
    
    yearly = True
    weekly = freq_upper in ("D", "H", "MIN", "S")
    daily = freq_upper in ("H", "MIN", "S")
    
    return yearly, weekly, daily


def _get_prophet_freq_alias(freq: str) -> str:
    """Konwertuje do formatu Prophet."""
    return FREQ_ALIASES.get(freq.upper(), "D")


def _get_seasonal_period(freq: str) -> int:
    """Zwraca okres sezonowy dla MASE."""
    return SEASONAL_PERIODS.get(freq.upper(), 1)


# ========================================================================================
# DATA PREPARATION
# ========================================================================================

def _prepare_target(series: pd.Series) -> pd.Series:
    """Przygotowuje target do modelowania."""
    return pd.to_numeric(series, errors="coerce")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści DataFrame."""
    df_clean = df[df.index.notna()].copy()
    df_clean = df_clean.sort_index()
    
    # Duplikaty - agreguj po średniej
    if df_clean.index.has_duplicates:
        n_dupes = df_clean.index.duplicated().sum()
        LOGGER.warning(f"Znaleziono {n_dupes} duplikatów, agreguję po średniej")
        df_clean = df_clean.groupby(df_clean.index).mean(numeric_only=True)
    
    return df_clean


def _clip_outliers(
    series: pd.Series,
    method: OutlierMethod
) -> pd.Series:
    """
    Clipuje outliery w serii.
    
    Args:
        series: Serie do przetworzenia
        method: Metoda ("iqr" lub "zscore")
        
    Returns:
        Serie z clipowanymi outlierami
    """
    series_clean = series.copy()
    
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        
        series_clean = series_clean.clip(lower=lower_bound, upper=upper_bound)
        
        n_clipped = ((series < lower_bound) | (series > upper_bound)).sum()
        LOGGER.debug(f"IQR clipping: {n_clipped} wartości")
    
    elif method == "zscore":
        mean = series.mean(skipna=True)
        std = series.std(skipna=True) or 1.0
        
        z_scores = (series - mean) / std
        series_clean = series_clean.mask(z_scores.abs() > ZSCORE_THRESHOLD, np.nan)
        
        n_masked = (z_scores.abs() > ZSCORE_THRESHOLD).sum()
        LOGGER.debug(f"Z-score masking: {n_masked} wartości")
    
    return series_clean


# ========================================================================================
# EXTERNAL REGRESSORS
# ========================================================================================

def _select_auto_regressors(
    df: pd.DataFrame,
    target: str,
    max_regressors: int = MAX_AUTO_REGRESSORS
) -> List[str]:
    """Automatyczny wybór regresorów."""
    candidates: List[Tuple[str, float]] = []
    
    for col in df.columns:
        if col == target:
            continue
        
        series = df[col]
        
        if not pd.api.types.is_numeric_dtype(series):
            continue
        
        n_unique = series.nunique(dropna=True)
        if n_unique < MIN_REGRESSOR_UNIQUE:
            continue
        
        # Score: std (większe = bardziej zmienne)
        std = float(series.std(skipna=True) or 0.0)
        candidates.append((col, std))
    
    # Sortuj i weź top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [col for col, _ in candidates[:max_regressors]]
    
    if selected:
        LOGGER.debug(f"Auto-wybrano {len(selected)} regresorów: {selected}")
    
    return selected


# ========================================================================================
# LOGISTIC GROWTH
# ========================================================================================

def _compute_logistic_bounds(
    y: pd.Series,
    cap: Optional[float],
    floor: Optional[float]
) -> Tuple[float, float]:
    """Oblicza cap i floor dla logistic growth."""
    y_clean = y.dropna()
    
    if len(y_clean) == 0:
        y_max, y_min = 1.0, 0.0
    else:
        y_max = float(y_clean.max())
        y_min = float(y_clean.min())
    
    # Cap
    if cap is None or cap <= y_max:
        computed_cap = max(y_max * 1.2, y_max + max(abs(y_max) * 0.1, 1.0))
    else:
        computed_cap = float(cap)
    
    # Floor
    if floor is None or floor >= y_min or floor >= computed_cap:
        computed_floor = min(y_min * 0.8, y_min - max(abs(y_min) * 0.1, 1.0))
        computed_floor = min(computed_floor, computed_cap - 1e-6)
    else:
        computed_floor = float(floor)
    
    LOGGER.debug(f"Logistic bounds: floor={computed_floor:.2f}, cap={computed_cap:.2f}")
    return computed_cap, computed_floor


# ========================================================================================
# METRICS
# ========================================================================================

def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Oblicza sMAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1.0, denominator)
    
    return float(np.mean(numerator / denominator) * 100.0)


def _compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonal_period: int
) -> float:
    """Oblicza MASE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_ins = np.asarray(y_insample, dtype=float)
    
    m = max(int(seasonal_period), 1)
    
    if len(y_ins) <= m:
        return float("nan")
    
    # Seasonal naive forecast
    scale = np.mean(np.abs(y_ins[m:] - y_ins[:-m]))
    
    if scale == 0 or np.isnan(scale):
        return float("nan")
    
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / scale)


def _compute_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """Oblicza RMSE, MAE, R²."""
    error = y_true - y_pred
    
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    
    # R²
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    
    if ss_tot == 0 or np.isnan(ss_tot):
        r2 = float("nan")
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    
    return rmse, mae, r2


def _compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonal_period: int
) -> ForecastMetrics:
    """Oblicza wszystkie metryki."""
    rmse, mae, r2 = _compute_basic_metrics(y_true, y_pred)
    smape = _compute_smape(y_true, y_pred)
    mase = _compute_mase(y_true, y_pred, y_insample, seasonal_period)
    
    return ForecastMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        smape=smape,
        mase=mase
    )


# ========================================================================================
# MODEL BUILDING
# ========================================================================================

def _build_prophet_model(
    options: ForecastOptions,
    freq: str,
    chosen_regressors: List[str]
) -> Prophet:
    """Buduje model Prophet."""
    yearly, weekly, daily = _get_seasonality_flags(freq)
    
    model = Prophet(
        seasonality_mode=options.seasonality_mode,
        changepoint_prior_scale=options.changepoint_prior_scale,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        growth=options.growth,
        holidays=options.holidays,
        uncertainty_samples=1000
    )
    
    # Country holidays
    if options.country_holidays:
        try:
            model.add_country_holidays(country_name=options.country_holidays)
            LOGGER.debug(f"Dodano święta kraju: {options.country_holidays}")
        except Exception as e:
            LOGGER.warning(f"Nie udało się dodać świąt {options.country_holidays}: {e}")
    
    # Dodatkowe seasonalities
    if freq in ("MS", "M"):
        model.add_seasonality(name="monthly", period=30.5, fourier_order=8)
        LOGGER.debug("Dodano monthly seasonality")
    
    if freq in ("Q", "QS"):
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=4)
        LOGGER.debug("Dodano quarterly seasonality")
    
    # Regressors
    for reg in chosen_regressors:
        model.add_regressor(reg, standardize="auto")
        LOGGER.debug(f"Dodano regressor: {reg}")
    
    return model


# ========================================================================================
# HOLDOUT VALIDATION
# ========================================================================================

def _holdout_validation(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    options: ForecastOptions,
    freq: str,
    chosen_regressors: List[str]
) -> ForecastMetrics:
    """Walidacja na holdout set."""
    n = len(df)
    test_len = max(1, min(horizon, max(int(np.ceil(0.2 * n)), 1)))
    
    train_df = df.iloc[:-test_len].copy()
    test_df = df.iloc[-test_len:].copy()
    
    # Przygotuj dane treningowe
    y_train = _prepare_target(train_df[target]).dropna()
    data_train = pd.DataFrame({
        "ds": train_df.index,
        "y": y_train
    }).dropna(subset=["y"])
    
    # Logistic growth
    if options.growth == "logistic":
        cap, floor = _compute_logistic_bounds(
            data_train["y"],
            options.cap,
            options.floor
        )
        data_train["cap"] = cap
        data_train["floor"] = floor
    else:
        cap, floor = None, None
    
    # Regressors
    for reg in chosen_regressors:
        data_train[reg] = pd.to_numeric(train_df[reg], errors="coerce").reindex(data_train["ds"]).values
    
    # Build and fit
    model = _build_prophet_model(options, freq, chosen_regressors)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data_train)
    
    # Future
    freq_alias = _get_prophet_freq_alias(freq)
    future = model.make_future_dataframe(
        periods=test_len,
        freq=freq_alias,
        include_history=False
    )
    
    # Logistic bounds
    if options.growth == "logistic":
        future["cap"] = cap
        future["floor"] = floor
    
    # Forward-fill regressors
    if chosen_regressors:
        aux = data_train.set_index("ds")[chosen_regressors]
        future = future.set_index("ds")
        for reg in chosen_regressors:
            future[reg] = aux[reg].reindex(future.index).ffill()
        future = future.reset_index()
    
    # Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)
    
    # Align predictions with truth
    test_truth = _prepare_target(test_df[target]).astype(float)
    forecast_indexed = forecast.set_index("ds")["yhat"]
    
    y_true = test_truth.reindex(forecast_indexed.index).astype(float).values
    y_pred = forecast_indexed.astype(float).values
    
    # Compute metrics
    y_insample = _prepare_target(train_df[target]).values
    seasonal_period = _get_seasonal_period(freq)
    
    return _compute_all_metrics(y_true, y_pred, y_insample, seasonal_period)


# ========================================================================================
# ROLLING BACKTEST
# ========================================================================================

def _rolling_backtest(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    options: ForecastOptions,
    freq: str,
    chosen_regressors: List[str],
    n_folds: int
) -> List[BacktestFold]:
    """Rolling backtest z wieloma foldami."""
    if n_folds < 1:
        return []
    
    n = len(df)
    step = max(1, min(horizon, int(np.ceil(n / (n_folds + 1)))))
    
    folds: List[BacktestFold] = []
    
    for k in range(1, n_folds + 1):
        cut = n - k * step
        
        if cut <= MIN_TIME_POINTS:
            LOGGER.debug(f"Fold {k}: za mało danych ({cut} punktów), przerywam")
            break
        
        part_df = df.iloc[:cut].copy()
        
        try:
            LOGGER.debug(f"Backtest fold {k}/{n_folds}: train={cut}, test={step}")
            
            fold_metrics = _holdout_validation(
                df=part_df,
                target=target,
                horizon=step,
                options=options,
                freq=freq,
                chosen_regressors=chosen_regressors
            )
            
            fold = BacktestFold(
                fold=k,
                train_end=str(part_df.index.max().date()),
                test_len=step,
                metrics=fold_metrics
            )
            
            folds.append(fold)
            
        except Exception as e:
            LOGGER.warning(f"Fold {k} nie powiódł się: {e}")
            continue
    
    return folds


# ========================================================================================
# MAIN API
# ========================================================================================

def fit_prophet(
    df: pd.DataFrame,
    target: str,
    horizon: int = DEFAULT_HORIZON,
    **options_kwargs: Any
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Trenuje model Prophet i zwraca prognozę.
    
    Args:
        df: DataFrame z danymi (musi zawierać target i opcjonalnie kolumnę daty)
        target: Nazwa kolumny do prognozowania
        horizon: Liczba okresów do prognozowania (default: 12)
        **options_kwargs: Opcje konfiguracyjne:
            - seasonality_mode: "additive" | "multiplicative" (default: "additive")
            - changepoint_prior_scale: float (default: 0.1)
            - growth: "linear" | "logistic" | "flat" (default: "linear")
            - cap: float dla logistic growth (optional, auto-computed)
            - floor: float dla logistic growth (optional, auto-computed)
            - extra_regressors: None | "auto" | List[str] (default: None)
            - holidays: DataFrame z ["ds", "holiday"] (optional)
            - country_holidays: str, np. "PL", "US" (optional)
            - include_history: bool (default: False)
            - freq: str, np. "MS", "D" (optional, auto-detect)
            - outlier_clip: None | "iqr" | "zscore" (optional)
            - backtest: int, liczba foldów (0 = disabled, 1 = holdout only)
            
    Returns:
        Tuple zawierający:
        - model: Wytrenowany Prophet model
        - forecast_df: DataFrame z prognozą, kolumny:
            - ds: timestamps
            - yhat: predicted values
            - yhat_lower: lower bound
            - yhat_upper: upper bound
          W forecast_df.attrs["forecast_meta"] znajdują się metadane
          
    Raises:
        ValueError: Jeśli dane są nieprawidłowe
        
    Example:
        >>> model, forecast = fit_prophet(
        ...     df,
        ...     target="sales",
        ...     horizon=30,
        ...     seasonality_mode="multiplicative",
        ...     extra_regressors="auto",
        ...     outlier_clip="iqr",
        ...     backtest=3
        ... )
        >>> print(forecast.tail())
        >>> print(forecast.attrs["forecast_meta"])
    """
    start_time = pd.Timestamp.now()
    
    LOGGER.info("="*80)
    LOGGER.info("Prophet Forecasting v2 - START")
    LOGGER.info("="*80)
    
    # Walidacja
    if target not in df.columns:
        raise ValueError(f"Kolumna celu '{target}' nie istnieje")
    
    if horizon < 1 or horizon > MAX_HORIZON_STEPS:
        raise ValueError(f"Horizon musi być w zakresie [1, {MAX_HORIZON_STEPS}]")
    
    # Parse options
    options = ForecastOptions(
        seasonality_mode=options_kwargs.get("seasonality_mode", DEFAULT_SEASONALITY_MODE),
        changepoint_prior_scale=options_kwargs.get("changepoint_prior_scale", DEFAULT_CHANGEPOINT_PRIOR_SCALE),
        growth=options_kwargs.get("growth", DEFAULT_GROWTH),
        cap=options_kwargs.get("cap"),
        floor=options_kwargs.get("floor"),
        extra_regressors=options_kwargs.get("extra_regressors"),
        holidays=options_kwargs.get("holidays"),
        country_holidays=options_kwargs.get("country_holidays"),
        include_history=options_kwargs.get("include_history", False),
        freq=options_kwargs.get("freq"),
        outlier_clip=options_kwargs.get("outlier_clip"),
        backtest_folds=int(options_kwargs.get("backtest", 0))
    )
    
    LOGGER.info(f"Opcje: {options.to_dict()}")
    
    # 1. Detekcja indeksu czasu
    LOGGER.info("Etap 1/8: Detekcja indeksu czasu")
    df_time = df.copy()
    
    # Try helper if available
    if HAS_HELPER:
        try:
            df_time = ensure_datetime_index(df_time)
        except Exception as e:
            LOGGER.debug(f"ensure_datetime_index nie powiodło się: {e}")
    
    # Fallback
    if not isinstance(df_time.index, pd.DatetimeIndex):
        df_time = _detect_datetime_index(df_time)
    
    if not isinstance(df_time.index, pd.DatetimeIndex):
        raise ValueError(
            "Nie znaleziono kolumny czasu. "
            "Dodaj kolumnę daty lub ustaw DatetimeIndex."
        )
    
    LOGGER.info("✓ DatetimeIndex wykryty")
    
    # 2. Czyszczenie
    LOGGER.info("Etap 2/8: Czyszczenie danych")
    df_clean = _clean_dataframe(df_time)
    
    # Walidacja rozmiaru
    if len(df_clean) < MIN_TIME_POINTS:
        raise ValueError(
            f"Za mało punktów czasowych: {len(df_clean)} < {MIN_TIME_POINTS}"
        )
    
    # Limit
    if len(df_clean) > MAX_TIME_POINTS:
        LOGGER.warning(
            f"Przycinam liczbę punktów {len(df_clean)} → {MAX_TIME_POINTS}"
        )
        df_clean = df_clean.iloc[-MAX_TIME_POINTS:]
    
    LOGGER.info(f"✓ Po czyszczeniu: {len(df_clean)} punktów")
    
    # 3. Przygotowanie targetu
    LOGGER.info("Etap 3/8: Przygotowanie targetu")
    y = _prepare_target(df_clean[target])
    
    # Outlier clipping
    if options.outlier_clip:
        LOGGER.info(f"Clipowanie outlierów metodą: {options.outlier_clip}")
        y = _clip_outliers(y, options.outlier_clip)
    
    # Prophet DataFrame
    prophet_df = pd.DataFrame({
        "ds": df_clean.index,
        "y": y
    }).dropna(subset=["y"])
    
    LOGGER.info(f"✓ Dane Prophet: {len(prophet_df)} obserwacji")
    
    # 4. Częstotliwość
    LOGGER.info("Etap 4/8: Wykrywanie częstotliwości")
    detected_freq = options.freq or _infer_frequency(df_clean.index)
    prophet_freq = _get_prophet_freq_alias(detected_freq)
    seasonal_period = _get_seasonal_period(detected_freq)
    
    LOGGER.info(f"✓ Częstotliwość: {detected_freq} (Prophet: {prophet_freq})")
    LOGGER.info(f"✓ Okres sezonowy: {seasonal_period}")
    
    # 5. Logistic growth
    if options.growth == "logistic":
        LOGGER.info("Etap 5/8: Konfiguracja logistic growth")
        cap, floor = _compute_logistic_bounds(
            prophet_df["y"],
            options.cap,
            options.floor
        )
        prophet_df["cap"] = cap
        prophet_df["floor"] = floor
        
        # Update options for consistency
        options.cap = cap
        options.floor = floor
        
        LOGGER.info(f"✓ Logistic bounds: floor={floor:.2f}, cap={cap:.2f}")
    else:
        LOGGER.info("Etap 5/8: Pomijam (growth != logistic)")
    
    # 6. Regressors
    LOGGER.info("Etap 6/8: Selekcja regresorów")
    chosen_regressors: List[str] = []
    
    if isinstance(options.extra_regressors, str) and options.extra_regressors.lower() == "auto":
        chosen_regressors = _select_auto_regressors(df_clean, target)
    elif isinstance(options.extra_regressors, list):
        chosen_regressors = [
            c for c in options.extra_regressors
            if c in df_clean.columns and c != target
        ]
    
    # Dodaj do DataFrame
    for reg in chosen_regressors:
        prophet_df[reg] = pd.to_numeric(
            df_clean[reg],
            errors="coerce"
        ).reindex(prophet_df["ds"]).values
    
    if chosen_regressors:
        LOGGER.info(f"✓ Regressors: {chosen_regressors}")
    else:
        LOGGER.info("✓ Brak external regressors")
    
    # 7. Budowa i trening modelu
    LOGGER.info("Etap 7/8: Trening modelu Prophet")
    
    model = _build_prophet_model(options, detected_freq, chosen_regressors)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    
    LOGGER.info("✓ Model wytrenowany")
    
    # 8. Generowanie prognozy
    LOGGER.info("Etap 8/8: Generowanie prognozy")
    
    future = model.make_future_dataframe(
        periods=horizon,
        freq=prophet_freq,
        include_history=options.include_history
    )
    
    # Logistic bounds dla przyszłości
    if options.growth == "logistic":
        future["cap"] = options.cap
        future["floor"] = options.floor
    
    # Forward-fill regressors
    if chosen_regressors:
        aux = prophet_df.set_index("ds")[chosen_regressors]
        future = future.set_index("ds")
        for reg in chosen_regressors:
            future[reg] = aux[reg].reindex(future.index).ffill()
        future = future.reset_index()
    
    # Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict(future)
    
    forecast_df = predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    LOGGER.info(f"✓ Prognoza wygenerowana: {len(forecast_df)} punktów")
    
    # 9. Metryki walidacyjne (holdout)
    LOGGER.info("Obliczanie metryk walidacyjnych (holdout)")
    
    try:
        holdout_metrics = _holdout_validation(
            df=df_clean,
            target=target,
            horizon=horizon,
            options=options,
            freq=detected_freq,
            chosen_regressors=chosen_regressors
        )
        
        LOGGER.info(f"✓ Metryki holdout:")
        LOGGER.info(f"  - RMSE: {holdout_metrics.rmse:.4f}")
        LOGGER.info(f"  - MAE: {holdout_metrics.mae:.4f}")
        LOGGER.info(f"  - R²: {holdout_metrics.r2:.4f}")
        LOGGER.info(f"  - sMAPE: {holdout_metrics.smape:.2f}%")
        LOGGER.info(f"  - MASE: {holdout_metrics.mase:.4f}")
        
    except Exception as e:
        LOGGER.warning(f"Nie udało się obliczyć metryk holdout: {e}")
        holdout_metrics = ForecastMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            smape=float("nan"),
            mase=float("nan")
        )
    
    # 10. Rolling backtest (opcjonalnie)
    backtest_results: List[BacktestFold] = []
    
    if options.backtest_folds > 1:
        LOGGER.info(f"Wykonuję rolling backtest ({options.backtest_folds} folds)")
        
        try:
            backtest_results = _rolling_backtest(
                df=df_clean,
                target=target,
                horizon=horizon,
                options=options,
                freq=detected_freq,
                chosen_regressors=chosen_regressors,
                n_folds=options.backtest_folds
            )
            
            if backtest_results:
                LOGGER.info(f"✓ Backtest: {len(backtest_results)} folds zakończonych")
                
                # Average metrics
                avg_rmse = np.mean([f.metrics.rmse for f in backtest_results])
                avg_mae = np.mean([f.metrics.mae for f in backtest_results])
                avg_smape = np.mean([f.metrics.smape for f in backtest_results])
                
                LOGGER.info(f"  - Średnie RMSE: {avg_rmse:.4f}")
                LOGGER.info(f"  - Średnie MAE: {avg_mae:.4f}")
                LOGGER.info(f"  - Średnie sMAPE: {avg_smape:.2f}%")
        
        except Exception as e:
            LOGGER.warning(f"Backtest nie powiódł się: {e}")
    
    # 11. Metadane
    date_range = (
        str(df_clean.index.min().date()),
        str(df_clean.index.max().date())
    )
    
    metadata = {
        "target": target,
        "n_obs": len(prophet_df),
        "freq_inferred": detected_freq,
        "freq": prophet_freq,
        "horizon": horizon,
        "seasonality_mode": options.seasonality_mode,
        "changepoint_prior_scale": options.changepoint_prior_scale,
        "extra_regressors": chosen_regressors if chosen_regressors else None,
        "growth": options.growth,
        "cap": options.cap if options.growth == "logistic" else None,
        "floor": options.floor if options.growth == "logistic" else None,
        "include_history": options.include_history,
        "outlier_clip": options.outlier_clip,
        "country_holidays": options.country_holidays,
        "date_range": date_range,
        "seasonal_period": seasonal_period,
        "metrics": holdout_metrics.to_dict(),
        "backtest": [f.to_dict() for f in backtest_results] if backtest_results else None,
    }
    
    forecast_df.attrs["forecast_meta"] = metadata
    
    # Podsumowanie
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    
    LOGGER.info("="*80)
    LOGGER.info(f"Prophet Forecasting v2 - KONIEC (czas: {elapsed:.2f}s)")
    LOGGER.info("="*80)
    
    return model, forecast_df


# ========================================================================================
# UTILITIES
# ========================================================================================

def extract_future_only(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Wyciąga tylko przyszłe prognozy."""
    metadata = forecast_df.attrs.get("forecast_meta", {})
    horizon = metadata.get("horizon", 0)
    
    if horizon > 0:
        return forecast_df.tail(horizon).copy()
    
    return forecast_df.copy()


def get_forecast_summary(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """Zwraca podsumowanie prognozy."""
    metadata = forecast_df.attrs.get("forecast_meta", {})
    future_df = extract_future_only(forecast_df)
    
    return {
        "target": metadata.get("target", "unknown"),
        "n_historical": metadata.get("n_obs", 0),
        "n_forecast": len(future_df),
        "freq": metadata.get("freq", "unknown"),
        "date_range": metadata.get("date_range", ("unknown", "unknown")),
        "mean_forecast": float(future_df["yhat"].mean()) if not future_df.empty else None,
        "metrics": metadata.get("metrics"),
        "backtest_folds": len(metadata.get("backtest", [])) if metadata.get("backtest") else 0
    }


def calculate_prediction_intervals(
    forecast_df: pd.DataFrame,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Oblicza przedziały ufności dla prognozy.
    
    Args:
        forecast_df: DataFrame z prognozą
        confidence: Poziom ufności (default: 0.95)
        
    Returns:
        DataFrame z dodatkowymi kolumnami width i relative_width
    """
    result = forecast_df.copy()
    
    result["interval_width"] = result["yhat_upper"] - result["yhat_lower"]
    result["relative_width"] = result["interval_width"] / result["yhat"].abs()
    
    return result


# ========================================================================================
# BACKWARD COMPATIBILITY
# ========================================================================================

def forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = DEFAULT_HORIZON,
    **kwargs
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Backward compatible wrapper dla fit_prophet.
    
    Args:
        df: DataFrame
        target: Target column
        horizon: Forecast horizon
        **kwargs: Options
        
    Returns:
        (model, forecast_df)
    """
    return fit_prophet(df, target, horizon, **kwargs)