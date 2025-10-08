"""
Time Series Forecasting Engine - Zaawansowane prognozowanie z Prophet.

Funkcjonalności:
- Automatyczna detekcja kolumny czasu i częstotliwości
- Prophet z pełną konfiguracją (seasonality, growth, holidays)
- Obsługa external regressors (auto-select lub manual)
- Logistic growth z auto cap/floor
- Multiple seasonalities (yearly, weekly, daily, monthly, quarterly)
- Comprehensive metrics (sMAPE, MASE, RMSE, MAE)
- Robust preprocessing i walidacja
- Detailed metadata tracking
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress Prophet's verbose output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")

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
MAX_HORIZON_STEPS = 1200
MIN_VALIDATION_SIZE = 5
DEFAULT_HORIZON = 12

# Parametry Prophet
DEFAULT_CHANGEPOINT_PRIOR_SCALE = 0.1
DEFAULT_SEASONALITY_MODE = "additive"
DEFAULT_GROWTH = "linear"

# External regressors
MAX_AUTO_REGRESSORS = 15
MIN_REGRESSOR_UNIQUE = 3
MIN_REGRESSOR_COMPLETENESS = 0.8

# Frequency aliases mapping
FREQ_ALIASES = {
    "Y": "Y", "A": "Y", "YE": "Y", "AS": "Y",
    "Q": "Q", "QS": "QS", "QE": "Q",
    "M": "MS", "MS": "MS", "ME": "MS",
    "W": "W", "W-SUN": "W",
    "D": "D", "B": "D",
    "H": "H",
    "T": "min", "MIN": "min",
    "S": "S",
}

# Seasonal periods for metrics
SEASONAL_PERIODS = {
    "Y": 1, "Q": 4, "MS": 12, "W": 52,
    "D": 7, "H": 24, "min": 60, "S": 60
}

# Types
GrowthType = Literal["linear", "logistic", "flat"]
SeasonalityMode = Literal["additive", "multiplicative"]

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "forecasting", level: int = logging.INFO) -> logging.Logger:
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

@dataclass
class ForecastConfig:
    """Konfiguracja dla forecasting."""
    horizon: int
    date_col: Optional[str]
    seasonality_mode: SeasonalityMode
    changepoint_prior_scale: float
    extra_regressors: Union[None, str, List[str]]
    holidays: Optional[pd.DataFrame]
    growth: GrowthType
    cap: Optional[float]
    floor: Optional[float]
    freq: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "horizon": self.horizon,
            "date_col": self.date_col,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "extra_regressors": self.extra_regressors,
            "has_holidays": self.holidays is not None,
            "growth": self.growth,
            "cap": self.cap,
            "floor": self.floor,
            "freq": self.freq
        }


@dataclass
class ForecastMetrics:
    """Metryki prognozy."""
    smape: float
    mase: float
    rmse: float
    mae: float
    val_len: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass
class ForecastMetadata:
    """Metadane prognozy."""
    target: str
    n_obs: int
    freq: str
    horizon: int
    seasonality_mode: str
    changepoint_prior_scale: float
    extra_regressors: Optional[List[str]]
    growth: str
    cap: Optional[float]
    floor: Optional[float]
    metrics: Optional[Dict[str, Any]]
    seasonal_period: int
    warnings: List[str]
    date_range: Tuple[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)


# ========================================================================================
# DATETIME DETECTION
# ========================================================================================

def _detect_datetime_column(
    df: pd.DataFrame,
    date_col: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Wykrywa i ustawia kolumnę z datami jako indeks.
    
    Args:
        df: DataFrame źródłowy
        date_col: Preferowana kolumna z datami
        
    Returns:
        Tuple (DataFrame z DatetimeIndex, nazwa kolumny)
        
    Raises:
        ValueError: Jeśli nie znaleziono kolumny z datami
    """
    df_copy = df.copy()
    
    # Jeśli już mamy DatetimeIndex
    if isinstance(df_copy.index, pd.DatetimeIndex):
        idx = df_copy.index
        
        # Remove timezone
        if idx.tz is not None:
            df_copy.index = idx.tz_convert("UTC").tz_localize(None)
        
        df_copy = df_copy.sort_index()
        LOGGER.debug("Używam istniejącego DatetimeIndex")
        return df_copy, df_copy.index.name or "index"
    
    # Próba konwersji wskazanej kolumny
    if date_col and date_col in df_copy.columns:
        result = _try_convert_to_datetime_index(df_copy, date_col)
        if result is not None:
            LOGGER.debug(f"Skonwertowano kolumnę '{date_col}' na DatetimeIndex")
            return result, date_col
    
    # Szukaj po hintach
    for col in df_copy.columns:
        col_lower = str(col).lower()
        
        if any(hint in col_lower for hint in DATE_COLUMN_HINTS):
            result = _try_convert_to_datetime_index(df_copy, col)
            if result is not None:
                LOGGER.debug(f"Automatycznie wykryto kolumnę daty: '{col}'")
                return result, col
    
    # Fallback: pierwsza kolumna
    if len(df_copy.columns) > 0:
        first_col = df_copy.columns[0]
        result = _try_convert_to_datetime_index(df_copy, first_col)
        if result is not None:
            LOGGER.debug(f"Używam pierwszej kolumny jako daty: '{first_col}'")
            return result, first_col
    
    raise ValueError(
        "Nie znaleziono kolumny z datami. "
        "Podaj parametr date_col lub dodaj kolumnę z nazwą zawierającą: "
        f"{', '.join(DATE_COLUMN_HINTS[:5])}"
    )


def _try_convert_to_datetime_index(
    df: pd.DataFrame,
    col: str
) -> Optional[pd.DataFrame]:
    """
    Próbuje skonwertować kolumnę na DatetimeIndex.
    
    Args:
        df: DataFrame
        col: Nazwa kolumny
        
    Returns:
        DataFrame z DatetimeIndex lub None jeśli się nie udało
    """
    try:
        datetime_series = pd.to_datetime(
            df[col],
            errors="coerce",
            infer_datetime_format=True
        )
        
        # Sprawdź czy konwersja się powiodła (min 50% wartości)
        if datetime_series.notna().mean() < 0.5:
            return None
        
        # Remove timezone if present
        if datetime_series.dt.tz is not None:
            datetime_series = datetime_series.dt.tz_convert("UTC").dt.tz_localize(None)
        
        # Ustaw jako indeks
        df_result = df.copy()
        df_result[col] = datetime_series
        df_result = df_result.set_index(col)
        df_result = df_result.sort_index()
        
        return df_result
        
    except Exception as e:
        LOGGER.debug(f"Nie udało się skonwertować kolumny '{col}': {e}")
        return None


# ========================================================================================
# FREQUENCY INFERENCE
# ========================================================================================

def _infer_frequency(idx: pd.DatetimeIndex) -> str:
    """
    Wykrywa częstotliwość szeregu czasowego.
    
    Args:
        idx: DatetimeIndex
        
    Returns:
        Kod częstotliwości (Y, Q, MS, W, D, H, min, S)
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        LOGGER.debug("Za mało punktów do wykrycia częstotliwości, zakładam 'D'")
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
    
    # Fallback: heurystyka po medianie odstępów
    freq_heuristic = _infer_frequency_from_median_diff(idx)
    LOGGER.debug(f"Częstotliwość z heurystyki: {freq_heuristic}")
    return freq_heuristic


def _normalize_frequency(freq: str) -> str:
    """
    Normalizuje alias częstotliwości do standardowej formy.
    
    Args:
        freq: Alias częstotliwości
        
    Returns:
        Znormalizowany alias
    """
    freq_upper = freq.upper()
    
    # Sprawdź prefixy
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
        return "min"
    if freq_upper.startswith("S"):
        return "S"
    
    # Sprawdź mapowanie
    return FREQ_ALIASES.get(freq_upper, "D")


def _infer_frequency_from_median_diff(idx: pd.DatetimeIndex) -> str:
    """
    Wykrywa częstotliwość z mediany różnic między timestampami.
    
    Args:
        idx: DatetimeIndex
        
    Returns:
        Kod częstotliwości
    """
    if len(idx) < 2:
        return "D"
    
    # Różnice w nanosekundach → sekundy
    diffs = np.diff(idx.view("i8"))
    if len(diffs) == 0:
        return "D"
    
    median_seconds = float(np.median(diffs) / 1e9)
    
    # Progi czasowe
    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 7 * DAY
    MONTH = 30 * DAY
    QUARTER = 91 * DAY
    YEAR = 365 * DAY
    
    if median_seconds < MINUTE:
        return "S"
    elif median_seconds < HOUR:
        return "min"
    elif median_seconds < DAY:
        return "H"
    elif median_seconds < WEEK:
        return "D"
    elif median_seconds < MONTH:
        return "W"
    elif median_seconds < QUARTER:
        return "MS"
    elif median_seconds < YEAR:
        return "Q"
    else:
        return "Y"


def _get_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    """
    Określa które seasonalities mają sens dla danej częstotliwości.
    
    Args:
        freq: Kod częstotliwości
        
    Returns:
        Tuple (yearly, weekly, daily)
    """
    freq_upper = freq.upper()
    
    yearly = True  # Prawie zawsze ma sens
    weekly = freq_upper in ("D", "H", "MIN", "S")
    daily = freq_upper in ("H", "MIN", "S")
    
    return yearly, weekly, daily


def _get_prophet_freq_alias(freq: str) -> str:
    """
    Konwertuje częstotliwość do formatu akceptowanego przez Prophet.
    
    Args:
        freq: Kod częstotliwości
        
    Returns:
        Prophet-compatible frequency string
    """
    return FREQ_ALIASES.get(freq.upper(), "D")


def _get_seasonal_period(freq: str) -> int:
    """
    Zwraca okres sezonowy (m) dla metryki MASE.
    
    Args:
        freq: Kod częstotliwości
        
    Returns:
        Okres sezonowy
    """
    return SEASONAL_PERIODS.get(freq.upper(), 1)


# ========================================================================================
# DATA PREPARATION
# ========================================================================================

def _prepare_target(series: pd.Series) -> pd.Series:
    """
    Przygotowuje target do modelowania.
    
    Args:
        series: Serie z wartościami target
        
    Returns:
        Oczyszczona serie numeryczna
    """
    return pd.to_numeric(series, errors="coerce")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści DataFrame z duplikatów i NaT w indeksie.
    
    Args:
        df: DataFrame do oczyszczenia
        
    Returns:
        Oczyszczony DataFrame
    """
    # Usuń NaT w indeksie
    df_clean = df[df.index.notna()].copy()
    
    # Sort by index
    df_clean = df_clean.sort_index()
    
    # Usuń duplikaty w indeksie (zachowaj ostatni)
    if df_clean.index.has_duplicates:
        n_dupes = df_clean.index.duplicated().sum()
        LOGGER.warning(f"Znaleziono {n_dupes} duplikatów w indeksie, usuwam")
        df_clean = df_clean[~df_clean.index.duplicated(keep="last")]
    
    return df_clean


def _validate_time_series(df: pd.DataFrame, target: str) -> None:
    """
    Waliduje szereg czasowy.
    
    Args:
        df: DataFrame z danymi
        target: Nazwa kolumny celu
        
    Raises:
        ValueError: Jeśli walidacja się nie powiedzie
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dane muszą być w formacie DataFrame")
    
    if target not in df.columns:
        raise ValueError(f"Kolumna celu '{target}' nie istnieje")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame musi mieć DatetimeIndex")
    
    if len(df) < MIN_TIME_POINTS:
        raise ValueError(
            f"Za mało punktów czasowych: {len(df)} < {MIN_TIME_POINTS}"
        )
    
    # Sprawdź czy target ma jakieś wartości
    y = _prepare_target(df[target])
    n_valid = y.notna().sum()
    
    if n_valid == 0:
        raise ValueError(f"Kolumna celu '{target}' nie zawiera żadnych wartości")
    
    if n_valid < MIN_TIME_POINTS:
        raise ValueError(
            f"Za mało prawidłowych wartości w targecie: {n_valid} < {MIN_TIME_POINTS}"
        )


# ========================================================================================
# EXTERNAL REGRESSORS
# ========================================================================================

def _select_auto_regressors(
    df: pd.DataFrame,
    target: str,
    max_regressors: int = MAX_AUTO_REGRESSORS
) -> List[str]:
    """
    Automatycznie wybiera zewnętrzne regresory.
    
    Args:
        df: DataFrame z danymi
        target: Nazwa kolumny celu
        max_regressors: Maksymalna liczba regresorów
        
    Returns:
        Lista nazw kolumn regresorów
    """
    candidates: List[Tuple[str, float]] = []
    
    for col in df.columns:
        if col == target:
            continue
        
        series = df[col]
        
        # Musi być numeryczna
        if not pd.api.types.is_numeric_dtype(series):
            continue
        
        # Minimum unikalnych wartości
        n_unique = series.nunique(dropna=True)
        if n_unique < MIN_REGRESSOR_UNIQUE:
            continue
        
        # Minimum kompletności
        completeness = series.notna().mean()
        if completeness < MIN_REGRESSOR_COMPLETENESS:
            continue
        
        # Score: odchylenie standardowe (większe = bardziej zmienne)
        std = float(series.std(skipna=True) or 0.0)
        candidates.append((col, std))
    
    # Sortuj po std i weź top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [col for col, _ in candidates[:max_regressors]]
    
    if selected:
        LOGGER.debug(f"Auto-wybrano {len(selected)} regresorów: {selected}")
    else:
        LOGGER.debug("Brak odpowiednich regresorów zewnętrznych")
    
    return selected


def _validate_regressors(
    df: pd.DataFrame,
    regressors: List[str],
    target: str
) -> List[str]:
    """
    Waliduje listę regresorów.
    
    Args:
        df: DataFrame z danymi
        regressors: Lista nazw regresorów
        target: Nazwa kolumny celu
        
    Returns:
        Lista zwalidowanych regresorów
    """
    valid_regressors: List[str] = []
    
    for col in regressors:
        if col == target:
            LOGGER.warning(f"Pomijam regressor '{col}' (to kolumna celu)")
            continue
        
        if col not in df.columns:
            LOGGER.warning(f"Pomijam regressor '{col}' (nie istnieje)")
            continue
        
        series = df[col]
        
        if not pd.api.types.is_numeric_dtype(series):
            LOGGER.warning(f"Pomijam regressor '{col}' (nie-numeryczny)")
            continue
        
        valid_regressors.append(col)
    
    return valid_regressors


# ========================================================================================
# LOGISTIC GROWTH
# ========================================================================================

def _compute_logistic_bounds(
    y: pd.Series,
    cap: Optional[float],
    floor: Optional[float]
) -> Tuple[float, float]:
    """
    Oblicza cap i floor dla logistic growth.
    
    Args:
        y: Serie z wartościami target
        cap: User-provided cap (optional)
        floor: User-provided floor (optional)
        
    Returns:
        Tuple (cap, floor)
    """
    y_clean = y.dropna()
    
    if len(y_clean) == 0:
        y_max = 1.0
        y_min = 0.0
    else:
        y_max = float(y_clean.max())
        y_min = float(y_clean.min())
    
    # Oblicz cap
    if cap is None or cap <= y_max:
        computed_cap = max(y_max * 1.2, y_max + max(abs(y_max) * 0.1, 1.0))
    else:
        computed_cap = float(cap)
    
    # Oblicz floor
    if floor is None or floor >= y_min or floor >= computed_cap:
        computed_floor = min(y_min * 0.8, y_min - max(abs(y_min) * 0.1, 1.0))
        # Zapewnij że floor < cap
        computed_floor = min(computed_floor, computed_cap - 1e-6)
    else:
        computed_floor = float(floor)
    
    LOGGER.debug(f"Logistic bounds: floor={computed_floor:.2f}, cap={computed_cap:.2f}")
    
    return computed_cap, computed_floor


# ========================================================================================
# PROPHET MODEL
# ========================================================================================

def _build_prophet_model(
    config: ForecastConfig,
    freq: str,
    holidays: Optional[pd.DataFrame]
) -> Prophet:
    """
    Buduje i konfiguruje model Prophet.
    
    Args:
        config: Konfiguracja forecasting
        freq: Częstotliwość danych
        holidays: DataFrame ze świętami
        
    Returns:
        Skonfigurowany model Prophet
    """
    # Seasonalities
    yearly, weekly, daily = _get_seasonality_flags(freq)
    
    # Create model
    model = Prophet(
        seasonality_mode=config.seasonality_mode,
        changepoint_prior_scale=config.changepoint_prior_scale,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        growth=config.growth,
        holidays=holidays if _is_valid_holidays_df(holidays) else None,
        uncertainty_samples=1000,  # Dla bardziej stabilnych przedziałów
    )
    
    # Dodatkowe seasonalities
    if freq in ("MS", "M"):
        model.add_seasonality(
            name="monthly",
            period=30.5,
            fourier_order=8
        )
        LOGGER.debug("Dodano monthly seasonality")
    
    if freq in ("Q", "QS"):
        model.add_seasonality(
            name="quarterly",
            period=91.25,
            fourier_order=4
        )
        LOGGER.debug("Dodano quarterly seasonality")
    
    return model


def _is_valid_holidays_df(holidays: Optional[pd.DataFrame]) -> bool:
    """
    Sprawdza czy DataFrame ze świętami jest prawidłowy.
    
    Args:
        holidays: DataFrame do sprawdzenia
        
    Returns:
        True jeśli prawidłowy
    """
    if holidays is None or not isinstance(holidays, pd.DataFrame):
        return False
    
    if holidays.empty:
        return False
    
    # Musi mieć kolumny 'ds' i 'holiday'
    required_cols = {"ds", "holiday"}
    actual_cols = {c.lower() for c in holidays.columns}
    
    return required_cols.issubset(actual_cols)


# ========================================================================================
# METRICS
# ========================================================================================

def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        
    Returns:
        sMAPE w procentach
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1.0, denominator)
    
    smape = np.mean(numerator / denominator) * 100.0
    return float(smape)


def _compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seasonal_period: int = 1
) -> float:
    """
    Oblicza Mean Absolute Scaled Error.
    
    Args:
        y_true: Prawdziwe wartości (cała historia)
        y_pred: Predykcje
        seasonal_period: Okres sezonowy
        
    Returns:
        MASE
    """
    y = np.asarray(y_true, dtype=float)
    
    # Adjust seasonal period if needed
    m = min(seasonal_period, len(y) - 1)
    if m < 1:
        m = 1
    
    # Naive forecast error (seasonal naive)
    if len(y) > m:
        naive_errors = np.abs(y[m:] - y[:-m])
    else:
        naive_errors = np.abs(np.diff(y))
    
    # Scale (mean absolute error of naive forecast)
    scale = np.mean(naive_errors)
    
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    
    # MAE / scale
    mae = np.mean(np.abs(y_true[-len(y_pred):] - y_pred))
    mase = mae / scale
    
    return float(mase)


def _compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_full: np.ndarray,
    seasonal_period: int
) -> ForecastMetrics:
    """
    Oblicza wszystkie metryki prognozy.
    
    Args:
        y_true: Prawdziwe wartości (validation set)
        y_pred: Predykcje
        y_full: Pełna historia (dla MASE)
        seasonal_period: Okres sezonowy
        
    Returns:
        ForecastMetrics
    """
    smape = _compute_smape(y_true, y_pred)
    mase = _compute_mase(y_full, y_pred, seasonal_period)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    return ForecastMetrics(
        smape=smape,
        mase=mase,
        rmse=rmse,
        mae=mae,
        val_len=len(y_true)
    )


# ========================================================================================
# WARNINGS
# ========================================================================================

def _collect_warnings(
    df: pd.DataFrame,
    regressors: List[str],
    freq: str
) -> List[str]:
    """
    Zbiera ostrzeżenia o danych.
    
    Args:
        df: DataFrame z danymi
        regressors: Lista regresorów
        freq: Częstotliwość
        
    Returns:
        Lista ostrzeżeń
    """
    warnings_list: List[str] = []
    
    # Nieregularna częstotliwość
    try:
        inferred = pd.infer_freq(df.index)
        if inferred is None:
            warnings_list.append("irregular_frequency_detected")
    except Exception:
        pass
    
    # Braki w regresorach
    for col in regressors:
        if col not in df.columns:
            continue
        
        series = pd.to_numeric(df[col], errors="coerce")
        missing_pct = series.isna().mean()
        
        if missing_pct > 0.2:
            warnings_list.append(
                f"regressor_{col}_high_missing({missing_pct:.1%})"
            )
    
    # Za krótki szereg dla danej częstotliwości
    if freq in ("H", "min", "S") and len(df) < 100:
        warnings_list.append(f"short_series_for_freq_{freq}({len(df)}_points)")
    
    return warnings_list


# ========================================================================================
# MAIN API
# ========================================================================================

def forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = DEFAULT_HORIZON,
    *,
    date_col: Optional[str] = None,
    seasonality_mode: SeasonalityMode = DEFAULT_SEASONALITY_MODE,
    changepoint_prior_scale: float = DEFAULT_CHANGEPOINT_PRIOR_SCALE,
    extra_regressors: Union[None, str, List[str]] = None,
    holidays: Optional[pd.DataFrame] = None,
    growth: GrowthType = DEFAULT_GROWTH,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    freq: Optional[str] = None,
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Trenuje model Prophet i generuje prognozę.
    
    Pipeline:
    1. Detekcja i ustawienie kolumny czasu
    2. Walidacja szeregu czasowego
    3. Czyszczenie duplikatów i NaT
    4. Wykrycie częstotliwości
    5. Selekcja regresorów zewnętrznych
    6. Konfiguracja i trening Prophet
    7. Generowanie prognozy
    8. Obliczanie metryk walidacyjnych
    
    Args:
        df: DataFrame z danymi (musi zawierać kolumnę target i opcjonalnie datę)
        target: Nazwa kolumny do prognozowania
        horizon: Liczba okresów do prognozowania (default: 12)
        date_col: Nazwa kolumny z datami (auto-detect jeśli None)
        seasonality_mode: "additive" lub "multiplicative" (default: "additive")
        changepoint_prior_scale: Elastyczność trendu, 0.001-0.5 (default: 0.1)
        extra_regressors: None, "auto" lub lista kolumn (default: None)
        holidays: DataFrame z kolumnami ['ds', 'holiday'] (optional)
        growth: "linear", "logistic" lub "flat" (default: "linear")
        cap: Górne ograniczenie dla logistic growth (optional)
        floor: Dolne ograniczenie dla logistic growth (optional)
        freq: Częstotliwość danych, np. "D", "MS", "H" (auto-detect jeśli None)
        
    Returns:
        Tuple zawierający:
        - model: Wytrenowany Prophet model
        - forecast_df: DataFrame z prognozą, kolumny:
            - ds: timestamps
            - yhat: predicted values
            - yhat_lower: lower bound
            - yhat_upper: upper bound
          Dodatkowo w forecast_df.attrs["forecast_meta"] znajdują się metadane
          
    Raises:
        ValueError: Jeśli dane są nieprawidłowe lub brak wymaganej kolumny
        
    Example:
        >>> model, forecast_df = forecast(
        ...     df,
        ...     target="sales",
        ...     horizon=30,
        ...     date_col="date",
        ...     seasonality_mode="multiplicative",
        ...     extra_regressors="auto"
        ... )
        >>> print(forecast_df.tail())
        >>> print(forecast_df.attrs["forecast_meta"])
        
    Notes:
        - Prophet automatycznie obsługuje brakujące wartości
        - Dla logistic growth, cap/floor są auto-computed jeśli nie podano
        - Regressors są forward-filled dla przyszłych wartości
        - Metrics obliczane na ostatnich 25% danych (min 12, max horizon)
        - Wyniki zawierają comprehensive metadata w attrs
    """
    start_time = pd.Timestamp.now()
    
    LOGGER.info("="*80)
    LOGGER.info("Prophet Forecasting - START")
    LOGGER.info("="*80)
    
    # Walidacja podstawowa
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError(f"Horizon musi być dodatnią liczbą całkowitą, otrzymano: {horizon}")
    
    if horizon > MAX_HORIZON_STEPS:
        LOGGER.warning(
            f"Horizon {horizon} przekracza maksimum {MAX_HORIZON_STEPS}, "
            f"obcinam do {MAX_HORIZON_STEPS}"
        )
        horizon = MAX_HORIZON_STEPS
    
    # Konfiguracja
    config = ForecastConfig(
        horizon=horizon,
        date_col=date_col,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        extra_regressors=extra_regressors,
        holidays=holidays,
        growth=growth,
        cap=cap,
        floor=floor,
        freq=freq
    )
    
    LOGGER.info(f"Konfiguracja: {config.to_dict()}")
    
    # 1. Detekcja kolumny czasu
    LOGGER.info("Etap 1/8: Detekcja kolumny czasu")
    try:
        df_time, detected_date_col = _detect_datetime_column(df, date_col)
        LOGGER.info(f"✓ Kolumna czasu: '{detected_date_col}'")
    except ValueError as e:
        raise ValueError(f"Nie udało się wykryć kolumny czasu: {e}")
    
    # 2. Walidacja
    LOGGER.info("Etap 2/8: Walidacja szeregu czasowego")
    _validate_time_series(df_time, target)
    LOGGER.info(f"✓ Szereg czasowy OK: {len(df_time)} punktów")
    
    # 3. Czyszczenie
    LOGGER.info("Etap 3/8: Czyszczenie danych")
    df_clean = _clean_dataframe(df_time)
    
    # Limit liczby punktów
    if len(df_clean) > MAX_TIME_POINTS:
        LOGGER.warning(
            f"Przycinam liczbę punktów {len(df_clean)} → {MAX_TIME_POINTS}"
        )
        df_clean = df_clean.iloc[-MAX_TIME_POINTS:]
    
    LOGGER.info(f"✓ Po czyszczeniu: {len(df_clean)} punktów")
    
    # 4. Wykrycie częstotliwości
    LOGGER.info("Etap 4/8: Wykrywanie częstotliwości")
    detected_freq = freq or _infer_frequency(df_clean.index)
    prophet_freq = _get_prophet_freq_alias(detected_freq)
    seasonal_period = _get_seasonal_period(detected_freq)
    
    LOGGER.info(f"✓ Częstotliwość: {detected_freq} (Prophet: {prophet_freq})")
    LOGGER.info(f"✓ Okres sezonowy: {seasonal_period}")
    
    # 5. Przygotowanie danych dla Prophet
    LOGGER.info("Etap 5/8: Przygotowanie danych")
    
    # Target
    y = _prepare_target(df_clean[target])
    
    # Podstawowy DataFrame
    prophet_df = pd.DataFrame({
        "ds": df_clean.index,
        "y": y
    }).dropna(subset=["y"])
    
    LOGGER.info(f"✓ Dane Prophet: {len(prophet_df)} obserwacji")
    
    # Logistic growth bounds
    if growth == "logistic":
        computed_cap, computed_floor = _compute_logistic_bounds(
            prophet_df["y"],
            cap,
            floor
        )
        prophet_df["cap"] = computed_cap
        prophet_df["floor"] = computed_floor
        LOGGER.info(f"✓ Logistic growth: floor={computed_floor:.2f}, cap={computed_cap:.2f}")
    
    # 6. Regressors
    LOGGER.info("Etap 6/8: Selekcja regresorów")
    
    selected_regressors: List[str] = []
    
    if isinstance(extra_regressors, str) and extra_regressors.lower() == "auto":
        selected_regressors = _select_auto_regressors(df_clean, target)
    elif isinstance(extra_regressors, list):
        selected_regressors = _validate_regressors(df_clean, extra_regressors, target)
    
    # Dodaj regressors do DataFrame
    for reg_col in selected_regressors:
        prophet_df[reg_col] = pd.to_numeric(df_clean[reg_col], errors="coerce")
    
    if selected_regressors:
        LOGGER.info(f"✓ Regressors: {selected_regressors}")
    else:
        LOGGER.info("✓ Brak external regressors")
    
    # 7. Budowa i trening modelu
    LOGGER.info("Etap 7/8: Trening modelu Prophet")
    
    model = _build_prophet_model(config, detected_freq, holidays)
    
    # Rejestracja regresorów
    for reg_col in selected_regressors:
        model.add_regressor(reg_col, standardize="auto")
        LOGGER.debug(f"Dodano regressor: {reg_col}")
    
    # Fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    
    LOGGER.info("✓ Model wytrenowany")
    
    # 8. Generowanie prognozy
    LOGGER.info("Etap 8/8: Generowanie prognozy")
    
    # Future dataframe
    future_df = model.make_future_dataframe(
        periods=horizon,
        freq=prophet_freq,
        include_history=True
    )
    
    # Logistic growth bounds dla przyszłości
    if growth == "logistic":
        future_df["cap"] = computed_cap
        future_df["floor"] = computed_floor
    
    # Forward-fill regressors
    if selected_regressors:
        # Przygotuj auxiliary DataFrame z regressors
        aux_df = prophet_df.set_index("ds")[selected_regressors]
        
        # Merge z future i ffill
        future_df = future_df.set_index("ds")
        for reg_col in selected_regressors:
            future_df[reg_col] = aux_df[reg_col].reindex(future_df.index).ffill()
        future_df = future_df.reset_index()
        
        LOGGER.debug("Forward-filled regressors dla przyszłości")
    
    # Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = model.predict(future_df)
    
    # Wyciągnij kolumny do forecast
    forecast_df = predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    LOGGER.info(f"✓ Prognoza wygenerowana: {len(forecast_df)} punktów")
    
    # 9. Metryki walidacyjne
    LOGGER.info("Obliczanie metryk walidacyjnych")
    
    # Merge z historycznymi wartościami
    historical_df = forecast_df.merge(
        prophet_df[["ds", "y"]],
        on="ds",
        how="left"
    )
    
    # Wiersze z prawdziwymi wartościami
    known_mask = historical_df["y"].notna()
    known_df = historical_df[known_mask].copy()
    
    # Validation set: ostatnie X obserwacji
    val_length = min(
        max(12, horizon),
        max(1, len(known_df) // 4)
    )
    
    validation_df = known_df.tail(val_length)
    
    metrics: Optional[ForecastMetrics] = None
    
    if len(validation_df) >= MIN_VALIDATION_SIZE:
        y_true = validation_df["y"].to_numpy(dtype=float)
        y_pred = validation_df["yhat"].to_numpy(dtype=float)
        y_full = known_df["y"].to_numpy(dtype=float)
        
        metrics = _compute_forecast_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_full=y_full,
            seasonal_period=seasonal_period
        )
        
        LOGGER.info(f"✓ Metryki (validation set = {val_length} obs):")
        LOGGER.info(f"  - sMAPE: {metrics.smape:.2f}%")
        LOGGER.info(f"  - MASE: {metrics.mase:.3f}")
        LOGGER.info(f"  - RMSE: {metrics.rmse:.2f}")
        LOGGER.info(f"  - MAE: {metrics.mae:.2f}")
    else:
        LOGGER.warning("Za mało danych do obliczenia metryk walidacyjnych")
    
    # 10. Ostrzeżenia
    warnings_list = _collect_warnings(df_clean, selected_regressors, detected_freq)
    
    if warnings_list:
        LOGGER.warning(f"Ostrzeżenia: {warnings_list}")
    
    # 11. Metadane
    date_range = (
        str(df_clean.index.min().date()),
        str(df_clean.index.max().date())
    )
    
    metadata = ForecastMetadata(
        target=target,
        n_obs=len(prophet_df),
        freq=prophet_freq,
        horizon=horizon,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        extra_regressors=selected_regressors if selected_regressors else None,
        growth=growth,
        cap=computed_cap if growth == "logistic" else None,
        floor=computed_floor if growth == "logistic" else None,
        metrics=metrics.to_dict() if metrics else None,
        seasonal_period=seasonal_period,
        warnings=warnings_list,
        date_range=date_range
    )
    
    forecast_df.attrs["forecast_meta"] = metadata.to_dict()
    
    # Podsumowanie
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    
    LOGGER.info("="*80)
    LOGGER.info(f"Prophet Forecasting - KONIEC (czas: {elapsed:.2f}s)")
    LOGGER.info("="*80)
    
    return model, forecast_df


# ========================================================================================
# UTILITIES
# ========================================================================================

def extract_future_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyciąga tylko przyszłe prognozy (bez historii).
    
    Args:
        forecast_df: DataFrame zwrócony z forecast()
        
    Returns:
        DataFrame tylko z przyszłymi prognozami
    """
    metadata = forecast_df.attrs.get("forecast_meta", {})
    horizon = metadata.get("horizon", 0)
    
    if horizon > 0:
        return forecast_df.tail(horizon).copy()
    else:
        return forecast_df.copy()


def get_forecast_summary(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca podsumowanie prognozy.
    
    Args:
        forecast_df: DataFrame zwrócony z forecast()
        
    Returns:
        Słownik z podsumowaniem
    """
    metadata = forecast_df.attrs.get("forecast_meta", {})
    
    future_df = extract_future_forecast(forecast_df)
    
    summary = {
        "target": metadata.get("target", "unknown"),
        "n_historical": metadata.get("n_obs", 0),
        "n_forecast": len(future_df),
        "freq": metadata.get("freq", "unknown"),
        "date_range": metadata.get("date_range", ("unknown", "unknown")),
        "forecast_range": (
            str(future_df["ds"].min().date()) if not future_df.empty else "N/A",
            str(future_df["ds"].max().date()) if not future_df.empty else "N/A"
        ),
        "mean_forecast": float(future_df["yhat"].mean()) if not future_df.empty else None,
        "metrics": metadata.get("metrics"),
        "warnings": metadata.get("warnings", [])
    }
    
    return summary


def detect_anomalies_in_forecast(
    forecast_df: pd.DataFrame,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Wykrywa anomalie w prognozie (punkty poza threshold * sigma).
    
    Args:
        forecast_df: DataFrame zwrócony z forecast()
        threshold: Liczba odchyleń standardowych (default: 2.0)
        
    Returns:
        DataFrame z kolumną 'is_anomaly'
    """
    result = forecast_df.copy()
    
    # Szerokość przedziału ufności
    result["uncertainty"] = result["yhat_upper"] - result["yhat_lower"]
    
    # Średnia i std uncertainty
    mean_uncertainty = result["uncertainty"].mean()
    std_uncertainty = result["uncertainty"].std()
    
    # Anomalie: punkty z bardzo dużą uncertainty
    threshold_uncertainty = mean_uncertainty + threshold * std_uncertainty
    result["is_anomaly"] = (result["uncertainty"] > threshold_uncertainty).astype(int)
    
    return result


def compare_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    metric: str = "smape"
) -> pd.DataFrame:
    """
    Porównuje różne prognozy.
    
    Args:
        forecasts: Dict {name: forecast_df}
        metric: Metryka do porównania ("smape", "mase", "rmse", "mae")
        
    Returns:
        DataFrame z porównaniem
    """
    comparison_data = []
    
    for name, forecast_df in forecasts.items():
        metadata = forecast_df.attrs.get("forecast_meta", {})
        metrics = metadata.get("metrics")
        
        if metrics and metric in metrics:
            comparison_data.append({
                "model": name,
                metric: metrics[metric],
                "horizon": metadata.get("horizon", 0),
                "n_obs": metadata.get("n_obs", 0)
            })
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(metric)
    
    return comparison_df


def save_forecast(
    forecast_df: pd.DataFrame,
    filepath: str,
    include_metadata: bool = True
) -> None:
    """
    Zapisuje prognozę do pliku.
    
    Args:
        forecast_df: DataFrame z prognozą
        filepath: Ścieżka do pliku (CSV lub pickle)
        include_metadata: Czy zapisać metadane
    """
    import pathlib
    
    path = pathlib.Path(filepath)
    
    if path.suffix == ".csv":
        forecast_df.to_csv(path, index=False)
        
        if include_metadata:
            metadata = forecast_df.attrs.get("forecast_meta", {})
            metadata_path = path.with_suffix(".json")
            
            import json
            metadata_path.write_text(
                json.dumps(metadata, indent=2, default=str),
                encoding="utf-8"
            )
            LOGGER.info(f"Zapisano metadane: {metadata_path}")
    
    elif path.suffix in (".pkl", ".pickle"):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(forecast_df, f)
    
    else:
        raise ValueError(f"Nieobsługiwany format: {path.suffix}")
    
    LOGGER.info(f"Zapisano prognozę: {path}")


def load_forecast(filepath: str) -> pd.DataFrame:
    """
    Wczytuje prognozę z pliku.
    
    Args:
        filepath: Ścieżka do pliku
        
    Returns:
        DataFrame z prognozą
    """
    import pathlib
    
    path = pathlib.Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Plik nie istnieje: {path}")
    
    if path.suffix == ".csv":
        forecast_df = pd.read_csv(path, parse_dates=["ds"])
        
        # Spróbuj wczytać metadane
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            forecast_df.attrs["forecast_meta"] = metadata
            LOGGER.info(f"Wczytano metadane: {metadata_path}")
    
    elif path.suffix in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as f:
            forecast_df = pickle.load(f)
    
    else:
        raise ValueError(f"Nieobsługiwany format: {path.suffix}")
    
    LOGGER.info(f"Wczytano prognozę: {path}")
    return forecast_df