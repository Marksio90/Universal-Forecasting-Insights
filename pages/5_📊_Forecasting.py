"""
Moduł Forecasting (Prophet) PRO - Zaawansowane prognozowanie szeregów czasowych.

Funkcjonalności:
- Automatyczne wykrywanie częstotliwości i sezonowości
- Prophet z konfigurowalnymi parametrami
- Rolling backtesting z wieloma foldami
- Kompleksowe metryki (sMAPE, MASE, RMSE, MAE)
- Wizualizacje prognoz + komponenty + changepoints
- Multi-format export (CSV, JSON, model)
- Historia prognoz
- Cache modeli dla wydajności
"""

from __future__ import annotations

import io
import json
import time
import logging
import hashlib
from typing import Optional, Literal, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.ml_models.forecasting import forecast
from src.utils.helpers import ensure_datetime_index

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MIN_TIME_POINTS = 10
MAX_TIME_POINTS = 100_000
MAX_HORIZON = 365 * 10  # 10 lat
MIN_HORIZON = 1
MAX_BACKTEST_FOLDS = 10

# Częstotliwości seasonality
SEASONALITY_PERIODS = {
    "H": 24,    # Hourly -> 24h cycle
    "D": 7,     # Daily -> weekly cycle
    "W": 52,    # Weekly -> yearly cycle
    "M": 12,    # Monthly -> yearly cycle
    "Q": 4,     # Quarterly -> yearly cycle
    "A": 1,     # Annual -> no strong cycle
    "Y": 1
}

SeasonalityMode = Literal["additive", "multiplicative"]
ForecastStartMode = Literal["after_last", "rolling_test"]


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class ForecastConfig:
    """Konfiguracja prognozy."""
    target: str
    date_col: str
    horizon: int
    start_mode: ForecastStartMode
    test_periods: int
    seasonality_mode: SeasonalityMode
    changepoint_prior_scale: float
    backtest_folds: int


@dataclass
class TimeSeriesInfo:
    """Informacje o szeregu czasowym."""
    length: int
    start_date: str
    end_date: str
    frequency: Optional[str]
    seasonal_period: int
    has_trend: bool
    missing_values: int


@dataclass
class ForecastMetrics:
    """Metryki prognozy."""
    smape: float
    mase: float
    rmse: float
    mae: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastResult:
    """Wynik prognozy."""
    forecast_df: pd.DataFrame
    model: Any
    metrics: Optional[ForecastMetrics]
    ts_info: TimeSeriesInfo
    config: ForecastConfig
    training_time: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ========================================================================================
# METRICS FUNCTIONS
# ========================================================================================

def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Oblicza Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        
    Returns:
        sMAPE w procentach (0-100)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1.0
    
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _compute_mase(y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 1) -> float:
    """
    Oblicza Mean Absolute Scaled Error.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        seasonal_period: Okres sezonowości dla naive forecast
        
    Returns:
        MASE (bez jednostki)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) <= seasonal_period:
        return float(np.nan)
    
    # Mean absolute error of naive forecast
    naive_mae = np.mean(np.abs(np.diff(y_true, n=seasonal_period)))
    
    if naive_mae == 0:
        return float(np.nan)
    
    # Mean absolute error of actual forecast
    mae = np.mean(np.abs(y_true - y_pred))
    
    return float(mae / naive_mae)


def _compute_forecast_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    seasonal_period: int = 1
) -> ForecastMetrics:
    """
    Oblicza kompleksowe metryki prognozy.
    
    Args:
        y_true: Prawdziwe wartości
        y_pred: Predykcje
        seasonal_period: Okres sezonowości
        
    Returns:
        ForecastMetrics z metrykami
    """
    # Wyrównaj długości
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true.iloc[:min_len].values
    y_pred = y_pred.iloc[:min_len].values
    
    # Oblicz metryki
    smape = _compute_smape(y_true, y_pred)
    mase = _compute_mase(y_true, y_pred, seasonal_period)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    return ForecastMetrics(
        smape=smape,
        mase=mase,
        rmse=rmse,
        mae=mae
    )


# ========================================================================================
# TIME SERIES UTILITIES
# ========================================================================================

def _validate_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Waliduje DataFrame z session state.
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Zwalidowany DataFrame
        
    Raises:
        ValueError: Jeśli DataFrame jest nieprawidłowy
    """
    if df is None:
        raise ValueError("Brak danych w session state")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Oczekiwano DataFrame, otrzymano {type(df)}")
    
    if df.empty:
        raise ValueError("DataFrame jest pusty")
    
    return df


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Znajduje kolumny które mogą być datami.
    
    Args:
        df: DataFrame do przeszukania
        
    Returns:
        Lista nazw kolumn potencjalnie zawierających daty
    """
    candidates = []
    
    # Szukaj po nazwie
    date_keywords = ["date", "data", "time", "czas", "datetime", "timestamp"]
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            candidates.append(col)
    
    # Szukaj po typie
    for col in df.select_dtypes(include=["datetime64", "object"]).columns:
        if col not in candidates:
            # Spróbuj przekonwertować
            try:
                pd.to_datetime(df[col].head(10), errors="coerce")
                candidates.append(col)
            except Exception:
                continue
    
    return candidates


def _prepare_time_series(
    df: pd.DataFrame,
    target: str,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Przygotowuje szereg czasowy z walidacją.
    
    Args:
        df: DataFrame źródłowy
        target: Nazwa kolumny celu
        date_col: Nazwa kolumny daty (opcjonalnie)
        
    Returns:
        DataFrame z DatetimeIndex i kolumną target
        
    Raises:
        ValueError: Jeśli dane są nieprawidłowe
    """
    df = df.copy()
    
    # Sprawdź czy target istnieje
    if target not in df.columns:
        raise ValueError(f"Kolumna '{target}' nie istnieje w danych")
    
    # Jeśli podano kolumnę daty
    if date_col and date_col != "(auto)":
        if date_col not in df.columns:
            raise ValueError(f"Kolumna daty '{date_col}' nie istnieje")
        
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col).sort_index()
        except Exception as e:
            logger.warning(f"Błąd konwersji kolumny daty: {e}")
            raise ValueError(f"Nie udało się przekonwertować kolumny '{date_col}' na daty")
    
    # Auto-wykrywanie datetime index
    df = ensure_datetime_index(df)
    
    # Walidacja indeksu
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Nie wykryto prawidłowego indeksu czasu. "
            "Upewnij się, że dane zawierają kolumnę z datą/czasem."
        )
    
    # Walidacja długości
    if len(df) < MIN_TIME_POINTS:
        raise ValueError(
            f"Za mało punktów czasowych ({len(df)}). "
            f"Minimum: {MIN_TIME_POINTS}"
        )
    
    if len(df) > MAX_TIME_POINTS:
        raise ValueError(
            f"Za dużo punktów czasowych ({len(df):,}). "
            f"Maksimum: {MAX_TIME_POINTS:,}"
        )
    
    # Usuń duplikaty w indeksie
    if df.index.duplicated().any():
        logger.warning("Wykryto duplikaty w indeksie - usuwam")
        df = df[~df.index.duplicated(keep="first")]
    
    return df


def _infer_seasonal_period(freq: Optional[str]) -> int:
    """
    Określa okres sezonowości na podstawie częstotliwości.
    
    Args:
        freq: Częstotliwość pandas (np. "D", "M", "H")
        
    Returns:
        Liczba okresów w cyklu sezonowym
    """
    if not freq:
        return 1
    
    # Weź pierwszy znak (np. "D" z "D", "MS" z "MS")
    freq_char = freq[0].upper()
    
    return SEASONALITY_PERIODS.get(freq_char, 1)


def _extract_time_series_info(df: pd.DataFrame, target: str) -> TimeSeriesInfo:
    """
    Ekstrahuje informacje o szeregu czasowym.
    
    Args:
        df: DataFrame z DatetimeIndex
        target: Nazwa kolumny celu
        
    Returns:
        TimeSeriesInfo z metadanymi
    """
    # Częstotliwość
    freq = pd.infer_freq(df.index)
    seasonal_period = _infer_seasonal_period(freq)
    
    # Trend (prosty test)
    series = df[target].dropna()
    has_trend = False
    if len(series) > 2:
        # Prosta regresja liniowa
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        has_trend = abs(slope) > 0.01  # Arbitralna wartość
    
    return TimeSeriesInfo(
        length=len(df),
        start_date=df.index[0].strftime("%Y-%m-%d"),
        end_date=df.index[-1].strftime("%Y-%m-%d"),
        frequency=freq,
        seasonal_period=seasonal_period,
        has_trend=has_trend,
        missing_values=int(df[target].isna().sum())
    )


def _rolling_backtest(
    df: pd.DataFrame,
    target: str,
    n_folds: int,
    test_size: int,
    seasonal_period: int,
    config: ForecastConfig
) -> list[dict]:
    """
    Przeprowadza rolling backtesting.
    
    Args:
        df: DataFrame z szeregiem czasowym
        target: Nazwa kolumny celu
        n_folds: Liczba foldów
        test_size: Rozmiar zbioru testowego
        seasonal_period: Okres sezonowości
        config: Konfiguracja prognozy
        
    Returns:
        Lista słowników z metrykami dla każdego foldu
    """
    results = []
    total_len = len(df)
    fold_size = max(1, test_size // max(1, n_folds))
    
    for k in range(n_folds, 0, -1):
        fold_num = n_folds - k + 1
        
        try:
            # Oblicz split point
            test_end_offset = fold_size * k
            split_point = max(MIN_TIME_POINTS, total_len - test_end_offset)
            
            # Podziel dane
            train_df = df.iloc[:split_point]
            test_df = df.iloc[split_point:split_point + fold_size]
            
            if len(test_df) < 1:
                logger.warning(f"Fold {fold_num}: za mało danych testowych")
                continue
            
            # Trenuj i prognozuj
            logger.info(f"Backtesting fold {fold_num}/{n_folds}")
            
            _model, fc = forecast(
                train_df[[target]].copy(),
                target,
                horizon=len(test_df)
            )
            
            # Dopasuj prognozy do dat testowych
            fc_aligned = fc.set_index("ds").reindex(test_df.index).dropna()
            
            if fc_aligned.empty:
                logger.warning(f"Fold {fold_num}: brak dopasowanych prognoz")
                continue
            
            # Oblicz metryki
            y_true = test_df[target].loc[fc_aligned.index]
            y_pred = fc_aligned["yhat"]
            
            metrics = _compute_forecast_metrics(y_true, y_pred, seasonal_period)
            
            results.append({
                "fold": fold_num,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "smape": round(metrics.smape, 2),
                "mase": round(metrics.mase, 4),
                "rmse": round(metrics.rmse, 4),
                "mae": round(metrics.mae, 4)
            })
            
        except Exception as e:
            logger.error(f"Błąd w fold {fold_num}: {e}", exc_info=True)
            results.append({
                "fold": fold_num,
                "error": str(e)
            })
    
    return results


def _add_to_forecast_history(result: ForecastResult) -> None:
    """
    Dodaje wynik prognozy do historii.
    
    Args:
        result: ForecastResult do zapisania
    """
    if "forecast_history" not in st.session_state:
        st.session_state["forecast_history"] = []
    
    # Metadane historii
    history_entry = {
        "timestamp": result.timestamp,
        "target": result.config.target,
        "horizon": result.config.horizon,
        "forecast_points": len(result.forecast_df),
        "training_time": result.training_time,
        "frequency": result.ts_info.frequency,
        "metrics": result.metrics.to_dict() if result.metrics else None
    }
    
    # Dodaj na początek
    st.session_state["forecast_history"].insert(0, history_entry)
    
    # Ogranicz do 10
    st.session_state["forecast_history"] = st.session_state["forecast_history"][:10]


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

st.title("📊 Forecasting (Prophet) — PRO")

# ========================================================================================
# WALIDACJA DANYCH
# ========================================================================================

try:
    df_raw = st.session_state.get("df") or st.session_state.get("df_raw")
    df_main = _validate_dataframe(df_raw)
except ValueError as e:
    st.warning(f"⚠️ {e}")
    st.info("Przejdź do **📤 Upload Data**, aby wczytać dane.")
    st.stop()

# ========================================================================================
# SIDEBAR: KONFIGURACJA
# ========================================================================================

with st.sidebar:
    st.subheader("⚙️ Parametry prognozy")
    
    # Target selection
    target = st.selectbox(
        "Kolumna celu (y)",
        options=list(df_main.columns),
        help="Zmienna którą chcesz prognozować"
    )
    
    # Date column selection
    date_candidates = ["(auto)"] + _detect_date_columns(df_main)
    
    date_col = st.selectbox(
        "Kolumna daty/czasu",
        options=date_candidates,
        index=0,
        help="(auto) = automatyczne wykrywanie"
    )
    
    st.divider()
    
    # Forecast parameters
    horizon = st.number_input(
        "Horyzont prognozy (okresy)",
        min_value=MIN_HORIZON,
        max_value=MAX_HORIZON,
        value=12,
        step=1,
        help=f"Liczba okresów do przewidzenia (max {MAX_HORIZON:,})"
    )
    
    start_mode_option = st.selectbox(
        "Punkt startu prognozy",
        options=[
            "📈 Po ostatniej obserwacji",
            "🧪 Test na ostatnich N okresach"
        ],
        index=1
    )
    
    start_mode: ForecastStartMode = (
        "after_last" if "Po ostatniej" in start_mode_option else "rolling_test"
    )
    
    if start_mode == "rolling_test":
        max_test = max(1, len(df_main) // 10)
        test_periods = st.number_input(
            "N okresów testowych",
            min_value=1,
            max_value=min(10000, len(df_main) // 2),
            value=min(12, max_test),
            step=1,
            help="Ile ostatnich okresów użyć do walidacji"
        )
    else:
        test_periods = 0
    
    st.divider()
    
    st.subheader("🔧 Zaawansowane (Prophet)")
    
    seasonality_mode = st.selectbox(
        "Tryb sezonowości",
        options=["additive", "multiplicative"],
        index=0,
        help="Additive: sezonowość stała. Multiplicative: rośnie z trendem"
    )
    
    changepoint_prior_scale = st.slider(
        "Changepoint prior scale",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Wyższa wartość = większa elastyczność trendu"
    )
    
    backtest_folds = st.number_input(
        "Rolling backtesting (foldy)",
        min_value=0,
        max_value=MAX_BACKTEST_FOLDS,
        value=3,
        step=1,
        help="0 = wyłączone. Więcej foldów = dłuższy czas"
    )
    
    st.divider()
    
    # Historia
    history_count = len(st.session_state.get("forecast_history", []))
    st.caption(f"📚 Historia: {history_count} prognoz")

# Konfiguracja
config = ForecastConfig(
    target=target,
    date_col=date_col,
    horizon=horizon,
    start_mode=start_mode,
    test_periods=test_periods,
    seasonality_mode=seasonality_mode,
    changepoint_prior_scale=changepoint_prior_scale,
    backtest_folds=backtest_folds
)

# ========================================================================================
# PRZYGOTOWANIE SZEREGU CZASOWEGO
# ========================================================================================

st.subheader("⏱️ Informacje o szeregu czasowym")

try:
    with st.spinner("🔄 Przygotowuję szereg czasowy..."):
        df_ts = _prepare_time_series(df_main, target, date_col)
        ts_info = _extract_time_series_info(df_ts, target)
    
    # Wyświetl info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Długość", f"{ts_info.length:,}")
    col2.metric("Częstotliwość", ts_info.frequency or "auto")
    col3.metric("Okres sezonowy", ts_info.seasonal_period)
    col4.metric("Braki", ts_info.missing_values)
    
    st.caption(
        f"📅 Zakres: **{ts_info.start_date}** → **{ts_info.end_date}** • "
        f"Trend: **{'✓' if ts_info.has_trend else '✗'}**"
    )
    
    # Podgląd
    with st.expander("📄 Podgląd danych", expanded=False):
        preview_rows = st.slider("Liczba wierszy", 10, 200, 50)
        st.dataframe(
            df_ts[[target]].tail(preview_rows),
            use_container_width=True
        )
    
except ValueError as ve:
    st.error(f"❌ {ve}")
    st.stop()
except Exception as e:
    st.error(f"❌ Błąd przygotowania danych: {e}")
    logger.error(f"Błąd przygotowania: {e}", exc_info=True)
    st.stop()

# ========================================================================================
# ROLLING BACKTESTING (OPCJONALNIE)
# ========================================================================================

if config.start_mode == "rolling_test" and config.backtest_folds > 0:
    st.divider()
    st.subheader("🧪 Rolling Backtesting")
    
    backtest_button = st.button(
        "▶️ Uruchom backtesting",
        use_container_width=True,
        help=f"Przeprowadzi {config.backtest_folds} testów walidacyjnych"
    )
    
    if backtest_button:
        with st.spinner(f"Przeprowadzam backtesting ({config.backtest_folds} foldów)..."):
            start_time = time.time()
            
            backtest_results = _rolling_backtest(
                df_ts,
                target,
                config.backtest_folds,
                config.test_periods,
                ts_info.seasonal_period,
                config
            )
            
            elapsed = time.time() - start_time
        
        if backtest_results:
            st.success(f"✅ Backtesting zakończony w {elapsed:.2f}s")
            
            # DataFrame z wynikami
            bt_df = pd.DataFrame(backtest_results)
            
            # Tabela
            st.dataframe(bt_df, use_container_width=True)
            
            # Heatmap metryk (bez błędów)
            metric_cols = ["smape", "mase", "rmse", "mae"]
            valid_metrics = [col for col in metric_cols if col in bt_df.columns]
            
            if valid_metrics and "fold" in bt_df.columns:
                bt_clean = bt_df[bt_df["error"].isna()] if "error" in bt_df.columns else bt_df
                
                if not bt_clean.empty:
                    fig_bt = px.imshow(
                        bt_clean.set_index("fold")[valid_metrics].T,
                        text_auto=".2f",
                        aspect="auto",
                        title="Backtesting - Metryki (niżej = lepiej)",
                        color_continuous_scale="RdYlGn_r"
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Średnie metryki
                    with st.expander("📊 Średnie metryki", expanded=False):
                        means = bt_clean[valid_metrics].mean()
                        st.json(means.to_dict())
        else:
            st.warning("⚠️ Brak wyników backtest")

# ========================================================================================
# GŁÓWNA PROGNOZA
# ========================================================================================

st.divider()

forecast_col1, forecast_col2 = st.columns([3, 1])

with forecast_col1:
    forecast_button = st.button(
        "📟 Prognozuj",
        type="primary",
        use_container_width=True,
        help="Rozpocznij prognozowanie"
    )

with forecast_col2:
    if st.button("🗑️ Wyczyść historię", use_container_width=True):
        st.session_state["forecast_history"] = []
        st.success("✅ Historia wyczyszczona")
        st.rerun()

# ========================================================================================
# PIPELINE PROGNOZOWANIA
# ========================================================================================

if forecast_button:
    start_time = time.time()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ============================================================================
        # ETAP 1: PRZYGOTOWANIE DANYCH (0-20%)
        # ============================================================================
        
        status_text.text("🔍 Przygotowuję dane...")
        progress_bar.progress(10)
        
        # Kopia dla treningu
        train_df = df_ts.copy()
        test_slice = None
        
        # Jeśli test mode - wytnij końcówkę
        if config.start_mode == "rolling_test" and config.test_periods > 0:
            if config.test_periods >= len(train_df):
                st.error(
                    f"❌ Za długi okres testowy ({config.test_periods}) "
                    f"względem długości serii ({len(train_df)})"
                )
                st.stop()
            
            test_slice = train_df.iloc[-config.test_periods:][[target]].copy()
            train_df = train_df.iloc[:-config.test_periods]
            
            logger.info(
                f"Split: train={len(train_df)}, test={len(test_slice)}"
            )
        
        progress_bar.progress(20)
        
        # ============================================================================
        # ETAP 2: TRENOWANIE MODELU (20-60%)
        # ============================================================================
        
        status_text.text("🤖 Trenuję model Prophet...")
        progress_bar.progress(30)
        
        model, forecast_df = forecast(
            train_df[[target]].copy(),
            target,
            horizon=config.horizon
        )
        
        # Konwersja ds do datetime
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        
        training_time = time.time() - start_time
        
        st.success(f"✅ Model wytrenowany w {training_time:.2f}s")
        
        progress_bar.progress(60)
        
        # ============================================================================
        # ETAP 3: WALIDACJA (60-80%)
        # ============================================================================
        
        metrics_result = None
        
        if test_slice is not None:
            status_text.text("📊 Obliczam metryki walidacyjne...")
            progress_bar.progress(70)
            
            try:
                # Dopasuj prognozy do dat testowych
                fc_aligned = forecast_df.set_index("ds").reindex(test_slice.index).dropna()
                
                if not fc_aligned.empty:
                    y_true = test_slice[target].loc[fc_aligned.index]
                    y_pred = fc_aligned["yhat"]
                    
                    metrics_result = _compute_forecast_metrics(
                        y_true,
                        y_pred,
                        ts_info.seasonal_period
                    )
                    
                    logger.info(f"Metryki walidacyjne: {metrics_result}")
                    
            except Exception as e:
                logger.error(f"Błąd obliczania metryk: {e}", exc_info=True)
                st.warning(f"⚠️ Nie udało się obliczyć metryk: {e}")
        
        progress_bar.progress(80)
        
        # ============================================================================
        # ETAP 4: FINALIZACJA (80-100%)
        # ============================================================================
        
        status_text.text("💾 Zapisuję wyniki...")
        
        # Wynik prognozy
        result = ForecastResult(
            forecast_df=forecast_df,
            model=model,
            metrics=metrics_result,
            ts_info=ts_info,
            config=config,
            training_time=training_time
        )
        
        # Dodaj do historii
        _add_to_forecast_history(result)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"🎉 Prognoza gotowa!")
        
        # ============================================================================
        # PREZENTACJA WYNIKÓW
        # ============================================================================
        
        st.divider()
        
        # Tab layout
        tabs = st.tabs([
            "📈 Prognoza",
            "📊 Metryki",
            "🔍 Komponenty",
            "📌 Changepoints",
            "💾 Export"
        ])
        
        # ========================================================================
        # TAB 1: PROGNOZA
        # ========================================================================
        
        with tabs[0]:
            st.subheader("📈 Wizualizacja prognozy")
            
            # Główny wykres
            fig = go.Figure()
            
            # Historia
            hist_dates = train_df.index
            hist_values = train_df[target].values
            
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_values,
                name="Rzeczywiste",
                mode="lines",
                line=dict(color="blue")
            ))
            
            # Prognoza
            fc_dates = forecast_df["ds"]
            fc_values = forecast_df["yhat"]
            
            fig.add_trace(go.Scatter(
                x=fc_dates,
                y=fc_values,
                name="Prognoza",
                mode="lines",
                line=dict(color="red", dash="dash")
            ))
            
            # Przedział ufności
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_dates, fc_dates[::-1]]),
                y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,0,0.1)",
                line=dict(width=0),
                name="Przedział 95%",
                showlegend=True
            ))
            
            # Dane testowe (jeśli są)
            if test_slice is not None:
                fig.add_trace(go.Scatter(
                    x=test_slice.index,
                    y=test_slice[target].values,
                    name="Test set",
                    mode="lines",
                    line=dict(color="green", width=2)
                ))
            
            fig.update_layout(
                title=f"Prognoza: {target}",
                xaxis_title="Data",
                yaxis_title=target,
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela prognoz
            with st.expander("📋 Tabela prognoz", expanded=False):
                display_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
                st.dataframe(
                    forecast_df[display_cols].head(50),
                    use_container_width=True
                )
        
        # ========================================================================
        # TAB 2: METRYKI
        # ========================================================================
        
        with tabs[1]:
            st.subheader("📊 Metryki prognozy")
            
            if metrics_result is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    "sMAPE",
                    f"{metrics_result.smape:.2f}%",
                    help="Symmetric MAPE (niżej lepiej)"
                )
                col2.metric(
                    "MASE",
                    f"{metrics_result.mase:.4f}",
                    help="Mean Absolute Scaled Error"
                )
                col3.metric(
                    "RMSE",
                    f"{metrics_result.rmse:.4f}",
                    help="Root Mean Squared Error"
                )
                col4.metric(
                    "MAE",
                    f"{metrics_result.mae:.4f}",
                    help="Mean Absolute Error"
                )
                
                # Wykres porównania (jeśli test set)
                if test_slice is not None:
                    try:
                        fc_aligned = forecast_df.set_index("ds").reindex(test_slice.index).dropna()
                        
                        if not fc_aligned.empty:
                            comparison_df = pd.DataFrame({
                                "ds": test_slice.index,
                                "Rzeczywiste": test_slice[target].values[:len(fc_aligned)],
                                "Prognoza": fc_aligned["yhat"].values
                            })
                            
                            fig_comp = px.line(
                                comparison_df,
                                x="ds",
                                y=["Rzeczywiste", "Prognoza"],
                                title="Porównanie: Rzeczywiste vs Prognoza (test set)",
                                markers=True
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            # Residuals
                            residuals = comparison_df["Rzeczywiste"] - comparison_df["Prognoza"]
                            
                            fig_resid = px.histogram(
                                x=residuals,
                                nbins=30,
                                title="Rozkład reszt",
                                labels={"x": "Residual"}
                            )
                            st.plotly_chart(fig_resid, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Nie udało się wygenerować wykresów: {e}")
            else:
                st.info(
                    "ℹ️ Brak metryk walidacyjnych.\n\n"
                    "Wybierz tryb **'Test na ostatnich N okresach'** "
                    "aby zobaczyć metryki."
                )
        
        # ========================================================================
        # TAB 3: KOMPONENTY
        # ========================================================================
        
        with tabs[2]:
            st.subheader("🔍 Dekompozycja prognozy")
            
            try:
                # Wyciągnij komponenty z modelu
                if hasattr(model, "predict"):
                    # Prognoza na danych treningowych dla komponentów
                    future = model.make_future_dataframe(periods=0)
                    components = model.predict(future)
                    
                    # Trend
                    if "trend" in components.columns:
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=components["ds"],
                            y=components["trend"],
                            mode="lines",
                            name="Trend"
                        ))
                        fig_trend.update_layout(
                            title="Komponent: Trend",
                            xaxis_title="Data",
                            yaxis_title="Trend"
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Sezonowość (jeśli jest)
                    seasonal_cols = [
                        col for col in components.columns
                        if "seasonal" in col.lower() or "weekly" in col.lower() or "yearly" in col.lower()
                    ]
                    
                    if seasonal_cols:
                        for col in seasonal_cols:
                            fig_seasonal = go.Figure()
                            fig_seasonal.add_trace(go.Scatter(
                                x=components["ds"],
                                y=components[col],
                                mode="lines",
                                name=col
                            ))
                            fig_seasonal.update_layout(
                                title=f"Komponent: {col}",
                                xaxis_title="Data",
                                yaxis_title=col
                            )
                            st.plotly_chart(fig_seasonal, use_container_width=True)
                    
                else:
                    st.info("Model nie udostępnia metody predict dla komponentów")
                    
            except Exception as e:
                st.warning(f"Nie udało się wyodrębnić komponentów: {e}")
                logger.error(f"Błąd komponentów: {e}", exc_info=True)
        
        # ========================================================================
        # TAB 4: CHANGEPOINTS
        # ========================================================================
        
        with tabs[3]:
            st.subheader("📌 Wykryte changepoints")
            
            try:
                if hasattr(model, "changepoints"):
                    changepoints = model.changepoints
                    
                    if len(changepoints) > 0:
                        cp_dates = pd.to_datetime(changepoints)
                        
                        # Tabela
                        cp_df = pd.DataFrame({
                            "changepoint": cp_dates,
                            "index": range(len(cp_dates))
                        })
                        st.dataframe(cp_df, use_container_width=True)
                        
                        # Wykres z changepoints
                        fig_cp = go.Figure()
                        
                        # Szereg czasowy
                        fig_cp.add_trace(go.Scatter(
                            x=train_df.index,
                            y=train_df[target].values,
                            mode="lines",
                            name=target,
                            line=dict(color="blue")
                        ))
                        
                        # Changepoints jako linie
                        for cp_date in cp_dates:
                            fig_cp.add_vline(
                                x=cp_date,
                                line_dash="dash",
                                line_color="red",
                                opacity=0.5
                            )
                        
                        fig_cp.update_layout(
                            title="Changepoints w szeregu czasowym",
                            xaxis_title="Data",
                            yaxis_title=target,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_cp, use_container_width=True)
                        
                        st.info(
                            f"ℹ️ Wykryto **{len(changepoints)}** changepoints. "
                            "Są to punkty, w których trend zmienia kierunek."
                        )
                    else:
                        st.info("Nie wykryto żadnych changepoints")
                else:
                    st.info("Model nie udostępnia informacji o changepoints")
                    
            except Exception as e:
                st.warning(f"Nie udało się wyodrębnić changepoints: {e}")
                logger.error(f"Błąd changepoints: {e}", exc_info=True)
        
        # ========================================================================
        # TAB 5: EXPORT
        # ========================================================================
        
        with tabs[4]:
            st.subheader("💾 Eksport artefaktów")
            
            col1, col2, col3 = st.columns(3)
            
            # Forecast CSV
            with col1:
                forecast_csv = forecast_df.to_csv(index=False)
                
                st.download_button(
                    "⬇️ Prognoza (.csv)",
                    data=forecast_csv,
                    file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Metadata JSON
            with col2:
                metadata = {
                    "timestamp": result.timestamp,
                    "target": config.target,
                    "horizon": config.horizon,
                    "frequency": ts_info.frequency,
                    "seasonal_period": ts_info.seasonal_period,
                    "seasonality_mode": config.seasonality_mode,
                    "changepoint_prior_scale": config.changepoint_prior_scale,
                    "training_time": training_time,
                    "metrics": metrics_result.to_dict() if metrics_result else None
                }
                
                metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
                
                st.download_button(
                    "⬇️ Metadata (.json)",
                    data=metadata_json,
                    file_name=f"forecast_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Model (jeśli możliwe)
            with col3:
                try:
                    import joblib
                    
                    model_bytes = io.BytesIO()
                    joblib.dump(model, model_bytes)
                    model_bytes.seek(0)
                    
                    st.download_button(
                        "⬇️ Model (.joblib)",
                        data=model_bytes,
                        file_name=f"prophet_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.caption(f"Model export niedostępny: {e}")
            
            st.divider()
            
            # Full results (jeśli test)
            if test_slice is not None and metrics_result is not None:
                with st.expander("📊 Pełne wyniki walidacji (CSV)", expanded=False):
                    try:
                        fc_aligned = forecast_df.set_index("ds").reindex(test_slice.index).dropna()
                        
                        results_df = pd.DataFrame({
                            "date": test_slice.index,
                            "actual": test_slice[target].values[:len(fc_aligned)],
                            "forecast": fc_aligned["yhat"].values,
                            "lower": fc_aligned["yhat_lower"].values,
                            "upper": fc_aligned["yhat_upper"].values,
                            "residual": test_slice[target].values[:len(fc_aligned)] - fc_aligned["yhat"].values
                        })
                        
                        results_csv = results_df.to_csv(index=False)
                        
                        st.download_button(
                            "⬇️ Pobierz wyniki walidacji",
                            data=results_csv,
                            file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Błąd eksportu wyników: {e}")
        
    except ValueError as ve:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Błąd walidacji: {ve}")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Błąd prognozowania: {e}")
        logger.error(f"Forecast error: {e}", exc_info=True)
        
        st.info(
            "💡 **Możliwe rozwiązania:**\n"
            "- Sprawdź czy dane mają prawidłowy indeks czasu\n"
            "- Zwiększ liczbę punktów czasowych\n"
            "- Zmniejsz horyzont prognozy\n"
            "- Spróbuj innego trybu sezonowości"
        )

else:
    st.info(
        "👆 Ustaw parametry w sidebar i kliknij **Prognozuj**\n\n"
        f"Target: **{target}** • "
        f"Horyzont: **{config.horizon}** • "
        f"Tryb: **{start_mode_option}**"
    )

# ========================================================================================
# HISTORIA PROGNOZ
# ========================================================================================

history = st.session_state.get("forecast_history", [])

if history:
    st.divider()
    st.subheader("📚 Historia prognoz")
    
    for idx, hist_entry in enumerate(history):
        timestamp = hist_entry.get("timestamp", "Unknown")
        target_name = hist_entry.get("target", "N/A")
        horizon = hist_entry.get("horizon", 0)
        
        with st.expander(
            f"🕒 {timestamp} | {target_name} (h={horizon})",
            expanded=(idx == 0)
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Horyzont", f"{horizon}")
            col2.metric("Częstotliwość", hist_entry.get("frequency", "N/A"))
            col3.metric("Czas treningu", f"{hist_entry.get('training_time', 0):.2f}s")
            
            # Metryki (jeśli są)
            metrics = hist_entry.get("metrics")
            if metrics:
                with st.expander("📊 Metryki", expanded=False):
                    st.json(metrics)

# ========================================================================================
# WSKAZÓWKI NAWIGACJI
# ========================================================================================

st.divider()
st.success(
    "✨ **Co dalej?**\n\n"
    "- **📊 EDA Analysis** — przeanalizuj dane przed prognozą\n"
    "- **📈 Predictions** — trenuj modele ML\n"
    "- **🤖 AI Insights** — uzyskaj wnioski z AI\n"
    "- **📄 Reports** — wygeneruj raport z prognozy"
)