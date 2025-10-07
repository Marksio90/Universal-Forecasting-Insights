# src/ml_models/forecasting.py — TURBO PRO (back-compat API)
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet

# =========================
# Logger
# =========================
LOGGER = logging.getLogger("forecasting")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(_h)
    LOGGER.propagate = False

# =========================
# Stałe / heurystyki
# =========================
DATE_HINTS = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")
MIN_TIME_POINTS = 10
MAX_TIME_POINTS = 100_000
MAX_HORIZON_STEPS = 1200  # twardy limit bezpieczeństwa

# =========================
# Utils: czas i częstotliwość
# =========================
def _detect_datetime_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Ustawia (i sortuje) indeks czasowy.
    - preferuje `date_col` jeśli podano,
    - w innym wypadku szuka po nazwach z DATE_HINTS,
    - ostatni fallback: próba parsowania 1. kolumny.
    """
    out = df.copy()
    # jeśli już mamy DatetimeIndex → tylko posortuj i od-strefuj
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        if idx.tz is not None:
            out.index = idx.tz_convert("UTC").tz_localize(None)
        return out.sort_index()

    cand = None
    if date_col and date_col in out.columns:
        cand = date_col
    else:
        for c in out.columns:
            lc = str(c).lower()
            if any(h in lc for h in DATE_HINTS):
                cand = c
                break

    def _try_set(col: str) -> Optional[pd.DataFrame]:
        series = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)
        if series.notna().mean() < 0.5:
            return None
        idx = series
        if idx.dt.tz is not None:
            idx = idx.dt.tz_convert("UTC").dt.tz_localize(None)
        tmp = out.copy()
        tmp[col] = idx
        tmp = tmp.set_index(col)
        return tmp.sort_index()

    if cand is not None:
        try:
            tmp = _try_set(cand)
            if tmp is not None:
                return tmp
        except Exception:
            pass

    # fallback: 1. kolumna
    try:
        first = out.columns[0]
        tmp = _try_set(first)
        if tmp is not None:
            return tmp
    except Exception:
        pass

    return out  # bez indeksu czasu — później rzucimy kontrolowany błąd


def _infer_freq(idx: pd.DatetimeIndex) -> str:
    """
    Zgadnij częstotliwość z indeksu; fallback na heurystykę po medianie różnic.
    Normalizuje aliasy do: Y, Q, M/MS, W, D, H, min, S.
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return "D"

    try:
        f = pd.infer_freq(idx)
    except Exception:
        f = None

    if f:
        f = f.upper()
        # normalizacja najczęstszych wariantów
        if f.startswith("A") or f.startswith("Y"):
            return "Y"
        if f.startswith("Q"):
            return "Q"
        if f in ("MS", "M"):
            return "MS"
        if f.startswith("M"):
            return "MS"
        if f.startswith("W"):
            return "W"
        if f.startswith("D"):
            return "D"
        if f.startswith("H"):
            return "H"
        if f.startswith("T"):
            return "min"
        if f.startswith("S"):
            return "S"

    # heurystyka po medianie odstępów (ns → s)
    diffs = np.diff(idx.view("i8"))
    if len(diffs) == 0:
        return "D"
    med_s = float(np.median(diffs) / 1e9)
    day = 86400.0
    if med_s < 60: return "S"
    if med_s < 3600: return "min"
    if med_s < day: return "H"
    if med_s < 7 * day: return "D"
    if med_s < 28 * day: return "W"
    if med_s < 92 * day: return "MS"
    if med_s < 366 * day: return "Q"
    return "Y"


def _boolean_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    """Określa, które sezonowości Propheta mają sens przy danej częstotliwości."""
    f = (freq or "D").upper()
    yearly = True
    weekly = f in ("D", "H", "S", "MIN")
    daily = f in ("H", "S", "MIN")
    return yearly, weekly, daily


def _freq_to_prophet_alias(freq: str) -> str:
    """Mapuje alias na taki, jaki lubi Prophet.make_future_dataframe."""
    return {
        "Y": "Y",
        "A": "Y",
        "Q": "Q",
        "QS": "QS",
        "M": "MS",
        "MS": "MS",
        "W": "W",
        "D": "D",
        "H": "H",
        "MIN": "min",
        "S": "S",
    }.get(freq.upper(), "D")


def _seasonal_period_from_freq(freq: str) -> int:
    """Heurystyka okresu sezonowego m (dla MASE i interpretacji)."""
    f = freq.upper()
    if f in ("MS", "M"): return 12
    if f in ("Q", "QS"): return 4
    if f == "W": return 52
    if f == "D": return 7
    if f == "H": return 24
    if f == "MIN": return 60
    if f == "S": return 60
    return 1


# =========================
# Utils: przygotowanie i metryki
# =========================
def _prepare_y(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    return y


def _pick_exogenous(df: pd.DataFrame, target: str, limit: int = 15) -> List[str]:
    """Wybór kandydatów na regresory zewnętrzne (numeryczne, >=3 wartości unikalne, najwyższa zmienność)."""
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) >= 3:
            cols.append(c)
    cols = sorted(cols, key=lambda c: float(df[c].std(skipna=True) or 0.0), reverse=True)[:limit]
    return cols


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.abs(y_true - y_pred)
    b = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(b == 0, 1.0, b)
    return float(np.mean(a / denom) * 100.0)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, m: int = 1) -> float:
    y = np.asarray(y_true, dtype=float)
    if len(y) <= m + 1:
        m = 1
    denom = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.mean(np.abs(np.diff(y)))
    denom = denom if denom and np.isfinite(denom) else 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


# =========================
# Publiczny interfejs
# =========================
def forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = 12,
    *,
    date_col: Optional[str] = None,
    seasonality_mode: str = "additive",            # "additive" | "multiplicative"
    changepoint_prior_scale: float = 0.1,          # większa → bardziej czuły trend
    extra_regressors: Union[None, str, List[str]] = None,  # None | "auto" | ["col1", ...]
    holidays: Optional[pd.DataFrame] = None,       # opcjonalnie: kolumny ["ds","holiday", ...]
    growth: str = "linear",                        # "linear" | "logistic"
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    freq: Optional[str] = None,                    # np. "MS"
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Trenuje Prophet i zwraca (model, forecast_df).
    forecast_df zawiera: `ds`, `yhat`, `yhat_lower`, `yhat_upper` (+ attrs['forecast_meta']).
    """
    # --- walidacje wejścia ---
    assert isinstance(df, pd.DataFrame) and target in df.columns, "Nieprawidłowe dane lub brak kolumny celu."
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("`horizon` musi być dodatnią liczbą całkowitą.")
    if horizon > MAX_HORIZON_STEPS:
        LOGGER.warning("przycinam horizon=%s → %s (MAX_HORIZON_STEPS)", horizon, MAX_HORIZON_STEPS)
        horizon = MAX_HORIZON_STEPS

    # 1) Indeks czasu + sanity
    frame = _detect_datetime_index(df, date_col=date_col)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Nie wykryto prawidłowej kolumny czasu. Podaj `date_col` lub dodaj kolumnę daty.")

    # posortuj, usuń duplikaty w indeksie (ostatnia wygrywa), usuń NaT
    frame = frame[frame.index.notna()].sort_index()
    if frame.index.has_duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]

    if len(frame) < MIN_TIME_POINTS:
        raise ValueError(f"Zbyt mało punktów czasowych: {len(frame)} < {MIN_TIME_POINTS}.")
    if len(frame) > MAX_TIME_POINTS:
        LOGGER.warning("przycinam liczbę punktów %s → %s (MAX_TIME_POINTS)", len(frame), MAX_TIME_POINTS)
        frame = frame.iloc[-MAX_TIME_POINTS:, :]

    # 2) Przygotowanie ds/y (+ regresory)
    y = _prepare_y(frame[target])
    data = pd.DataFrame({"ds": frame.index, "y": y}).dropna(subset=["y"])

    # growth logistic – bezpieczne cap/floor
    if growth == "logistic":
        y_max = float(np.nanmax(data["y"])) if len(data) else 1.0
        y_min = float(np.nanmin(data["y"])) if len(data) else 0.0
        if cap is None or cap <= y_max:
            cap = max(y_max * 1.2, y_max + 1e-6)
        if floor is None or floor >= cap or floor >= y_min:
            floor = min(y_min * 0.8, cap - 1e-6)
        data["cap"] = cap
        data["floor"] = floor

    # regresory zewnętrzne
    chosen_reg: List[str] = []
    if isinstance(extra_regressors, list):
        chosen_reg = [c for c in extra_regressors if c in frame.columns and c != target]
    elif isinstance(extra_regressors, str) and extra_regressors.lower() == "auto":
        chosen_reg = _pick_exogenous(frame, target)

    for c in chosen_reg:
        data[c] = pd.to_numeric(frame[c], errors="coerce")

    # 3) Konfiguracja częstotliwości i sezonowości
    _freq = (freq or _infer_freq(frame.index))
    yearly, weekly, daily = _boolean_seasonality_flags(_freq)

    # 4) Model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=float(changepoint_prior_scale),
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        growth=growth,
        holidays=holidays if _is_valid_holidays(holidays) else None,
    )

    # dodatkowe sezonowości (miesięczna/kwartalna) gdy ma sens
    if _freq in ("M", "MS"):
        model.add_seasonality(name="monthly", period=12, fourier_order=8)
    if _freq in ("Q", "QS"):
        model.add_seasonality(name="quarterly", period=4, fourier_order=4)

    # rejestrowanie regresorów
    for c in chosen_reg:
        model.add_regressor(c, standardize="auto")

    # 5) Trening
    model.fit(data)

    # 6) Future DF
    freq_alias = _freq_to_prophet_alias(_freq)
    future = model.make_future_dataframe(periods=int(horizon), freq=freq_alias, include_history=True)

    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = floor

    # Przyszłe wartości regresorów – prosty FFill ostatniej obserwacji
    if chosen_reg:
        aux = data.set_index("ds")[chosen_reg]
        future = future.set_index("ds")
        for c in chosen_reg:
            future[c] = aux[c].reindex(future.index).ffill()
        future = future.reset_index()

    # 7) Prognoza
    pred = model.predict(future)
    fcst = pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # 8) Metryki (na końcówce historii – pseudo holdout z in-sample)
    hist = fcst.merge(data[["ds", "y"]], on="ds", how="left")
    hist_mask = hist["y"].notna()
    hist_known = hist.loc[hist_mask].copy()
    # walidacja na końcówce: min( max(12, horizon), 25% historii )
    val_len = int(min(max(12, horizon), max(1, len(hist_known) // 4)))
    tail = hist_known.tail(val_len)
    if not tail.empty:
        y_true = tail["y"].to_numpy(dtype=float)
        y_hat = tail["yhat"].to_numpy(dtype=float)
        smape = _smape(y_true, y_hat)
        m = _seasonal_period_from_freq(_freq)
        mase = _mase(y_true, y_hat, m=m)
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_hat)))
        metrics = {"sMAPE": smape, "MASE": mase, "RMSE": rmse, "MAE": mae, "val_len": int(val_len)}
    else:
        metrics = None

    # 9) Metadane
    fcst.attrs["forecast_meta"] = {
        "target": target,
        "n_obs": int(len(data)),
        "freq": freq_alias,
        "horizon": int(horizon),
        "seasonality_mode": seasonality_mode,
        "changepoint_prior_scale": float(changepoint_prior_scale),
        "extra_regressors": chosen_reg or None,
        "growth": growth,
        "cap": cap,
        "floor": floor,
        "metrics_tail": metrics,
        "seasonal_period_hint": _seasonal_period_from_freq(_freq),
        "warnings": _collect_warnings(frame, chosen_reg),
    }

    return model, fcst


# =========================
# Drobne helpery bezpieczeństwa
# =========================
def _is_valid_holidays(h: Optional[pd.DataFrame]) -> bool:
    if h is None or not isinstance(h, pd.DataFrame) or h.empty:
        return False
    cols = {c.lower() for c in h.columns}
    return "ds" in cols and "holiday" in cols


def _collect_warnings(frame: pd.DataFrame, regressors: List[str]) -> List[str]:
    warns: List[str] = []
    # nieregularny szereg
    try:
        if pd.infer_freq(frame.index) is None:
            warns.append("irregular_frequency_detected")
    except Exception:
        pass
    # braki w regresorach
    for c in regressors:
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.isna().mean() > 0.2:
            warns.append(f"regressor_{c}_many_nans({s.isna().mean():.0%})")
    return warns
