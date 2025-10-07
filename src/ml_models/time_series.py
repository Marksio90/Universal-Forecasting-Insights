# src/ml_models/forecasting.py — TURBO PRO v2 (compat)
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from prophet import Prophet

from ..utils.helpers import ensure_datetime_index  # Twoja utilka (zostawiamy)

DATE_HINTS = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")

# ======== Konstants & limits ========
MIN_TIME_POINTS = 10
MAX_TIME_POINTS = 100_000
MAX_HORIZON = 10_000

# ======== Dataclasses ========
@dataclass(frozen=True)
class ForecastMetrics:
    rmse: float
    mae: float
    r2: float
    smape: float
    mase: float

@dataclass(frozen=True)
class BacktestFold:
    fold: int
    train_end: str
    test_len: int
    metrics: ForecastMetrics

# ======== Heurystyki czasu ========
def _detect_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback, gdy ensure_datetime_index nie ustawi DatetimeIndex."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    cand = None
    for c in out.columns:
        lc = str(c).lower()
        if any(h in lc for h in DATE_HINTS):
            cand = c
            break

    if cand:
        try:
            out[cand] = pd.to_datetime(out[cand], errors="coerce", infer_datetime_format=True)
            out = out.set_index(cand).sort_index()
            return out
        except Exception:
            pass

    # ostatnia próba — pierwsza kolumna
    first = out.columns[0]
    try:
        out[first] = pd.to_datetime(out[first], errors="coerce", infer_datetime_format=True)
        if out[first].notna().mean() > 0.6:
            out = out.set_index(first).sort_index()
            return out
    except Exception:
        pass

    return df

def _infer_freq(idx: pd.DatetimeIndex) -> str:
    """Zgadnij częstotliwość; mapuj do aliasów lubianych przez Propheta."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return "D"
    f = pd.infer_freq(idx)
    if f:
        f = f.upper()
        if f.startswith(("A", "Y")): return "Y"
        if f.startswith("Q"):        return "Q"
        if f.startswith("M"):        return "M"
        if f.startswith("W"):        return "W"
        if f.startswith("D"):        return "D"
        if f.startswith("H"):        return "H"
        if f.startswith("T"):        return "MIN"
        if f.startswith("S"):        return "S"
    # heurystyka po medianie różnic
    diffs = np.diff(idx.view("i8"))  # ns
    if len(diffs) == 0:
        return "D"
    med_s = float(np.median(diffs) / 1e9)
    if med_s < 60:       return "S"
    if med_s < 3600:     return "MIN"
    if med_s < 86400:    return "H"
    if med_s < 7*86400:  return "D"
    if med_s < 28*86400: return "W"
    if med_s < 92*86400: return "M"
    if med_s < 366*86400:return "Q"
    return "Y"

def _boolean_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    f = (freq or "D").upper()
    yearly = True
    weekly = f in ("D", "H", "S", "MIN")
    daily  = f in ("H", "S", "MIN")
    return yearly, weekly, daily

def _seasonal_period(freq: str) -> int:
    """Okres sezonowy do MASE (heurystyki)."""
    f = (freq or "D").upper()
    return {
        "S": 60,     # minuta
        "MIN": 60,   # godzina
        "H": 24,     # doby
        "D": 7,      # tydzień
        "W": 52,     # rok
        "M": 12, "MS": 12,
        "Q": 4,  "QS": 4,
        "Y": 1,
    }.get(f, 1)

def _freq_alias_for_prophet(freq_in: str) -> str:
    return {
        "Y": "Y", "A": "Y",
        "Q": "Q", "QS": "QS",
        "M": "MS", "MS": "MS",
        "W": "W",
        "D": "D",
        "H": "H",
        "MIN": "min",
        "S": "S",
    }.get((freq_in or "D").upper(), "D")

# ======== Przygotowanie danych ========
def _prepare_y(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _pick_exogenous(df: pd.DataFrame, target: str, limit: int = 15) -> List[str]:
    """Wybierz do `limit` numerycznych, zmiennych regresorów (na bazie odchylenia)."""
    cols = []
    for c in df.columns:
        if c == target: continue
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser) and ser.nunique(dropna=True) >= 3:
            cols.append(c)
    cols = sorted(cols, key=lambda c: float(df[c].std(skipna=True) or 0.0), reverse=True)[:limit]
    return cols

# ======== Metryki ========
def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray, m: int) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_ins  = np.asarray(y_insample, dtype=float)
    m = max(int(m or 1), 1)
    if len(y_ins) <= m:
        return float("nan")
    scale = np.mean(np.abs(y_ins[m:] - y_ins[:-m]))
    if scale == 0 or np.isnan(scale):
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def _basic_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    # proste R^2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) or np.nan
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot and not np.isnan(ss_tot) else float("nan")
    return rmse, mae, r2

# ======== Backtest ========
def _holdout_metrics(
    df_idx: pd.DataFrame,
    target: str,
    freq_in: str,
    horizon: int,
    model_builder: Callable[[pd.DataFrame, List[str], Dict[str, Any]], Prophet],
    model_opts: Dict[str, Any],
    chosen_reg: List[str],
) -> ForecastMetrics:
    """Trenuje na części, testuje na ostatnich N=horizon punktach (min 20% lub 1)."""
    n = len(df_idx)
    test_len = max(1, min(horizon, max(int(np.ceil(0.2*n)), 1)))
    train_df = df_idx.iloc[:-test_len].copy()
    test_df  = df_idx.iloc[-test_len:].copy()

    # przygotuj dataframes
    y_train = _prepare_y(train_df[target]).dropna()
    data_tr = pd.DataFrame({"ds": train_df.index, "y": y_train}).dropna(subset=["y"])

    if model_opts.get("growth") == "logistic":
        cap = model_opts.get("cap") or float(np.nanmax(data_tr["y"]) * 1.2)
        floor = model_opts.get("floor") or float(np.nanmin(data_tr["y"]) * 0.8)
        data_tr["cap"], data_tr["floor"] = cap, floor
        model_opts = {**model_opts, "cap": cap, "floor": floor}

    for c in chosen_reg:
        data_tr[c] = pd.to_numeric(train_df[c], errors="coerce").reindex(data_tr["ds"]).values

    # fit + predict
    model = model_builder(data_tr, chosen_reg, model_opts)
    freq_alias = _freq_alias_for_prophet(freq_in)
    future = model.make_future_dataframe(periods=test_len, freq=freq_alias, include_history=False)

    if model_opts.get("growth") == "logistic":
        future["cap"], future["floor"] = model_opts["cap"], model_opts["floor"]
    if chosen_reg:
        aux = data_tr.set_index("ds")[chosen_reg]
        future = future.set_index("ds")
        for c in chosen_reg:
            future[c] = aux[c].reindex(future.index).ffill()
        future = future.reset_index()

    fc = model.predict(future)[["ds", "yhat"]].set_index("ds")
    # align prawda vs prognoza
    test_truth = _prepare_y(test_df[target]).astype(float)
    y_true = test_truth.reindex(fc.index).astype(float).values
    y_pred = fc["yhat"].astype(float).values

    rmse, mae, r2 = _basic_reg_metrics(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    m = _seasonal_period(freq_in)
    mase = _mase(y_true, y_pred, _prepare_y(train_df[target]).values, m)

    return ForecastMetrics(rmse=rmse, mae=mae, r2=r2, smape=smape, mase=mase)

# ======== Główny interfejs (kompatybilny) ========
def fit_prophet(
    df: pd.DataFrame,
    target: str,
    horizon: int = 12,
    **options: Any,
):
    """
    Trenuje model Prophet i zwraca (model, fcst) dla najbliższych `horizon` okresów.

    Opcje (**options):
      seasonality_mode: "additive" | "multiplicative" (domyślnie "additive")
      changepoint_prior_scale: float (domyślnie 0.1)
      growth: "linear" | "logistic" (domyślnie "linear")
      cap, floor: float (dla growth="logistic"; auto jeśli brak)
      extra_regressors: None | "auto" | List[str]
      holidays: DataFrame z kolumnami ["ds","holiday"] (opcjonalnie)
      country_holidays: Optional[str] (np. "PL", "US")
      include_history: bool (domyślnie False)
      freq: str (np. "MS" żeby wymusić miesiące od początku)
      outlier_clip: None | "iqr" | "zscore"
      backtest: Optional[int] liczba foldów (0/None = wyłączone; 1 = tylko holdout)
    """
    if target not in df.columns:
        raise ValueError("Brak kolumny celu w danych.")

    if horizon < 1 or horizon > MAX_HORIZON:
        raise ValueError(f"horizon poza zakresem (1..{MAX_HORIZON}).")

    # 1) Indeks czasu
    df2 = df.copy()
    try:
        df2 = ensure_datetime_index(df2)
    except Exception:
        pass
    if not isinstance(df2.index, pd.DatetimeIndex):
        df2 = _detect_datetime_index(df2)
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("Nie znaleziono kolumny czasu – nie można trenować Prophet.")

    # sanity: rozmiar i duplikaty
    df2 = df2.sort_index()
    if len(df2) < MIN_TIME_POINTS:
        raise ValueError(f"Za mało obserwacji ({len(df2)} < {MIN_TIME_POINTS}).")
    if len(df2) > MAX_TIME_POINTS:
        df2 = df2.iloc[-MAX_TIME_POINTS:].copy()
    if df2.index.duplicated().any():
        # agreguj duplikaty po średniej
        df2 = df2.groupby(df2.index).mean(numeric_only=True)

    # 2) y + outlier_clip
    y_all = _prepare_y(df2[target])
    clip_mode = (options.get("outlier_clip") or "").lower() if options.get("outlier_clip") else None
    if clip_mode == "iqr":
        q1, q3 = y_all.quantile(0.25), y_all.quantile(0.75)
        iqr = float(q3 - q1)
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        y_all = y_all.clip(lower=lo, upper=hi)
    elif clip_mode == "zscore":
        m, s = y_all.mean(skipna=True), y_all.std(skipna=True) or 1.0
        z = (y_all - m) / s
        y_all = y_all.mask(z.abs() > 4.0, np.nan)

    data = pd.DataFrame({"ds": df2.index, "y": y_all}).dropna(subset=["y"])

    # 3) Opcje modelu
    seasonality_mode = options.get("seasonality_mode", "additive")
    cps = float(options.get("changepoint_prior_scale", 0.1))
    growth = options.get("growth", "linear")
    cap = options.get("cap")
    floor = options.get("floor")
    holidays = options.get("holidays")
    country_holidays = options.get("country_holidays")  # np. "PL"
    include_history = bool(options.get("include_history", False))
    freq_in = (options.get("freq") or _infer_freq(df2.index)).upper()

    # growth logistic: auto cap/floor
    if growth == "logistic":
        if cap is None:
            cap = float(np.nanmax(data["y"]) * 1.2) if len(data) else 1.0
        if floor is None:
            floor = float(np.nanmin(data["y"]) * 0.8) if len(data) else 0.0
        data["cap"] = cap
        data["floor"] = floor

    # 4) Regresory
    extra_reg = options.get("extra_regressors")
    chosen_reg: List[str] = []
    if isinstance(extra_reg, list):
        chosen_reg = [c for c in extra_reg if c in df2.columns and c != target]
    elif isinstance(extra_reg, str) and extra_reg.lower() == "auto":
        chosen_reg = _pick_exogenous(df2, target)
    for c in chosen_reg:
        data[c] = pd.to_numeric(df2[c], errors="coerce").reindex(data["ds"]).values

    # 5) Budowa modelu
    yearly, weekly, daily = _boolean_seasonality_flags(freq_in)
    def _build_model(dataframe: pd.DataFrame, regs: List[str], opts: Dict[str, Any]) -> Prophet:
        m = Prophet(
            seasonality_mode=opts.get("seasonality_mode", seasonality_mode),
            changepoint_prior_scale=opts.get("changepoint_prior_scale", cps),
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            growth=opts.get("growth", growth),
            holidays=opts.get("holidays", holidays),
        )
        if country_holidays:
            try:
                m.add_country_holidays(country_name=str(country_holidays))
            except Exception:
                # ignorujemy, jeśli nieznane państwo
                pass

        # Dodatkowe sezonowości
        if freq_in in ("M", "MS"):
            m.add_seasonality(name="monthly", period=12, fourier_order=8)
        if freq_in in ("Q", "QS"):
            m.add_seasonality(name="quarterly", period=4, fourier_order=4)

        for r in regs:
            m.add_regressor(r, standardize="auto")
        m.fit(dataframe)
        return m

    model = _build_model(data, chosen_reg, {
        "seasonality_mode": seasonality_mode,
        "changepoint_prior_scale": cps,
        "growth": growth,
        "holidays": holidays,
    })

    # 6) Przyszłość
    freq_alias = _freq_alias_for_prophet(freq_in)
    future = model.make_future_dataframe(periods=int(horizon), freq=freq_alias, include_history=include_history)

    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = floor

    if chosen_reg:
        aux = data.set_index("ds")[chosen_reg]
        future = future.set_index("ds")
        for c in chosen_reg:
            future[c] = aux[c].reindex(future.index).ffill()
        future = future.reset_index()

    # 7) Prognoza
    fcst = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # 8) Metryki (holdout z końcówki; bez ciężkiego CV)
    metrics = _holdout_metrics(
        df_idx=df2, target=target, freq_in=freq_in, horizon=horizon,
        model_builder=_build_model, model_opts={
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": cps,
            "growth": growth,
            "holidays": holidays,
            "cap": cap, "floor": floor
        },
        chosen_reg=chosen_reg,
    )

    # 9) (Opcjonalnie) rolling backtest — lightweight
    folds = int(options.get("backtest") or 0)
    backtest: List[BacktestFold] = []
    if folds and folds > 1:
        n = len(df2)
        step = max(1, min(horizon, int(np.ceil(n / (folds+1)))))
        for k in range(1, folds+1):
            cut = n - k*step
            if cut <= MIN_TIME_POINTS: break
            part = df2.iloc[:cut].copy()
            bt_metrics = _holdout_metrics(
                df_idx=part, target=target, freq_in=freq_in, horizon=step,
                model_builder=_build_model, model_opts={
                    "seasonality_mode": seasonality_mode,
                    "changepoint_prior_scale": cps,
                    "growth": growth,
                    "holidays": holidays,
                    "cap": cap, "floor": floor
                }, chosen_reg=chosen_reg,
            )
            backtest.append(
                BacktestFold(
                    fold=k,
                    train_end=str(part.index.max().date()),
                    test_len=step,
                    metrics=bt_metrics
                )
            )

    # 10) Metadane
    fcst.attrs["forecast_meta"] = {
        "target": target,
        "n_obs": int(len(data)),
        "freq_inferred": freq_in,
        "freq": freq_alias,
        "horizon": int(horizon),
        "seasonality_mode": seasonality_mode,
        "changepoint_prior_scale": float(cps),
        "extra_regressors": chosen_reg or None,
        "growth": growth,
        "cap": cap if growth == "logistic" else None,
        "floor": floor if growth == "logistic" else None,
        "include_history": include_history,
        "metrics": asdict(metrics),
        "backtest": [  # krótko, do logów/eksportu
            {"fold": b.fold, "train_end": b.train_end, **asdict(b.metrics)} for b in backtest
        ] or None,
    }

    return model, fcst
