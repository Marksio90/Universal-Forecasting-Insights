# === PROPHET_PRO+++ ===
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import warnings, math
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore", category=UserWarning)

# === DATACLASSY ===
@dataclass
class ForecastParams:
    periods: int = 30
    freq: Optional[str] = None               # np. "D","H","W"; auto jeśli None
    seasonality_mode: str = "auto"           # "auto" | "additive" | "multiplicative"
    country_holidays: Optional[str] = None   # np. "PL","US"
    growth: str = "linear"                   # "linear" | "logistic"
    cap: Optional[float] = None              # wymagane dla logistic
    floor: Optional[float] = None
    changepoint_prior_scale: float = 0.05
    weekly_seasonality: Optional[bool] = None
    yearly_seasonality: Optional[bool] = None
    daily_seasonality: Optional[bool] = None
    add_regressors: Optional[Dict[str, Dict[str, Any]]] = None  # {"col":{"mode":"additive"}}
    tune_changepoint: bool = False
    cv_horizon: str = "30 days"
    cv_period: Optional[str] = None
    cv_initial: Optional[str] = None

@dataclass
class ForecastResult:
    model: Prophet
    forecast: pd.DataFrame
    params: ForecastParams
    metrics: Dict[str, float]                # MAPE/sMAPE/RMSE/MAE (jeśli dostępne)
    cv_table: Optional[pd.DataFrame] = None  # z prophet.diagnostics

# === POMOCNICZE ===
def _infer_freq(ts: pd.DataFrame) -> Optional[str]:
    try:
        f = pd.infer_freq(ts["ds"].sort_values().iloc[:50])
        # mapy drobnych aliasów
        return {"MS":"MS","M":"MS"}.get(f, f)
    except Exception:
        return None

def _is_positive(series: pd.Series) -> bool:
    return bool((series > 0).all())

def _seasonality_auto(freq: Optional[str]) -> Dict[str, bool]:
    # Rozsądne defaulty: dzień/tydzień/rok w zależności od częstotliwości
    if not freq:
        return {"daily": False, "weekly": True, "yearly": True}
    f = freq.upper()
    if f.startswith("H"):   # godzinowe
        return {"daily": True, "weekly": True, "yearly": True}
    if f.startswith("D"):   # dobowe
        return {"daily": False, "weekly": True, "yearly": True}
    if f.startswith("W"):   # tygodniowe
        return {"daily": False, "weekly": False, "yearly": True}
    if f.startswith("MS") or f.startswith("M"):  # miesięczne
        return {"daily": False, "weekly": False, "yearly": True}
    return {"daily": False, "weekly": True, "yearly": True}

def _calc_basic_metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str,float]:
    y_true = y_true.astype(float)
    y_hat = y_hat.astype(float)
    mae = float(np.mean(np.abs(y_true - y_hat)))
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    # MAPE/sMAPE z ochroną przed 0
    eps = 1e-9
    mape = float(np.mean(np.abs((y_true - y_hat) / np.clip(np.abs(y_true), eps, None))) * 100.0)
    smape = float(np.mean(200.0 * np.abs(y_hat - y_true) / (np.abs(y_true) + np.abs(y_hat) + eps)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}

# === PREPARE TS ===
def prepare_ts(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    *,
    freq: Optional[str] = None,
    impute: str = "ffill",            # "ffill"|"bfill"|"zero"|"mean"|"none"
    clip_outliers_sigma: Optional[float] = 4.0,
) -> pd.DataFrame:
    """
    Przygotuj szeregi: kolumny -> ds,y; sortowanie; resampling do freq; imputacja braków; miękkie ucięcie outlierów.
    """
    ts = df[[date_col, target]].copy()
    ts.columns = ["ds", "y"]
    # Daty
    ts["ds"] = pd.to_datetime(ts["ds"], errors="coerce")
    ts = ts.dropna(subset=["ds"]).sort_values("ds")
    # Resampling (jeśli brak równego kroku)
    f = freq or _infer_freq(ts)
    if f:
        ts = ts.set_index("ds").resample(f).mean(numeric_only=True)
        ts["y"] = ts["y"].interpolate(limit_direction="both")
        ts = ts.reset_index()
    # Imputacja braków
    if ts["y"].isna().any():
        if impute == "ffill":
            ts["y"] = ts["y"].ffill().bfill()
        elif impute == "bfill":
            ts["y"] = ts["y"].bfill().ffill()
        elif impute == "zero":
            ts["y"] = ts["y"].fillna(0.0)
        elif impute == "mean":
            ts["y"] = ts["y"].fillna(ts["y"].mean())
    # Outliery (winsoryzacja)
    if clip_outliers_sigma and len(ts) > 20:
        mu, sd = ts["y"].mean(), ts["y"].std()
        lo, hi = mu - clip_outliers_sigma * sd, mu + clip_outliers_sigma * sd
        ts["y"] = ts["y"].clip(lower=lo, upper=hi)
    return ts

# === FIT & FORECAST (BACKWARD COMPAT WRAPPER) ===
def fit_forecast(ts: pd.DataFrame, periods: int = 30) -> Tuple[Prophet, pd.DataFrame]:
    """
    Zgodne z Twoim oryginałem: szybki fit + prognoza + zwrot (model, forecast).
    """
    params = ForecastParams(periods=periods)
    res = fit_forecast_pro(ts, params)
    return res.model, res.forecast

# === FIT PRO ===
def fit_forecast_pro(ts: pd.DataFrame, params: ForecastParams) -> ForecastResult:
    """
    Wersja PRO: automatyczna sezonowość/święta, opcjonalny tuning changepointów, metryki i tabela CV (jeśli możliwe).
    """
    assert {"ds","y"} <= set(ts.columns), "Input ts must have columns: ['ds','y']"
    ts = ts.sort_values("ds").dropna()
    freq = params.freq or _infer_freq(ts)
    season = _seasonality_auto(freq)

    # Sezonowość explicit?
    weekly = params.weekly_seasonality if params.weekly_seasonality is not None else season["weekly"]
    yearly = params.yearly_seasonality if params.yearly_seasonality is not None else season["yearly"]
    daily  = params.daily_seasonality  if params.daily_seasonality  is not None else season["daily"]

    # Tryb sezonowości
    if params.seasonality_mode == "auto":
        # multiplicative tylko gdy dane dodatnie
        seasonality_mode = "multiplicative" if _is_positive(ts["y"]) else "additive"
    else:
        seasonality_mode = params.seasonality_mode

    # Growth (logistic wymaga cap/floor)
    m_kwargs: Dict[str, Any] = dict(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=params.changepoint_prior_scale,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=daily,
    )
    m = Prophet(**m_kwargs)
    if params.country_holidays:
        try:
            m.add_country_holidays(country_name=params.country_holidays)
        except Exception:
            pass

    # Regressors
    if params.add_regressors:
        for col, kw in params.add_regressors.items():
            m.add_regressor(col, **{k:v for k,v in (kw or {}).items() if k in {"mode","standardize","prior_scale"}})

    ts_fit = ts.copy()
    if params.growth == "logistic":
        if params.cap is None:
            raise ValueError("For logistic growth, 'cap' must be provided in ForecastParams")
        ts_fit["cap"] = float(params.cap)
        if params.floor is not None:
            ts_fit["floor"] = float(params.floor)
        m.growth = "logistic"

    # Fit
    m.fit(ts_fit)

    # Opcjonalny tuning changepointów (lekki – 3 kandydatury + CV)
    if params.tune_changepoint:
        cps_candidates = [0.02, 0.05, 0.1]
        best_cps, best_mape = params.changepoint_prior_scale, math.inf
        for cps in cps_candidates:
            mt = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=cps,
                weekly_seasonality=weekly,
                yearly_seasonality=yearly,
                daily_seasonality=daily,
            )
            if params.country_holidays:
                try: mt.add_country_holidays(country_name=params.country_holidays)
                except Exception: pass
            if params.add_regressors:
                for col, kw in params.add_regressors.items():
                    mt.add_regressor(col, **{k:v for k,v in (kw or {}).items() if k in {"mode","standardize","prior_scale"}})
            mt.fit(ts_fit)
            try:
                cv = cross_validation(mt, horizon=params.cv_horizon, period=params.cv_period, initial=params.cv_initial)
                pm = performance_metrics(cv)
                mape = float(pm["mape"].mean() * 100.0)
                if mape < best_mape:
                    best_mape, best_cps = mape, cps
            except Exception:
                pass
        if best_cps != params.changepoint_prior_scale:
            # refit z najlepszym cps
            m = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=best_cps,
                weekly_seasonality=weekly,
                yearly_seasonality=yearly,
                daily_seasonality=daily,
            )
            if params.country_holidays:
                try: m.add_country_holidays(country_name=params.country_holidays)
                except Exception: pass
            if params.add_regressors:
                for col, kw in params.add_regressors.items():
                    m.add_regressor(col, **{k:v for k,v in (kw or {}).items() if k in {"mode","standardize","prior_scale"}})
            m.fit(ts_fit)

    # Future
    fut = m.make_future_dataframe(periods=params.periods, freq=freq or "D")
    if params.growth == "logistic":
        fut["cap"] = float(params.cap)
        if params.floor is not None:
            fut["floor"] = float(params.floor)

    # Forecast
    fcst = m.predict(fut)

    # Metryki in-sample (na wspólnym zakresie)
    merged = pd.merge(ts[["ds","y"]], fcst[["ds","yhat"]], on="ds", how="inner")
    metrics = _calc_basic_metrics(merged["y"], merged["yhat"])

    # CV tabelka (best effort)
    cv_tbl: Optional[pd.DataFrame] = None
    try:
        cv = cross_validation(m, horizon=params.cv_horizon, period=params.cv_period, initial=params.cv_initial)
        cv_tbl = performance_metrics(cv)
        # preferuj metryki z CV jeśli dostępne
        if "rmse" in cv_tbl.columns:
            metrics["rmse_cv"] = float(cv_tbl["rmse"].mean())
        if "mape" in cv_tbl.columns:
            metrics["mape_cv"] = float(cv_tbl["mape"].mean() * 100.0)
        if "mae" in cv_tbl.columns:
            metrics["mae_cv"] = float(cv_tbl["mae"].mean())
    except Exception:
        pass

    return ForecastResult(model=m, forecast=fcst, params=params, metrics=metrics, cv_table=cv_tbl)
