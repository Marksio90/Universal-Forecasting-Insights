# src/ml_models/forecasting.py
from __future__ import annotations
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet

DATE_HINTS = ("date", "time", "timestamp", "data", "czas", "dt")

# ----------------------------
# Utils: czas i częstotliwość
# ----------------------------
def _detect_datetime_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """Ustawia indeks czasowy. Najpierw używa `date_col` jeśli podano, w przeciwnym razie heurystyka po nazwach."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    cand = None
    if date_col and date_col in out.columns:
        cand = date_col
    else:
        # heurystyka po nazwach
        for c in out.columns:
            lc = str(c).lower()
            if any(h in lc for h in DATE_HINTS):
                cand = c
                break

    if cand is not None:
        try:
            out[cand] = pd.to_datetime(out[cand], errors="coerce", infer_datetime_format=True)
            out = out.set_index(cand).sort_index()
            return out
        except Exception:
            pass

    # last resort – jeżeli pierwsza kolumna wygląda na datę
    first = out.columns[0]
    try:
        out[first] = pd.to_datetime(out[first], errors="coerce", infer_datetime_format=True)
        if out[first].notna().mean() > 0.6:
            out = out.set_index(first).sort_index()
            return out
    except Exception:
        pass

    # nie udało się – zwróć oryginał (Prophet i tak wymaga kolumny ds)
    return out

def _infer_freq(idx: pd.DatetimeIndex) -> str:
    """Zgadnij częstotliwość z indeksu; fallback do 'D'."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return "D"
    f = pd.infer_freq(idx)
    if f:
        # normalizuj do aliasów akceptowanych przez Prophet
        f = f.upper()
        if f.startswith("A"): return "Y"
        if f.startswith("Y"): return "Y"
        if f.startswith("Q"): return "Q"
        if f.startswith("M"): return "M"
        if f.startswith("W"): return "W"
        if f.startswith("D"): return "D"
        if f.startswith("H"): return "H"
        if f.startswith("T"): return "min"  # minute
        if f.startswith("S"): return "S"
        return "D"

    # heurystyka po medianie różnic
    diffs = np.diff(idx.view("i8"))  # ns
    if len(diffs) == 0:
        return "D"
    med = np.median(diffs) / 1e9  # sekundy
    day = 86400
    if med < 60: return "S"
    if med < 3600: return "min"
    if med < day: return "H"
    if med < 7 * day: return "D"
    if med < 28 * day: return "W"
    if med < 92 * day: return "M"
    if med < 366 * day: return "Q"
    return "Y"

def _boolean_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    """Określa, które sezonowości Propheta mają sens przy danej częstotliwości."""
    f = (freq or "D").upper()
    yearly = True
    weekly = f in ("D", "H", "S", "MIN")
    daily = f in ("H", "S", "MIN")
    return yearly, weekly, daily

def _prepare_y(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    return y

def _pick_exogenous(df: pd.DataFrame, target: str, limit: int = 15) -> List[str]:
    """Wybór kandydatów na regresory zewnętrzne (numeryczne, co najmniej 3 unikalne)."""
    cols = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) >= 3:
            cols.append(c)
    # prosta heurystyka: wybierz do `limit` najbardziej zmiennych
    cols = sorted(cols, key=lambda c: float(df[c].std(skipna=True) or 0), reverse=True)[:limit]
    return cols

# --------------------------------
# Główny interfejs: forecast()
# --------------------------------
def forecast(
    df: pd.DataFrame,
    target: str,
    horizon: int = 12,
    *,
    date_col: Optional[str] = None,
    seasonality_mode: str = "additive",            # "additive" | "multiplicative"
    changepoint_prior_scale: float = 0.1,          # większa → bardziej czuły na zmiany trendu
    extra_regressors: Union[None, str, List[str]] = None,  # None | "auto" | ["col1", ...]
    holidays: Optional[pd.DataFrame] = None,       # opcjonalna tabela świąt (ds, holiday)
    growth: str = "linear",                        # "linear" | "logistic"
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    freq: Optional[str] = None,                    # wymuś częstotliwość (np. "MS")
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Trenuje model Prophet i zwraca (model, forecast_df).
    forecast_df: kolumny `ds`, `yhat`, `yhat_lower`, `yhat_upper` (+ ewentualne regresory).
    """
    assert isinstance(df, pd.DataFrame) and target in df.columns, "Nieprawidłowe dane lub brak kolumny celu."

    # 1) Indeks czasu + sortowanie
    df_idx = _detect_datetime_index(df, date_col=date_col)
    if not isinstance(df_idx.index, pd.DatetimeIndex):
        # jeśli nadal brak indeksu czasu – spróbuj wymusić z targetem jako seria
        raise ValueError("Nie wykryto prawidłowej kolumny czasu. Podaj `date_col` lub dodaj kolumnę daty.")
    df_idx = df_idx.sort_index()

    # 2) Przygotowanie ramki 'ds', 'y' (+ regresory)
    y = _prepare_y(df_idx[target])
    data = pd.DataFrame({"ds": df_idx.index, "y": y})
    # usuń wiersze bez y
    data = data.dropna(subset=["y"])

    # growth logistic?
    if growth == "logistic":
        if cap is None:
            cap = float(np.nanmax(y) * 1.2) if len(y) else 1.0
        if floor is None:
            floor = float(np.nanmin(y) * 0.8) if len(y) else 0.0
        data["cap"] = cap
        data["floor"] = floor

    # regresory zewnętrzne
    chosen_reg: List[str] = []
    if isinstance(extra_regressors, list):
        chosen_reg = [c for c in extra_regressors if c in df_idx.columns and c != target]
    elif isinstance(extra_regressors, str) and extra_regressors.lower() == "auto":
        chosen_reg = _pick_exogenous(df_idx, target)
    # dołącz dane regresorów
    for c in chosen_reg:
        data[c] = pd.to_numeric(df_idx[c], errors="coerce")

    # 3) Konfiguracja modelu Prophet
    _freq = (freq or _infer_freq(df_idx.index)).upper()
    yearly, weekly, daily = _boolean_seasonality_flags(_freq)

    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        growth=growth,
        holidays=holidays,  # None jeśli nie podano
    )

    # Dodatkowe sezonowości (miesięczna/kwartalna) – przy danych miesięcznych/kwartalnych
    if _freq in ("M", "MS"):
        model.add_seasonality(name="monthly", period=12, fourier_order=8)
    if _freq in ("Q", "QS"):
        model.add_seasonality(name="quarterly", period=4, fourier_order=4)

    # Zarejestruj regresory
    for c in chosen_reg:
        model.add_regressor(c, standardize="auto")

    # 4) Trening
    model.fit(data)

    # 5) Future DF
    # Mapowanie aliasów na coś, co Prophet lubi
    freq_alias = {"Y": "Y", "A": "Y", "Q": "Q", "QS": "QS", "M": "MS", "MS": "MS",
                  "W": "W", "D": "D", "H": "H", "MIN": "min", "S": "S"}.get(_freq, "D")

    future = model.make_future_dataframe(periods=int(horizon), freq=freq_alias, include_history=True)

    # growth logistic: uzupełnij cap/floor
    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = floor

    # Dołącz przyszłe wartości regresorów:
    # jeśli ich nie mamy — forward-fill ostatnią znaną wartością (prostolinijne założenie).
    if chosen_reg:
        # oryginalne dane w indeksie „ds”
        aux = data.set_index("ds")[chosen_reg]
        future = future.set_index("ds")
        for c in chosen_reg:
            future[c] = aux[c].reindex(future.index).ffill()
        future = future.reset_index()

    # 6) Prognoza
    fcst = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # 7) Metadane w atrybutach
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
    }

    return model, fcst
