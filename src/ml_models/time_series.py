from __future__ import annotations
from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from prophet import Prophet
from ..utils.helpers import ensure_datetime_index  # Twoja utilka (zostawiamy); mamy też fallback

DATE_HINTS = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")


def _detect_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback, gdy ensure_datetime_index nie ustawi DatetimeIndex."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    # heurystyka po nazwach
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

    # ostatnia próba – pierwsza kolumna
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
        if f.startswith("A") or f.startswith("Y"): return "Y"
        if f.startswith("Q"): return "Q"
        if f.startswith("M"): return "M"  # później zamienimy na "MS"
        if f.startswith("W"): return "W"
        if f.startswith("D"): return "D"
        if f.startswith("H"): return "H"
        if f.startswith("T"): return "MIN"
        if f.startswith("S"): return "S"
    # prosta heurystyka po medianie różnic
    diffs = np.diff(idx.view("i8"))  # ns
    if len(diffs) == 0:
        return "D"
    med = float(np.median(diffs) / 1e9)  # sekundy
    day = 86400.0
    if med < 60: return "S"
    if med < 3600: return "H"
    if med < day: return "D"
    if med < 7 * day: return "W"
    if med < 28 * day: return "M"
    if med < 92 * day: return "Q"
    return "Y"


def _boolean_seasonality_flags(freq: str) -> Tuple[bool, bool, bool]:
    f = (freq or "D").upper()
    yearly = True
    weekly = f in ("D", "H", "S", "MIN")
    daily = f in ("H", "S", "MIN")
    return yearly, weekly, daily


def _prepare_y(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    return y


def _pick_exogenous(df: pd.DataFrame, target: str, limit: int = 15) -> List[str]:
    """Wybierz do `limit` numerycznych, zmiennych regresorów (na bazie odchylenia)."""
    cols = []
    for c in df.columns:
        if c == target:
            continue
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser) and ser.nunique(dropna=True) >= 3:
            cols.append(c)
    cols = sorted(cols, key=lambda c: float(df[c].std(skipna=True) or 0.0), reverse=True)[:limit]
    return cols


def fit_prophet(
    df: pd.DataFrame,
    target: str,
    horizon: int = 12,
    **options: Any,
):
    """
    Trenuje model Prophet i zwraca (model, fcst) dla najbliższych `horizon` okresów.
    ZACHOWUJE kompatybilność z poprzednią implementacją (ds, yhat, yhat_lower, yhat_upper).

    Opcjonalne parametry (przez **options):
      - seasonality_mode: "additive" | "multiplicative" (domyślnie "additive")
      - changepoint_prior_scale: float (domyślnie 0.1)
      - growth: "linear" | "logistic" (domyślnie "linear")
      - cap, floor: float (dla growth="logistic"; auto jeśli brak)
      - extra_regressors: None | "auto" | List[str]
      - holidays: DataFrame z kolumnami ["ds","holiday"] (opcjonalnie)
      - include_history: bool (domyślnie False — jak wcześniej)
      - freq: str (np. "MS" aby wymusić miesiące od początku miesiąca)
      - outlier_clip: None | "iqr" | "zscore" (proste obcinanie odstających punktów na y)
    """
    if target not in df.columns:
        raise ValueError("Brak kolumny celu w danych.")

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

    df2 = df2.sort_index()

    # 2) Przygotowanie y (+ opcjonalne obcięcie outlierów)
    y = _prepare_y(df2[target])
    outlier_clip = (options.get("outlier_clip") or "").lower() if options.get("outlier_clip") else None
    if outlier_clip == "iqr":
        q1, q3 = y.quantile(0.25), y.quantile(0.75)
        iqr = float(q3 - q1)
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        y = y.clip(lower=lo, upper=hi)
    elif outlier_clip == "zscore":
        m, s = y.mean(skipna=True), y.std(skipna=True) or 1.0
        z = (y - m) / s
        y = y.mask(z.abs() > 4.0, np.nan)  # odetnij ekstremalne; potem dropna

    data = pd.DataFrame({"ds": df2.index, "y": y}).dropna(subset=["y"])

    # 3) Parametry modelu
    seasonality_mode = options.get("seasonality_mode", "additive")
    cps = float(options.get("changepoint_prior_scale", 0.1))
    growth = options.get("growth", "linear")
    cap = options.get("cap")
    floor = options.get("floor")
    holidays = options.get("holidays")
    include_history = bool(options.get("include_history", False))

    # wzrost logistyczny – uzupełnij cap/floor automatycznie jeśli brak
    if growth == "logistic":
        if cap is None:
            cap = float(np.nanmax(data["y"]) * 1.2) if len(data) else 1.0
        if floor is None:
            floor = float(np.nanmin(data["y"]) * 0.8) if len(data) else 0.0
        data["cap"] = cap
        data["floor"] = floor

    # 4) Regresory zewnętrzne
    extra_reg = options.get("extra_regressors")
    chosen_reg: List[str] = []
    if isinstance(extra_reg, list):
        chosen_reg = [c for c in extra_reg if c in df2.columns and c != target]
    elif isinstance(extra_reg, str) and extra_reg.lower() == "auto":
        chosen_reg = _pick_exogenous(df2, target)

    for c in chosen_reg:
        data[c] = pd.to_numeric(df2[c], errors="coerce").reindex(data["ds"]).values

    # 5) Częstotliwość i sezonowości
    freq_in = (options.get("freq") or _infer_freq(df2.index)).upper()
    yearly, weekly, daily = _boolean_seasonality_flags(freq_in)

    m = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        growth=growth,
        holidays=holidays,
    )

    # Dodatkowe sezonowości – przy miesiącach/kwartałach
    if freq_in in ("M", "MS"):
        m.add_seasonality(name="monthly", period=12, fourier_order=8)
    if freq_in in ("Q", "QS"):
        m.add_seasonality(name="quarterly", period=4, fourier_order=4)

    # Regressory
    for c in chosen_reg:
        m.add_regressor(c, standardize="auto")

    # 6) Trening
    m.fit(data)

    # 7) Przyszłość
    freq_alias = {"Y": "Y", "A": "Y", "Q": "Q", "QS": "QS", "M": "MS", "MS": "MS",
                  "W": "W", "D": "D", "H": "H", "MIN": "min", "S": "S"}.get(freq_in, "D")
    future = m.make_future_dataframe(periods=int(horizon), freq=freq_alias, include_history=include_history)

    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = floor

    if chosen_reg:
        aux = data.set_index("ds")[chosen_reg]
        future = future.set_index("ds")
        for c in chosen_reg:
            future[c] = aux[c].reindex(future.index).ffill()
        future = future.reset_index()

    # 8) Prognoza
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # 9) Metadane
    fcst.attrs["forecast_meta"] = {
        "target": target,
        "n_obs": int(len(data)),
        "freq": freq_alias,
        "horizon": int(horizon),
        "seasonality_mode": seasonality_mode,
        "changepoint_prior_scale": cps,
        "extra_regressors": chosen_reg or None,
        "growth": growth,
        "cap": cap if growth == "logistic" else None,
        "floor": floor if growth == "logistic" else None,
        "include_history": include_history,
    }

    return m, fcst
