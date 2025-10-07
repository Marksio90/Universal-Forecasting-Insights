from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List

DATE_LIKE = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")

def _is_date_series(s: pd.Series, sample_size: int = 200) -> bool:
    """Heurystycznie sprawdza, czy kolumna wygląda na datę."""
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    date_like = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    return date_like.mean() > 0.6

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Konwertuje kolumnę na datetime, zwracając pd.NaT przy błędach."""
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(series)))

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatyczne cechy:
      - daty → rok, miesiąc, dzień, kwartał, tydzień, weekend, age
      - tekst → długość, liczba słów, has_digits
      - kategorie niskiej krotności → one-hot / ordinal
    """
    out = df.copy(deep=True)
    fe_report: Dict[str, Any] = {"dates": [], "text": [], "categoricals": []}

    # -------------------------
    # 1️⃣ Cecha daty/czasu
    # -------------------------
    for c in list(out.columns):
        lc = str(c).lower()
        looks_like_date = any(k in lc for k in DATE_LIKE) or _is_date_series(out[c])
        if not looks_like_date:
            continue

        dt = _safe_to_datetime(out[c])
        if dt.notna().sum() < 0.5 * len(dt):
            continue

        # generowanie cech daty
        out[f"{c}_year"] = dt.dt.year
        out[f"{c}_month"] = dt.dt.month
        out[f"{c}_day"] = dt.dt.day
        out[f"{c}_dow"] = dt.dt.dayofweek
        out[f"{c}_quarter"] = dt.dt.quarter
        out[f"{c}_week"] = dt.dt.isocalendar().week.astype("Int16")
        out[f"{c}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("Int8")

        # jeśli data w przeszłości → "age"
        try:
            delta_years = (pd.Timestamp.now() - dt).dt.days / 365.25
            if delta_years.notna().sum() > 0.6 * len(delta_years):
                out[f"{c}_age"] = delta_years.round(1)
        except Exception:
            pass

        fe_report["dates"].append(c)

    # -------------------------
    # 2️⃣ Cecha tekstowa
    # -------------------------
    for c in out.select_dtypes(include="object").columns:
        s = out[c].astype(str)
        # jeśli zbyt dużo unikalnych (prawdopodobnie tekst opisowy)
        nunique = s.nunique(dropna=True)
        if nunique < 5 or nunique > 5000:
            continue

        out[f"{c}_len"] = s.str.len().fillna(0).astype("Int32")
        out[f"{c}_n_words"] = s.str.split().str.len().fillna(0).astype("Int16")
        out[f"{c}_has_digits"] = s.str.contains(r"\d", regex=True, na=False).astype("Int8")
        fe_report["text"].append(c)

    # -------------------------
    # 3️⃣ Kategorie niskiej krotności
    # -------------------------
    for c in out.select_dtypes(include="object").columns:
        nunique = out[c].nunique(dropna=True)
        if nunique <= 1:
            continue
        elif nunique <= 10:
            # one-hot
            dummies = pd.get_dummies(out[c], prefix=c, dummy_na=False)
            out = pd.concat([out.drop(columns=[c]), dummies], axis=1)
            fe_report["categoricals"].append({"col": c, "type": "one_hot", "nunique": int(nunique)})
        elif nunique <= 30:
            # ordinal encoding
            out[c] = out[c].astype("category").cat.codes
            fe_report["categoricals"].append({"col": c, "type": "ordinal", "nunique": int(nunique)})
        # >30 → pomijamy (zbyt wysoka krotność)

    # -------------------------
    # 4️⃣ Naprawa inf/NaN
    # -------------------------
    out = out.replace([np.inf, -np.inf], np.nan)

    # -------------------------
    # Raport
    # -------------------------
    out.fe_report = fe_report  # type: ignore[attr-defined]
    return out
