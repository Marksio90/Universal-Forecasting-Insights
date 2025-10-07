# feature_engineering.py — TURBO PRO
from __future__ import annotations

import re
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


# =========================
# Konfiguracja / typy
# =========================
@dataclass(frozen=True)
class FEOptions:
    # detekcja dat
    date_name_hints: Tuple[str, ...] = ("date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year")
    date_sample_size: int = 200
    date_hit_threshold: float = 0.60       # min. odsetek trafień wzorców by uznać kolumnę za datę
    min_parsed_ratio: float = 0.50         # min. % parsowalnych wartości po to_datetime

    # cechy dat
    add_age: bool = True                   # dodaj age (lata do teraz) tylko jeśli większość w przeszłości
    age_round_decimals: int = 1

    # tekst
    text_min_unique: int = 5               # omijaj kolumny z bardzo małą unikalnością
    text_max_unique: int = 5_000           # i bardzo dużą (typowy free text)
    text_len_dtype: str = "Int32"
    text_words_dtype: str = "Int16"

    # kategorie
    cat_low_card_max: int = 10             # one-hot dla <= 10 unikalnych
    cat_mid_card_max: int = 30             # ordinal dla (10, 30]
    one_hot_drop_first: bool = False
    one_hot_dummy_na: bool = False
    ordinal_dtype: str = "Int32"
    one_hot_prefix_sep: str = "_"

    # inne
    replace_inf_with_nan: bool = True
    cap_new_columns: int = 5_000           # bezpieczeństwo przed eksplozją kolumn


@dataclass(frozen=True)
class FEAction:
    kind: Literal["date", "text", "categorical", "repair"]
    column: Optional[str]
    produced: List[str]
    note: Optional[str] = None


@dataclass(frozen=True)
class FEResult:
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    actions: List[FEAction]
    warnings: List[str]


# =========================
# Heurystyki dat
# =========================
_DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",            # 2024-09-01 / 2024/9/1
    r"^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$",      # 01/09/2024, 1.9.24
    r"^\d{4}[-/]\d{1,2}$",                       # 2024-09
    r"^\d{8}$",                                  # 20240901
    r"^\d{14}$",                                 # 20240901123045
    r"^\d{10}$",                                 # unix s
    r"^\d{13}$",                                 # unix ms
    r"^\d{4}-W\d{2}-\d$",                        # ISO week-date
]

def _is_date_series(s: pd.Series, *, sample_size: int, hit_threshold: float) -> bool:
    if not (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(max(1, sample_size))
    if sample.empty:
        return False
    hits = 0
    for pat in _DATE_PATTERNS:
        hits = max(hits, int(sample.str.match(pat).sum()))
        if hits / len(sample) >= hit_threshold:
            return True
    return False

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=False)
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT] * len(series)))


# =========================
# Public API — wstecznie kompatybilne
# =========================
def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Back-compat: generuje cechy w miejscu nowego silnika,
    dokleja raport jako `df.fe_report`.
    """
    df2, rep = basic_feature_engineering_pro(df)
    try:
        setattr(df2, "fe_report", {
            "dates": [a.column for a in rep.actions if a.kind == "date"],
            "text": [a.column for a in rep.actions if a.kind == "text"],
            "categoricals": [a.column for a in rep.actions if a.kind == "categorical"],
            "warnings": rep.warnings,
        })  # type: ignore[attr-defined]
    except Exception:
        pass
    return df2


# =========================
# Nowe API — bogatszy raport
# =========================
def basic_feature_engineering_pro(df: pd.DataFrame, opts: FEOptions = FEOptions()) -> tuple[pd.DataFrame, FEResult]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("basic_feature_engineering_pro: brak danych lub pusty DataFrame.")

    out = df.copy(deep=True)
    actions: List[FEAction] = []
    warnings: List[str] = []
    new_cols_count = 0
    input_shape = out.shape

    # 1) DATY/CZAS
    for c in list(out.columns):
        lc = str(c).lower()
        looks_like_name = any(k in lc for k in opts.date_name_hints)
        looks_like_regex = _is_date_series(out[c], sample_size=opts.date_sample_size, hit_threshold=opts.date_hit_threshold)
        if not (looks_like_name or looks_like_regex):
            continue

        dt = _safe_to_datetime(out[c])
        parsed_ratio = float(dt.notna().mean())
        if parsed_ratio < opts.min_parsed_ratio:
            continue

        produced: List[str] = []
        # podstawowe cechy
        out[f"{c}_year"] = dt.dt.year.astype("Int16")
        out[f"{c}_month"] = dt.dt.month.astype("Int8")
        out[f"{c}_day"] = dt.dt.day.astype("Int8")
        out[f"{c}_dow"] = dt.dt.dayofweek.astype("Int8")
        out[f"{c}_quarter"] = dt.dt.quarter.astype("Int8")
        # ISO week → zachowaj jako Int16 (UInt w nowych pandas)
        iso_week = dt.dt.isocalendar().week.astype("UInt16")
        out[f"{c}_iso_week"] = iso_week

        # weekend
        out[f"{c}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("Int8")

        produced += [f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dow", f"{c}_quarter", f"{c}_iso_week", f"{c}_is_weekend"]

        # age (tylko gdy większość dat w przeszłości)
        if opts.add_age:
            try:
                delta_days = (pd.Timestamp.now(tz=dt.dt.tz) - dt).dt.days
                if (delta_days > 0).mean() >= 0.6:
                    age_years = (delta_days / 365.25).round(opts.age_round_decimals).astype("Float32")
                    out[f"{c}_age"] = age_years
                    produced.append(f"{c}_age")
            except Exception:
                warnings.append(f"age: nie udało się policzyć dla '{c}'.")

        actions.append(FEAction(kind="date", column=c, produced=produced))

        new_cols_count += len(produced)
        if new_cols_count >= opts.cap_new_columns:
            warnings.append("Przekroczono limit nowych kolumn — dalsze cechy zostały ucięte.")
            break

    # 2) TEKST
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        s = out[c].astype("string")
        nunique = int(s.nunique(dropna=True))
        if not (opts.text_min_unique <= nunique <= opts.text_max_unique):
            continue

        produced: List[str] = []
        out[f"{c}_len"] = s.str.len().fillna(0).astype(opts.text_len_dtype)
        out[f"{c}_n_words"] = s.str.split().str.len().fillna(0).astype(opts.text_words_dtype)
        out[f"{c}_has_digits"] = s.str.contains(r"\d", regex=True, na=False).astype("Int8")
        produced += [f"{c}_len", f"{c}_n_words", f"{c}_has_digits"]

        actions.append(FEAction(kind="text", column=c, produced=produced))
        new_cols_count += len(produced)
        if new_cols_count >= opts.cap_new_columns:
            warnings.append("Przekroczono limit nowych kolumn — dalsze cechy zostały ucięte.")
            break

    # 3) KATEGORIE o niskiej krotności
    # robimy pętlę po obiektowych/stringowych po TEKŚCIE; część mogła zostać wzbogacona dodatkowymi kolumnami
    cat_candidates = out.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cat_candidates:
        nunique = int(out[c].nunique(dropna=True))
        if nunique <= 1:
            continue
        if nunique <= opts.cat_low_card_max:
            # one-hot
            dummies = pd.get_dummies(
                out[c],
                prefix=c,
                prefix_sep=opts.one_hot_prefix_sep,
                drop_first=opts.one_hot_drop_first,
                dummy_na=opts.one_hot_dummy_na,
            )
            out = pd.concat([out.drop(columns=[c]), dummies], axis=1)
            actions.append(FEAction(kind="categorical", column=c, produced=dummies.columns.tolist(), note="one_hot"))
            new_cols_count += dummies.shape[1]
        elif nunique <= opts.cat_mid_card_max:
            # ordinal
            out[c] = out[c].astype("category").cat.codes.astype(opts.ordinal_dtype)
            actions.append(FEAction(kind="categorical", column=c, produced=[c], note="ordinal"))
        else:
            # > mid_card_max — pomijamy
            continue

        if new_cols_count >= opts.cap_new_columns:
            warnings.append("Przekroczono limit nowych kolumn — dalsze cechy zostały ucięte.")
            break

    # 4) Naprawa inf/NaN
    if opts.replace_inf_with_nan:
        out = out.replace([np.inf, -np.inf], np.nan)
        actions.append(FEAction(kind="repair", column=None, produced=[], note="replaced inf with NaN"))

    result = FEResult(
        input_shape=input_shape,
        output_shape=out.shape,
        actions=actions,
        warnings=warnings,
    )

    # dopnij skrócony raport dla kompatybilności z Twoją appką
    try:
        setattr(out, "fe_report", {
            "dates": [a.column for a in actions if a.kind == "date"],
            "text": [a.column for a in actions if a.kind == "text"],
            "categoricals": [
                {"col": a.column, "type": (a.note or ""), "n_new": len(a.produced)}
                for a in actions if a.kind == "categorical"
            ],
            "warnings": warnings,
        })  # type: ignore[attr-defined]
    except Exception:
        pass

    return out, result
