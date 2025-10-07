from __future__ import annotations
import re
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ..utils.helpers import smart_cast_numeric  # zakładam, że już masz


# =========================
# Dataclasses
# =========================
@dataclass(frozen=True)
class CleanOptions:
    sample_size_dates: int = 200            # ile wartości do próby detekcji dat
    date_hit_threshold: float = 0.65        # jaki % trafień wymagamy, by uznać kolumnę za datę
    detect_binary: bool = True
    binary_bool_preferred: bool = True      # True -> pandas.BooleanDtype(), False -> Int8 dla 0/1
    cast_numeric: bool = True
    trim_strings: bool = True
    drop_duplicates: bool = True
    replace_inf_with_nan: bool = True
    impute_missing: bool = True
    impute_numeric_strategy: str = "median" # na razie: median
    impute_object_token: str = "<missing>"
    impute_categorical_token: str = "<unknown>"
    impute_bool_value: bool = False


@dataclass(frozen=True)
class CleanReport:
    shape_initial: Tuple[int, int]
    shape_final: Tuple[int, int]
    duplicates_removed: int
    date_columns_detected: List[str]
    imputed_columns: Dict[str, str]
    na_total_after: int
    notes: List[str]


# =========================
# Date helpers
# =========================
_DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",         # 2024-09-01 / 2024/9/1
    r"^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$",    # 01/09/2024, 1.9.24
    r"^\d{4}[-/]\d{1,2}$",                     # 2024-09 (rok-miesiąc)
    r"^\d{8}$",                                # 20240901
    r"^\d{14}$",                               # 20240901123045
    r"^\d{10}$",                               # sekundowy timestamp
    r"^\d{13}$",                               # ms timestamp
    r"^\d{4}-W\d{2}-\d$",                      # ISO week-date 2024-W35-1
]

def _detect_date_columns(df: pd.DataFrame, sample_size: int = 100, hit_threshold: float = 0.7) -> list[str]:
    """Wykrywa kolumny, które prawdopodobnie zawierają daty (kilka wzorców + timestampy)."""
    if df.empty:
        return []
    candidates: List[str] = []
    obj_like = list(df.select_dtypes(include=["object", "string"]).columns)
    for c in obj_like:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        sample = s.head(max(1, sample_size))
        hits = 0
        for pat in _DATE_PATTERNS:
            hits = max(hits, int(sample.str.match(pat).sum()))
            if hits / len(sample) >= hit_threshold:
                candidates.append(c)
                break
    return candidates


# =========================
# Binary helpers
# =========================
_TRUE_SET = {"1", "true", "t", "y", "yes", "tak"}
_FALSE_SET = {"0", "false", "f", "n", "no", "nie"}

def _convert_binaries(df: pd.DataFrame, prefer_boolean: bool = True) -> pd.DataFrame:
    """Konwertuje kolumny binarne do BooleanDtype lub Int8."""
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if ptypes.is_bool_dtype(s) or ptypes.is_integer_dtype(s) or ptypes.is_float_dtype(s):
            # jeżeli to już numeric/bool — pomiń (chyba że to float 0.0/1.0 — nie ruszamy tutaj)
            continue
        if ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
            lower = s.astype(str).str.strip().str.lower()
            uniq = set(lower.dropna().unique().tolist())
            # czysty 0/1?
            if uniq and uniq <= {"0", "1"}:
                if prefer_boolean:
                    out[c] = lower.map({"0": False, "1": True}).astype("boolean")
                else:
                    out[c] = lower.map({"0": 0, "1": 1}).astype("Int8")
                continue
            # zestaw {true_set} ∪ {false_set}?
            if uniq and uniq <= (_TRUE_SET | _FALSE_SET):
                out[c] = lower.map(lambda v: (True if v in _TRUE_SET else (False if v in _FALSE_SET else pd.NA))).astype("boolean")
    return out


# =========================
# Public API
# =========================
def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inteligentne czyszczenie danych (zachowuje podpis funkcji z Twojego kodu).
    Raport dostępny jako `df.clean_report` po zwróceniu.
    """
    df_clean, report = quick_clean_pro(df)
    # bezpiecznie dołącz atrybut raportu
    try:
        setattr(df_clean, "clean_report", asdict(report))  # type: ignore[attr-defined]
    except Exception:
        pass
    return df_clean


def quick_clean_pro(df: pd.DataFrame, opts: CleanOptions = CleanOptions()) -> tuple[pd.DataFrame, CleanReport]:
    """
    Wersja PRO: zwraca (DataFrame, CleanReport).
    Nie modyfikuje wejściowego df (kopiuje).
    """
    notes: List[str] = []
    if not isinstance(df, pd.DataFrame):
        raise TypeError("quick_clean_pro: df must be a pandas.DataFrame")

    df2 = df.copy(deep=True)
    shape_initial = df2.shape

    # 1) Drop duplicates
    duplicates_removed = 0
    if opts.drop_duplicates:
        duplicates_removed = int(df2.duplicated().sum())
        if duplicates_removed > 0:
            df2 = df2.drop_duplicates()
            notes.append(f"Removed {duplicates_removed} duplicate rows.")

    # 2) Trim strings / normalize empties
    if opts.trim_strings:
        obj_cols = df2.select_dtypes(include=["object", "string"]).columns
        for c in obj_cols:
            s = df2[c].astype("string")
            s = s.str.strip()
            s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA}, regex=False)
            df2[c] = s

    # 3) Cast numeric
    if opts.cast_numeric:
        try:
            df2 = smart_cast_numeric(df2)
        except Exception:
            notes.append("smart_cast_numeric failed; skipped numeric casting.")

    # 4) Dates detection & casting
    date_cols = _detect_date_columns(df2, sample_size=opts.sample_size_dates, hit_threshold=opts.date_hit_threshold)
    for c in date_cols:
        try:
            # infer_datetime_format jest deprecated; pandas sam radzi sobie dobrze z parsowaniem
            df2[c] = pd.to_datetime(df2[c], errors="coerce", utc=False)
        except Exception:
            notes.append(f"Failed to cast '{c}' to datetime; left as-is.")

    # 5) Binary conversion
    if opts.detect_binary:
        df2 = _convert_binaries(df2, prefer_boolean=opts.binary_bool_preferred)

    # 6) Inf -> NaN
    if opts.replace_inf_with_nan:
        df2 = df2.replace([np.inf, -np.inf], pd.NA)

    # 7) Imputation
    imputed: Dict[str, str] = {}
    if opts.impute_missing:
        for c in df2.columns:
            s = df2[c]
            if not s.isna().any():
                continue

            if ptypes.is_numeric_dtype(s):
                if opts.impute_numeric_strategy == "median":
                    val = s.median(skipna=True)
                else:
                    val = s.median(skipna=True)
                df2[c] = s.fillna(val)
                imputed[c] = f"median={val}"
            elif ptypes.is_categorical_dtype(s):
                token = opts.impute_categorical_token
                # upewnij się, że token jest w kategoriach
                if token not in getattr(s, "cat", s).categories:
                    df2[c] = s.cat.add_categories([token])
                df2[c] = df2[c].fillna(token)
                imputed[c] = token
            elif ptypes.is_bool_dtype(s):
                df2[c] = s.fillna(opts.impute_bool_value)
                imputed[c] = str(opts.impute_bool_value)
            elif ptypes.is_string_dtype(s) or ptypes.is_object_dtype(s):
                token = opts.impute_object_token
                df2[c] = s.fillna(token)
                imputed[c] = token
            else:
                # inne typy – zostawiamy NaN
                pass

    # 8) Final stats
    shape_final = df2.shape
    na_total_after = int(df2.isna().sum().sum())

    report = CleanReport(
        shape_initial=shape_initial,
        shape_final=shape_final,
        duplicates_removed=duplicates_removed,
        date_columns_detected=date_cols,
        imputed_columns=imputed,
        na_total_after=na_total_after,
        notes=notes,
    )

    # doczep raport jako atrybut (dla kompatybilności z Twoją appką)
    try:
        setattr(df2, "clean_report", asdict(report))  # type: ignore[attr-defined]
    except Exception:
        pass

    return df2, report
