# src/utils/validators.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _dtype_group(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_categorical_dtype(s):
        return "category"
    return "object"


def _top_missing_cols(df: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
    if df.empty:
        return {}
    pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    pct = pct.round(2).head(top_n)
    return {str(k): _safe_float(v) for k, v in pct.to_dict().items() if v > 0}


def _cardinality_buckets(df: pd.DataFrame) -> Dict[str, List[str]]:
    lows, highs, binaries = [], [], []
    for c in df.select_dtypes(include=["object", "category"]).columns:
        nun = int(df[c].nunique(dropna=True))
        if nun == 2:
            binaries.append(c)
        elif 3 <= nun <= 10:
            lows.append(c)
        elif nun > 30:
            highs.append(c)
    return {"low_cardinality": lows, "high_cardinality": highs, "binary_like": binaries}


def _numeric_red_flags(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return {"non_finite_pct": 0.0, "skewed_cols": [], "high_kurtosis_cols": []}
    # udział niefinity (inf/-inf/NaN) względem wszystkich komórek numerycznych
    non_finite = (~np.isfinite(num.to_numpy(dtype=float))).mean()
    # skew/kurtosis – top 5 po wartości bezwzględnej
    try:
        sk = num.skew(numeric_only=True).abs().sort_values(ascending=False).dropna().head(5)
        hk = num.kurtosis(numeric_only=True).abs().sort_values(ascending=False).dropna().head(5)
        skewed = [str(c) for c in sk.index if np.isfinite(sk[c])]
        kurt = [str(c) for c in hk.index if np.isfinite(hk[c])]
    except Exception:
        skewed, kurt = [], []
    return {
        "non_finite_pct": round(float(non_finite * 100.0), 3),
        "skewed_cols": skewed,
        "high_kurtosis_cols": kurt,
    }


def basic_quality_checks(df: pd.DataFrame) -> dict:
    """
    Rozszerzony, szybki raport jakości danych.
    Zwraca m.in.:
      - rows, cols, missing_pct, dupes (+ dupes_pct)
      - memory_mb, dtypes_summary, n_numeric/n_object/n_category/n_bool/n_datetime
      - avg_nulls_per_col, avg_nulls_per_row_pct
      - top_missing_cols (Top 10 z % braków)
      - constant_columns (zerowa wariancja)
      - cardinality (low/high/binary dla kategorycznych)
      - numeric.red_flags (non_finite_pct, skewed_cols, high_kurtosis_cols)
      - sample_duplicate_indices (do 10)
    Wszystkie wartości są JSON-serializowalne.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Brak danych"}
    rows = int(len(df))
    cols = int(df.shape[1])
    if rows == 0:
        return {"rows": 0, "cols": cols, "missing_pct": 0.0, "dupes": 0}

    # Podstawowe KPI
    total_cells = max(1, df.size)
    missing_all = int(df.isna().sum().sum())
    missing_pct = float(missing_all / total_cells)
    dupes = int(df.duplicated().sum())
    dupes_pct = float(dupes / rows) if rows else 0.0

    # Pamięć
    try:
        memory_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    except Exception:
        memory_mb = float(df.memory_usage().sum() / (1024 ** 2))

    # Dtypes (zagregowane)
    groups = [_dtype_group(df[c]) for c in df.columns]
    dtypes_summary = {g: groups.count(g) for g in set(groups)}
    n_numeric = int(dtypes_summary.get("numeric", 0))
    n_object = int(dtypes_summary.get("object", 0))
    n_category = int(dtypes_summary.get("category", 0))
    n_bool = int(dtypes_summary.get("bool", 0))
    n_datetime = int(dtypes_summary.get("datetime", 0))

    # Braki: średnio na kolumnę / wiersz
    avg_nulls_per_col = float(df.isna().sum().mean())
    avg_nulls_per_row_pct = float(df.isna().mean(axis=1).mean())  # odsetek braków w wierszu (średnio)

    # Top braków per kolumna
    top_missing_cols = _top_missing_cols(df, top_n=10)

    # Kolumny stałe
    try:
        constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    except Exception:
        constant_columns = []

    # Kardynalność kategorycznych
    cardinality = _cardinality_buckets(df)

    # Flagi numeryczne (niefinity, skośności)
    numeric_flags = _numeric_red_flags(df)

    # Próbka duplikatów (indeksy)
    try:
        dup_idx = df.index[df.duplicated()].tolist()[:10]
        if not isinstance(dup_idx, list):
            dup_idx = list(dup_idx)
        sample_duplicate_indices = [str(i) for i in dup_idx]
    except Exception:
        sample_duplicate_indices = []

    # Ostrzeżenia tekstowe (ułatwia UI/raport)
    warnings: List[str] = []
    if missing_pct > 0.2:
        warnings.append("Wysoki odsetek braków (>20%).")
    if dupes_pct > 0.01:
        warnings.append("Zauważalna liczba duplikatów (>1%).")
    if constant_columns:
        warnings.append(f"Kolumny o zerowej wariancji: {min(len(constant_columns), 5)}+.")
    if numeric_flags.get("non_finite_pct", 0.0) > 0.1:
        warnings.append("Wartości niefinity w cechach numerycznych (>0.1%).")

    report: Dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "missing_pct": round(missing_pct, 6),
        "dupes": dupes,
        "dupes_pct": round(dupes_pct, 6),
        "memory_mb": round(memory_mb, 3),
        "dtypes_summary": dtypes_summary,
        "n_numeric": n_numeric,
        "n_object": n_object,
        "n_category": n_category,
        "n_bool": n_bool,
        "n_datetime": n_datetime,
        "avg_nulls_per_col": round(avg_nulls_per_col, 3),
        "avg_nulls_per_row_pct": round(avg_nulls_per_row_pct, 6),
        "top_missing_cols": top_missing_cols,
        "constant_columns": constant_columns,
        "cardinality": cardinality,
        "numeric": numeric_flags,
        "sample_duplicate_indices": sample_duplicate_indices,
        "warnings": warnings,
    }

    return report
