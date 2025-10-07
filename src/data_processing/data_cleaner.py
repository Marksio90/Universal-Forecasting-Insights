from __future__ import annotations
import pandas as pd
import numpy as np
import re
from typing import Any, Dict
from ..utils.helpers import smart_cast_numeric

def _detect_date_columns(df: pd.DataFrame, sample_size: int = 100) -> list[str]:
    """Wykrywa kolumny, które prawdopodobnie zawierają daty."""
    candidates = []
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].dropna().astype(str).head(sample_size)
        if sample.empty:
            continue
        # prosty regex dat
        date_like = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
        if date_like.mean() > 0.7:
            candidates.append(c)
    return candidates

def _convert_binaries(df: pd.DataFrame) -> pd.DataFrame:
    """Konwertuje kolumny typu yes/no, tak/nie, 0/1 na bool."""
    for c in df.columns:
        if df[c].dtype == object:
            lower = df[c].astype(str).str.lower().str.strip()
            if set(lower.unique()) <= {"0", "1"}:
                df[c] = lower.replace({"0": 0, "1": 1}).astype("Int8")
            elif set(lower.unique()) <= {"tak", "nie", "yes", "no", "true", "false"}:
                df[c] = lower.replace(
                    {"tak": True, "yes": True, "true": True, "nie": False, "no": False, "false": False}
                ).astype("boolean")
    return df

def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inteligentne czyszczenie danych:
    - usuwa duplikaty
    - trymuje stringi
    - konwertuje liczby, daty, binaria
    - usuwa inf
    - imputuje braki (median / '<missing>' / '<unknown>')
    """
    report: Dict[str, Any] = {}

    df2 = df.copy(deep=True)
    report["shape_initial"] = df2.shape

    # Usuń duplikaty
    dupes = int(df2.duplicated().sum())
    if dupes > 0:
        df2 = df2.drop_duplicates()
    report["duplicates_removed"] = dupes

    # Usuń puste ciągi i spacje
    obj_cols = df2.select_dtypes(include="object").columns
    for c in obj_cols:
        df2[c] = (
            df2[c]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

    # Rzutuj kolumny numeryczne
    df2 = smart_cast_numeric(df2)

    # Konwersja dat
    date_cols = _detect_date_columns(df2)
    for c in date_cols:
        try:
            df2[c] = pd.to_datetime(df2[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            pass
    report["date_columns_detected"] = date_cols

    # Konwersja binarna
    df2 = _convert_binaries(df2)

    # Inf -> NaN
    df2 = df2.replace([np.inf, -np.inf], pd.NA)

    # Imputacja braków
    fill_counts = {}
    for c in df2.columns:
        if df2[c].isna().any():
            if pd.api.types.is_numeric_dtype(df2[c]):
                val = df2[c].median(skipna=True)
                df2[c] = df2[c].fillna(val)
                fill_counts[c] = f"median={val}"
            elif pd.api.types.is_categorical_dtype(df2[c]):
                df2[c] = df2[c].cat.add_categories("<unknown>").fillna("<unknown>")
                fill_counts[c] = "<unknown>"
            elif pd.api.types.is_object_dtype(df2[c]):
                df2[c] = df2[c].fillna("<missing>")
                fill_counts[c] = "<missing>"
            elif pd.api.types.is_bool_dtype(df2[c]):
                df2[c] = df2[c].fillna(False)
                fill_counts[c] = "False"
    report["imputed_columns"] = fill_counts

    # Finalne statystyki
    report["shape_final"] = df2.shape
    report["na_total_after"] = int(df2.isna().sum().sum())

    # Zapisz raport jako atrybut (można odczytać w EDA)
    df2.clean_report = report  # type: ignore[attr-defined]
    return df2
