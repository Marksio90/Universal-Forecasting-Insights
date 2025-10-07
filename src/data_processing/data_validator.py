from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict
from ..utils.validators import basic_quality_checks


def _detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Zwraca kolumny o zerowej wariancji (stałe wartości)."""
    constants = []
    for c in df.columns:
        try:
            if df[c].nunique(dropna=True) <= 1:
                constants.append(c)
        except Exception:
            continue
    return constants


def _detect_unique_columns(df: pd.DataFrame) -> list[str]:
    """Zwraca kolumny o unikalnych wartościach (np. ID)."""
    uniques = []
    for c in df.columns:
        try:
            if df[c].nunique(dropna=True) == len(df):
                uniques.append(c)
        except Exception:
            continue
    return uniques


def _detect_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Zwraca kolumny z odsetkiem braków powyżej progu."""
    res = {}
    for c in df.columns:
        ratio = df[c].isna().mean()
        if ratio >= threshold:
            res[c] = round(float(ratio * 100), 2)
    return res


def _detect_outliers(df: pd.DataFrame, z_thresh: float = 4.0) -> dict[str, int]:
    """Zlicza wartości odstające (Z-score > z_thresh) dla kolumn numerycznych."""
    outliers = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        vals = df[c].dropna()
        if len(vals) < 10:
            continue
        zscores = (vals - vals.mean()) / (vals.std(ddof=0) or 1)
        n_out = int((np.abs(zscores) > z_thresh).sum())
        if n_out > 0:
            outliers[c] = n_out
    return outliers


def _detect_textual_anomalies(df: pd.DataFrame) -> list[str]:
    """Sprawdza kolumny tekstowe pod kątem pustych ciągów / whitespace."""
    anomalies = []
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].astype(str).dropna().head(100)
        if any(s.strip() == "" for s in sample):
            anomalies.append(c)
    return anomalies


def validate(df: pd.DataFrame) -> dict[str, Any]:
    """
    Zwraca raport jakości danych:
    {
        'rows': ..., 'cols': ...,
        'missing_pct': ...,
        'dupes': ...,
        'constant_columns': [...],
        'unique_columns': [...],
        'high_missing_cols': {...},
        'outliers_detected': {...},
        'textual_anomalies': [...]
    }
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Brak danych"}

    try:
        report: Dict[str, Any] = basic_quality_checks(df)
    except Exception:
        # fallback: uproszczony raport
        report = {
            "rows": len(df),
            "cols": df.shape[1],
            "missing_pct": float((df.isna().sum().sum()) / (df.size or 1)),
            "dupes": int(df.duplicated().sum()),
        }

    # Rozszerzenia PRO
    report["constant_columns"] = _detect_constant_columns(df)
    report["unique_columns"] = _detect_unique_columns(df)
    report["high_missing_cols"] = _detect_high_missing(df)
    report["outliers_detected"] = _detect_outliers(df)
    report["textual_anomalies"] = _detect_textual_anomalies(df)

    return report
