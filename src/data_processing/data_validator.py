# data_quality.py — TURBO PRO
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd


# =========================
# Opcje i typy raportu
# =========================
Severity = Literal["info", "warn", "error"]

@dataclass(frozen=True)
class QualityOptions:
    high_missing_threshold: float = 0.50     # kolumny z >= 50% braków
    near_constant_threshold: float = 0.99    # udział dominującej wartości
    unique_as_id_ratio: float = 0.98         # >=98% unikalności traktuj jako ID-kandydat
    zscore_threshold: float = 4.0            # odchyłki Z-score
    iqr_threshold: float = 3.0               # odchyłki IQR (robust)
    min_outlier_sample: int = 10             # min. obserwacji by liczyć outliery
    sample_text_check: int = 100             # próba dla anomalii tekstowych
    check_correlations: bool = False         # ostrzeż, gdy duplikują się silnie skorelowane numeryczne
    strong_corr_threshold: float = 0.98
    limit_cols_for_corr: int = 200           # bezpieczeństwo
    detect_mixed_types: bool = True          # wykrywanie mieszanych typów w kolumnach object
    detect_identical_columns: bool = True    # kolumny 1:1 identyczne
    memory_stats: bool = True                # dołącz zużycie pamięci

@dataclass(frozen=True)
class Finding:
    kind: str
    column: Optional[str]
    message: str
    severity: Severity
    details: Dict[str, Any] = None

@dataclass(frozen=True)
class QualityReport:
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    dtypes_summary: Dict[str, int]
    memory_bytes: Optional[int]
    constant_columns: List[str]
    near_constant_columns: Dict[str, float]
    unique_columns: List[str]
    high_missing_cols: Dict[str, float]
    outliers_detected_z: Dict[str, int]
    outliers_detected_iqr: Dict[str, int]
    textual_anomalies: List[str]
    mixed_type_columns: List[str]
    identical_columns: List[Tuple[str, str]]
    strong_correlations: List[Tuple[str, str, float]]
    findings: List[Finding]


# =========================
# Helpers
# =========================
def _dtype_summary(df: pd.DataFrame) -> Dict[str, int]:
    cats = {
        "numeric": df.select_dtypes(include=[np.number]).shape[1],
        "bool": df.select_dtypes(include=["boolean", bool]).shape[1],
        "datetime": df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, tz]"]).shape[1],
        "category": df.select_dtypes(include=["category"]).shape[1],
        "string": df.select_dtypes(include=["string"]).shape[1],
        "object": df.select_dtypes(include=["object"]).shape[1],
        "timedelta": df.select_dtypes(include=["timedelta"]).shape[1],
    }
    return {k: v for k, v in cats.items() if v > 0}

def _detect_constant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.Series(df[c]).nunique(dropna=True) <= 1]

def _detect_near_constant(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for c in df.columns:
        s = pd.Series(df[c])
        if len(s) == 0:
            continue
        vc = s.value_counts(dropna=True)
        if vc.empty:
            continue
        dom_ratio = float(vc.iloc[0]) / float(len(s))
        if dom_ratio >= threshold and s.nunique(dropna=True) > 1:
            res[c] = round(dom_ratio, 4)
    return res

def _detect_unique_columns(df: pd.DataFrame, as_id_ratio: float) -> list[str]:
    n = len(df)
    if n == 0:
        return []
    out = []
    for c in df.columns:
        try:
            r = df[c].nunique(dropna=True) / n
            if r >= as_id_ratio:
                out.append(c)
        except Exception:
            continue
    return out

def _detect_high_missing(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    ratios = df.isna().mean()
    mask = ratios >= threshold
    return {c: round(float(rat * 100.0), 2) for c, rat in ratios[mask].items()}

def _outliers_zscore(df: pd.DataFrame, z_thresh: float, min_n: int) -> dict[str, int]:
    out: Dict[str, int] = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        vals = pd.Series(df[c]).dropna()
        if len(vals) < min_n:
            continue
        sd = float(vals.std(ddof=0))
        if sd == 0 or math.isclose(sd, 0.0):
            continue
        z = (vals - float(vals.mean())) / sd
        n_out = int((np.abs(z) > z_thresh).sum())
        if n_out > 0:
            out[c] = n_out
    return out

def _outliers_iqr(df: pd.DataFrame, iqr_k: float, min_n: int) -> dict[str, int]:
    out: Dict[str, int] = {}
    for c in df.select_dtypes(include=[np.number]).columns:
        vals = pd.Series(df[c]).dropna()
        if len(vals) < min_n:
            continue
        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        n_out = int(((vals < lo) | (vals > hi)).sum())
        if n_out > 0:
            out[c] = n_out
    return out

def _textual_anomalies(df: pd.DataFrame, sample_n: int) -> list[str]:
    cols = df.select_dtypes(include=["object", "string"]).columns
    bad: List[str] = []
    for c in cols:
        s = pd.Series(df[c]).astype("string").dropna().head(sample_n)
        if s.empty:
            continue
        # pusty string lub whitespace-only
        if (s.str.strip() == "").any():
            bad.append(c)
    return bad

def _mixed_type_columns(df: pd.DataFrame) -> list[str]:
    out: List[str] = []
    for c in df.columns:
        s = pd.Series(df[c])
        if s.dtype == "object":
            types = set(type(x).__name__ for x in s.dropna().head(200))
            if len(types) > 1:
                out.append(c)
    return out

def _identical_columns(df: pd.DataFrame, limit: int = 1000) -> List[Tuple[str, str]]:
    """Zwraca pary kolumn, które są 1:1 identyczne (na podstawie hashów). Limit — bezpieczeństwo O(n^2)."""
    ncols = df.shape[1]
    if ncols == 0:
        return []
    # Szybki fingerprint każdej kolumny (hash wartości+NA pattern)
    fp = {}
    for c in df.columns:
        s = pd.Series(df[c])
        # użyj tuple pierwszych N i ostatnich N wartości + count NaN (tanio)
        head = tuple(s.head(50).tolist())
        tail = tuple(s.tail(50).tolist())
        nan_count = int(s.isna().sum())
        fp[c] = (hash(head), hash(tail), nan_count, s.dtype.str)
    pairs: List[Tuple[str, str]] = []
    checked = 0
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if checked > limit:
                return pairs
            checked += 1
            a, b = cols[i], cols[j]
            if fp[a] == fp[b]:
                # dodatkowe potwierdzenie (kosztowne) tylko dla kandydatów:
                if pd.Series(df[a]).equals(pd.Series(df[b])):
                    pairs.append((a, b))
    return pairs

def _strong_correlations(df: pd.DataFrame, threshold: float, max_cols: int) -> List[Tuple[str, str, float]]:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0 or num.shape[1] > max_cols:
        return []
    corr = num.corr(numeric_only=True).abs()
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(corr.iat[i, j])
            if v >= threshold:
                pairs.append((cols[i], cols[j], round(v, 4)))
    return pairs


# =========================
# API PRO
# =========================
def validate_pro(df: pd.DataFrame, opts: QualityOptions = QualityOptions()) -> QualityReport:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("validate_pro: brak danych lub pusty DataFrame.")

    rows, cols = df.shape
    missing_pct = float((df.isna().sum().sum()) / (df.size or 1))
    dupes = int(df.duplicated().sum())
    dtypes_sum = _dtype_summary(df)
    mem = int(df.memory_usage(deep=True).sum()) if opts.memory_stats else None

    const_cols = _detect_constant_columns(df)
    near_const = _detect_near_constant(df, opts.near_constant_threshold)
    unique_cols = _detect_unique_columns(df, opts.unique_as_id_ratio)
    high_missing = _detect_high_missing(df, opts.high_missing_threshold)
    out_z = _outliers_zscore(df, opts.zscore_threshold, opts.min_outlier_sample)
    out_iqr = _outliers_iqr(df, opts.iqr_threshold, opts.min_outlier_sample)
    text_anom = _textual_anomalies(df, opts.sample_text_check) if opts.sample_text_check > 0 else []
    mixed_cols = _mixed_type_columns(df) if opts.detect_mixed_types else []
    identical = _identical_columns(df) if opts.detect_identical_columns else []
    strong_corr = _strong_correlations(df, opts.strong_corr_threshold, opts.limit_cols_for_corr) if opts.check_correlations else []

    findings: List[Finding] = []

    if dupes > 0:
        findings.append(Finding("duplicates", None, f"{dupes:,} zduplikowanych wierszy.", "warn"))
    if missing_pct > 0.2:
        findings.append(Finding("missing_overall", None, f"Wysoki udział braków: {missing_pct:.1%}.", "warn"))
    for c in const_cols:
        findings.append(Finding("constant", c, "Kolumna stała (brak wariancji).", "warn"))
    for c, r in near_const.items():
        findings.append(Finding("near_constant", c, f"Dominująca wartość {r*100:.1f}% obserwacji.", "info"))
    for c in unique_cols:
        findings.append(Finding("id_candidate", c, "Kolumna wygląda na identyfikator (prawie unikalna).", "info"))
    for c, pct in high_missing.items():
        sev: Severity = "error" if pct >= 90 else ("warn" if pct >= 50 else "info")
        findings.append(Finding("missing_column", c, f"Braki: {pct:.2f}%.", sev))
    for c in text_anom:
        findings.append(Finding("text_whitespace", c, "Puste / whitespace-only wartości w próbie.", "info"))
    for c in mixed_cols:
        findings.append(Finding("mixed_types", c, "Mieszane typy w kolumnie object.", "warn"))
    for a, b in identical:
        findings.append(Finding("identical_columns", None, f"Kolumny '{a}' i '{b}' są identyczne.", "info"))
    for a, b, v in strong_corr:
        findings.append(Finding("strong_corr", None, f"Silna korelacja {a}~{b}: {v:.3f}.", "info"))

    return QualityReport(
        rows=rows,
        cols=cols,
        missing_pct=missing_pct,
        dupes=dupes,
        dtypes_summary=dtypes_sum,
        memory_bytes=mem,
        constant_columns=const_cols,
        near_constant_columns=near_const,
        unique_columns=unique_cols,
        high_missing_cols=high_missing,
        outliers_detected_z=out_z,
        outliers_detected_iqr=out_iqr,
        textual_anomalies=text_anom,
        mixed_type_columns=mixed_cols,
        identical_columns=identical,
        strong_correlations=strong_corr,
        findings=findings,
    )


# =========================
# Wsteczna kompatybilność
# =========================
def validate(df: pd.DataFrame) -> dict[str, Any]:
    """
    Back-compat wrapper: zwraca płaski dict podobny do Twojej wersji,
    ale korzysta z validate_pro pod spodem.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Brak danych"}

    rep = validate_pro(df)
    out: Dict[str, Any] = {
        "rows": rep.rows,
        "cols": rep.cols,
        "missing_pct": rep.missing_pct,
        "dupes": rep.dupes,
        "constant_columns": rep.constant_columns,
        "unique_columns": rep.unique_columns,
        "high_missing_cols": rep.high_missing_cols,
        "outliers_detected": {**rep.outliers_detected_z, **rep.outliers_detected_iqr},
        "textual_anomalies": rep.textual_anomalies,
        # dodatki (nie psują wstecznej zgodności, ale są przydatne)
        "near_constant_columns": rep.near_constant_columns,
        "mixed_type_columns": rep.mixed_type_columns,
        "identical_columns": rep.identical_columns,
        "dtypes_summary": rep.dtypes_summary,
    }
    return out
