# src/utils/validators.py
"""
Moduł validators PRO++++ - Zaawansowana walidacja i analiza jakości danych.

Funkcjonalności:
- Kompleksowa analiza jakości danych z metrykami PRO
- Wykrywanie anomalii i red flags
- Analiza kardynalności i rozkładów
- Walidacja typów i spójności danych
- Performance optimizations z caching
- Thread-safe operations
- Szczegółowe ostrzeżenia i rekomendacje
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Progi i limity
MISSING_HIGH_THRESHOLD = 0.20  # 20% braków = warning
DUPES_THRESHOLD = 0.01  # 1% duplikatów = warning
NON_FINITE_THRESHOLD = 0.001  # 0.1% inf/nan = warning
SKEW_THRESHOLD = 2.0  # |skewness| > 2 = silnie skośny
KURTOSIS_THRESHOLD = 7.0  # |kurtosis| > 7 = heavy tails
HIGH_CARDINALITY_THRESHOLD = 30
LOW_CARDINALITY_THRESHOLD = 10
OUTLIER_IQR_MULTIPLIER = 3.0  # IQR * 3 dla outliers

# Limity wydajnościowe
MAX_WORKERS = 4
MAX_SAMPLE_SIZE = 100_000


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class ColumnStats:
    """Statystyki pojedynczej kolumny."""
    name: str
    dtype: str
    dtype_group: str
    missing_count: int
    missing_pct: float
    nunique: int
    is_constant: bool
    memory_bytes: int


@dataclass(frozen=True)
class NumericAnalysis:
    """Analiza kolumn numerycznych."""
    non_finite_pct: float
    skewed_cols: List[str]
    high_kurtosis_cols: List[str]
    outlier_cols: Dict[str, int]  # kolumna -> liczba outlierów
    zero_variance_cols: List[str]
    negative_cols: List[str]  # kolumny z wartościami ujemnymi


@dataclass(frozen=True)
class CardinalityAnalysis:
    """Analiza kardynalności kategorycznych."""
    low_cardinality: List[str]  # 3-10 unikalnych
    high_cardinality: List[str]  # >30 unikalnych
    binary_like: List[str]  # 2 unikalne
    unique_like: List[str]  # każda wartość unikalna (potencjalny ID)


@dataclass(frozen=True)
class QualityReport:
    """Kompleksowy raport jakości danych."""
    # Podstawowe metryki
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    dupes_pct: float
    memory_mb: float
    
    # Typy danych
    dtypes_summary: Dict[str, int]
    n_numeric: int
    n_object: int
    n_category: int
    n_bool: int
    n_datetime: int
    
    # Analiza braków
    avg_nulls_per_col: float
    avg_nulls_per_row_pct: float
    top_missing_cols: Dict[str, float]
    
    # Kolumny problemowe
    constant_columns: List[str]
    
    # Analiza kategorycznych
    cardinality: CardinalityAnalysis
    
    # Analiza numerycznych
    numeric: NumericAnalysis
    
    # Duplikaty
    sample_duplicate_indices: List[str]
    
    # Ostrzeżenia i rekomendacje
    warnings: List[str]
    recommendations: List[str]
    
    # Metadata
    quality_score: float  # 0-100
    severity: Literal["ok", "warning", "critical"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika JSON-serializowalnego."""
        result = asdict(self)
        # Konwertuj zagnieżdżone dataclasses
        result["cardinality"] = asdict(self.cardinality)
        result["numeric"] = asdict(self.numeric)
        return result


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _safe_float(x: Any) -> float:
    """
    Bezpieczna konwersja do float.
    
    Args:
        x: Wartość do konwersji
        
    Returns:
        Float lub 0.0 przy błędzie
    """
    try:
        val = float(x)
        return val if np.isfinite(val) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _safe_int(x: Any) -> int:
    """
    Bezpieczna konwersja do int.
    
    Args:
        x: Wartość do konwersji
        
    Returns:
        Int lub 0 przy błędzie
    """
    try:
        return int(x)
    except (ValueError, TypeError, OverflowError):
        return 0


@lru_cache(maxsize=128)
def _dtype_group(dtype_str: str) -> str:
    """
    Grupuje typ danych do kategorii (cachowane).
    
    Args:
        dtype_str: String reprezentacja dtype
        
    Returns:
        Grupa typu danych
    """
    dtype_str_lower = dtype_str.lower()
    
    if "bool" in dtype_str_lower:
        return "bool"
    if "datetime" in dtype_str_lower or "timedelta" in dtype_str_lower:
        return "datetime"
    if any(t in dtype_str_lower for t in ["int", "float", "number"]):
        return "numeric"
    if "category" in dtype_str_lower:
        return "category"
    
    return "object"


def _classify_column_dtype(series: pd.Series) -> Tuple[str, str]:
    """
    Klasyfikuje dtype kolumny.
    
    Args:
        series: Seria do klasyfikacji
        
    Returns:
        Tuple (dtype_string, dtype_group)
    """
    dtype_str = str(series.dtype)
    
    # Szybka klasyfikacja
    if pd.api.types.is_bool_dtype(series):
        return dtype_str, "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return dtype_str, "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return dtype_str, "numeric"
    if pd.api.types.is_categorical_dtype(series):
        return dtype_str, "category"
    
    return dtype_str, "object"


def _compute_column_stats(col_name: str, series: pd.Series) -> ColumnStats:
    """
    Oblicza statystyki dla pojedynczej kolumny.
    
    Args:
        col_name: Nazwa kolumny
        series: Seria danych
        
    Returns:
        ColumnStats z metrykami
    """
    try:
        dtype_str, dtype_group = _classify_column_dtype(series)
        
        missing_count = _safe_int(series.isna().sum())
        missing_pct = _safe_float(missing_count / len(series) if len(series) > 0 else 0.0)
        
        nunique = _safe_int(series.nunique(dropna=True))
        is_constant = nunique <= 1
        
        memory_bytes = _safe_int(series.memory_usage(deep=True))
        
        return ColumnStats(
            name=col_name,
            dtype=dtype_str,
            dtype_group=dtype_group,
            missing_count=missing_count,
            missing_pct=missing_pct,
            nunique=nunique,
            is_constant=is_constant,
            memory_bytes=memory_bytes
        )
    except Exception as e:
        logger.warning(f"Error computing stats for column {col_name}: {e}")
        # Fallback stats
        return ColumnStats(
            name=col_name,
            dtype="unknown",
            dtype_group="object",
            missing_count=0,
            missing_pct=0.0,
            nunique=0,
            is_constant=False,
            memory_bytes=0
        )


# ========================================================================================
# MISSING VALUES ANALYSIS
# ========================================================================================

def _top_missing_cols(df: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
    """
    Znajduje kolumny z największym odsetkiem braków.
    
    Args:
        df: DataFrame do analizy
        top_n: Liczba top kolumn
        
    Returns:
        Słownik {kolumna: procent_braków}
    """
    if df.empty:
        return {}
    
    try:
        pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
        pct = pct.round(2).head(top_n)
        return {str(k): _safe_float(v) for k, v in pct.to_dict().items() if v > 0}
    except Exception as e:
        logger.error(f"Error computing top missing columns: {e}")
        return {}


# ========================================================================================
# CARDINALITY ANALYSIS
# ========================================================================================

def _cardinality_analysis(df: pd.DataFrame) -> CardinalityAnalysis:
    """
    Analizuje kardynalność kolumn kategorycznych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        CardinalityAnalysis z klasyfikacją
    """
    lows: List[str] = []
    highs: List[str] = []
    binaries: List[str] = []
    uniques: List[str] = []
    
    try:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        
        for col in categorical_cols:
            try:
                nunique = _safe_int(df[col].nunique(dropna=True))
                col_len = len(df[col].dropna())
                
                if nunique == 0:
                    continue
                
                # Binary
                if nunique == 2:
                    binaries.append(str(col))
                # Low cardinality
                elif 3 <= nunique <= LOW_CARDINALITY_THRESHOLD:
                    lows.append(str(col))
                # High cardinality
                elif nunique > HIGH_CARDINALITY_THRESHOLD:
                    highs.append(str(col))
                
                # Unique-like (potencjalny ID)
                if col_len > 0 and nunique / col_len > 0.95:
                    uniques.append(str(col))
                    
            except Exception as e:
                logger.warning(f"Error analyzing cardinality for column {col}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in cardinality analysis: {e}")
    
    return CardinalityAnalysis(
        low_cardinality=lows,
        high_cardinality=highs,
        binary_like=binaries,
        unique_like=uniques
    )


# ========================================================================================
# NUMERIC ANALYSIS
# ========================================================================================

def _detect_outliers_iqr(series: pd.Series, multiplier: float = OUTLIER_IQR_MULTIPLIER) -> int:
    """
    Wykrywa outliery metodą IQR.
    
    Args:
        series: Seria numeryczna
        multiplier: Mnożnik IQR
        
    Returns:
        Liczba outlierów
    """
    try:
        clean = series.dropna()
        if len(clean) < 4:
            return 0
        
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return 0
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = ((clean < lower_bound) | (clean > upper_bound)).sum()
        return _safe_int(outliers)
        
    except Exception as e:
        logger.warning(f"Error detecting outliers: {e}")
        return 0


def _numeric_analysis(df: pd.DataFrame) -> NumericAnalysis:
    """
    Zaawansowana analiza kolumn numerycznych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        NumericAnalysis z metrykami
    """
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return NumericAnalysis(
            non_finite_pct=0.0,
            skewed_cols=[],
            high_kurtosis_cols=[],
            outlier_cols={},
            zero_variance_cols=[],
            negative_cols=[]
        )
    
    try:
        # Non-finite values
        non_finite_mask = ~np.isfinite(numeric_df.to_numpy(dtype=float))
        non_finite_pct = _safe_float(non_finite_mask.mean() * 100.0)
        
        # Skewness
        skewed_cols: List[str] = []
        try:
            skewness = numeric_df.skew(numeric_only=True)
            for col, skew_val in skewness.items():
                if np.isfinite(skew_val) and abs(skew_val) > SKEW_THRESHOLD:
                    skewed_cols.append(str(col))
        except Exception as e:
            logger.warning(f"Error computing skewness: {e}")
        
        # Kurtosis
        high_kurtosis_cols: List[str] = []
        try:
            kurtosis = numeric_df.kurtosis(numeric_only=True)
            for col, kurt_val in kurtosis.items():
                if np.isfinite(kurt_val) and abs(kurt_val) > KURTOSIS_THRESHOLD:
                    high_kurtosis_cols.append(str(col))
        except Exception as e:
            logger.warning(f"Error computing kurtosis: {e}")
        
        # Outliers (parallel processing dla wydajności)
        outlier_cols: Dict[str, int] = {}
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_detect_outliers_iqr, numeric_df[col]): col
                    for col in numeric_df.columns
                }
                
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        n_outliers = future.result()
                        if n_outliers > 0:
                            outlier_cols[str(col)] = n_outliers
                    except Exception as e:
                        logger.warning(f"Error detecting outliers for {col}: {e}")
        except Exception as e:
            logger.warning(f"Error in parallel outlier detection: {e}")
        
        # Zero variance
        zero_variance_cols: List[str] = []
        try:
            variances = numeric_df.var(numeric_only=True)
            for col, var_val in variances.items():
                if np.isfinite(var_val) and var_val == 0.0:
                    zero_variance_cols.append(str(col))
        except Exception as e:
            logger.warning(f"Error computing variance: {e}")
        
        # Negative values
        negative_cols: List[str] = []
        try:
            for col in numeric_df.columns:
                if (numeric_df[col] < 0).any():
                    negative_cols.append(str(col))
        except Exception as e:
            logger.warning(f"Error checking negative values: {e}")
        
        return NumericAnalysis(
            non_finite_pct=round(non_finite_pct, 4),
            skewed_cols=skewed_cols,
            high_kurtosis_cols=high_kurtosis_cols,
            outlier_cols=outlier_cols,
            zero_variance_cols=zero_variance_cols,
            negative_cols=negative_cols
        )
        
    except Exception as e:
        logger.error(f"Error in numeric analysis: {e}", exc_info=True)
        return NumericAnalysis(
            non_finite_pct=0.0,
            skewed_cols=[],
            high_kurtosis_cols=[],
            outlier_cols={},
            zero_variance_cols=[],
            negative_cols=[]
        )


# ========================================================================================
# QUALITY SCORING
# ========================================================================================

def _compute_quality_score(
    missing_pct: float,
    dupes_pct: float,
    non_finite_pct: float,
    n_constant: int,
    total_cols: int
) -> float:
    """
    Oblicza ogólny score jakości (0-100).
    
    Args:
        missing_pct: Procent braków
        dupes_pct: Procent duplikatów
        non_finite_pct: Procent wartości niefinitowych
        n_constant: Liczba kolumn stałych
        total_cols: Całkowita liczba kolumn
        
    Returns:
        Score 0-100 (wyższy = lepsza jakość)
    """
    score = 100.0
    
    # Penalty za braki
    score -= min(missing_pct * 100, 30)
    
    # Penalty za duplikaty
    score -= min(dupes_pct * 100, 20)
    
    # Penalty za non-finite
    score -= min(non_finite_pct * 10, 15)
    
    # Penalty za kolumny stałe
    if total_cols > 0:
        constant_ratio = n_constant / total_cols
        score -= min(constant_ratio * 100, 20)
    
    return max(0.0, min(100.0, score))


def _determine_severity(quality_score: float) -> Literal["ok", "warning", "critical"]:
    """
    Określa poziom severity na podstawie score.
    
    Args:
        quality_score: Score jakości 0-100
        
    Returns:
        Poziom severity
    """
    if quality_score >= 80:
        return "ok"
    elif quality_score >= 60:
        return "warning"
    else:
        return "critical"


# ========================================================================================
# WARNINGS & RECOMMENDATIONS
# ========================================================================================

def _generate_warnings(
    missing_pct: float,
    dupes_pct: float,
    constant_cols: List[str],
    non_finite_pct: float,
    cardinality: CardinalityAnalysis,
    numeric: NumericAnalysis
) -> List[str]:
    """
    Generuje ostrzeżenia na podstawie analizy.
    
    Args:
        missing_pct: Procent braków
        dupes_pct: Procent duplikatów
        constant_cols: Lista kolumn stałych
        non_finite_pct: Procent wartości niefinitowych
        cardinality: Analiza kardynalności
        numeric: Analiza numeryczna
        
    Returns:
        Lista ostrzeżeń
    """
    warnings: List[str] = []
    
    if missing_pct > MISSING_HIGH_THRESHOLD:
        warnings.append(
            f"⚠️ Wysoki odsetek braków ({missing_pct*100:.1f}% > {MISSING_HIGH_THRESHOLD*100}%)"
        )
    
    if dupes_pct > DUPES_THRESHOLD:
        warnings.append(
            f"⚠️ Zauważalna liczba duplikatów ({dupes_pct*100:.2f}% > {DUPES_THRESHOLD*100}%)"
        )
    
    if constant_cols:
        n = len(constant_cols)
        warnings.append(
            f"⚠️ Wykryto {n} {'kolumnę' if n == 1 else 'kolumny' if n < 5 else 'kolumn'} "
            f"o zerowej wariancji"
        )
    
    if non_finite_pct > NON_FINITE_THRESHOLD:
        warnings.append(
            f"⚠️ Wartości niefinity w danych numerycznych ({non_finite_pct:.2f}%)"
        )
    
    if cardinality.unique_like:
        n = len(cardinality.unique_like)
        warnings.append(
            f"ℹ️ Wykryto {n} {'kolumnę' if n == 1 else 'kolumny' if n < 5 else 'kolumn'} "
            f"przypominające ID (95%+ unikalnych wartości)"
        )
    
    if len(numeric.outlier_cols) > 0:
        total_outliers = sum(numeric.outlier_cols.values())
        warnings.append(
            f"ℹ️ Wykryto {total_outliers} outlierów w {len(numeric.outlier_cols)} kolumnach"
        )
    
    return warnings


def _generate_recommendations(
    missing_pct: float,
    dupes: int,
    constant_cols: List[str],
    cardinality: CardinalityAnalysis,
    numeric: NumericAnalysis
) -> List[str]:
    """
    Generuje rekomendacje działań.
    
    Args:
        missing_pct: Procent braków
        dupes: Liczba duplikatów
        constant_cols: Lista kolumn stałych
        cardinality: Analiza kardynalności
        numeric: Analiza numeryczna
        
    Returns:
        Lista rekomendacji
    """
    recommendations: List[str] = []
    
    if missing_pct > 0.05:
        recommendations.append(
            "💡 Zastosuj imputation dla braków (median/mean dla numerycznych, mode dla kategorycznych)"
        )
    
    if dupes > 0:
        recommendations.append(
            "💡 Usuń duplikaty przed treningiem modelu (df.drop_duplicates())"
        )
    
    if constant_cols:
        recommendations.append(
            f"💡 Usuń kolumny stałe - nie wnoszą informacji: {constant_cols[:3]}"
            + ("..." if len(constant_cols) > 3 else "")
        )
    
    if cardinality.unique_like:
        recommendations.append(
            f"💡 Rozważ usunięcie kolumn ID-like: {cardinality.unique_like[:3]}"
            + ("..." if len(cardinality.unique_like) > 3 else "")
        )
    
    if cardinality.high_cardinality:
        recommendations.append(
            f"💡 Kolumny wysokiej kardynalności mogą wymagać encodingu: "
            f"{cardinality.high_cardinality[:3]}"
            + ("..." if len(cardinality.high_cardinality) > 3 else "")
        )
    
    if numeric.skewed_cols:
        recommendations.append(
            f"💡 Zastosuj transformację (log/box-cox) dla skośnych kolumn: "
            f"{numeric.skewed_cols[:3]}"
            + ("..." if len(numeric.skewed_cols) > 3 else "")
        )
    
    if numeric.outlier_cols:
        top_outliers = sorted(
            numeric.outlier_cols.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        recommendations.append(
            f"💡 Sprawdź outliery w kolumnach: "
            f"{', '.join(f'{col} ({n})' for col, n in top_outliers)}"
        )
    
    return recommendations


# ========================================================================================
# MAIN VALIDATION FUNCTION
# ========================================================================================

def basic_quality_checks(df: pd.DataFrame) -> dict:
    """
    Kompleksowa analiza jakości danych PRO++++.
    
    Funkcja wykonuje zaawansowaną analizę obejmującą:
    - Podstawowe metryki (wiersze, kolumny, pamięć)
    - Analizę braków (agregaty + top missing columns)
    - Wykrywanie duplikatów z przykładowymi indeksami
    - Analizę typów danych (numeric, object, category, bool, datetime)
    - Wykrywanie kolumn stałych (zero variance)
    - Analizę kardynalności kategorycznych (low/high/binary/unique-like)
    - Zaawansowaną analizę numeryczną (skewness, kurtosis, outliers, non-finite)
    - Obliczanie quality score (0-100)
    - Generowanie ostrzeżeń i rekomendacji
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Słownik z kompleksowym raportem jakości (JSON-serializowalny)
        
    Raises:
        Nie rzuca wyjątków - zwraca dict z kluczem "error" w przypadku błędu
        
    Examples:
        >>> report = basic_quality_checks(df)
        >>> print(f"Quality score: {report['quality_score']}")
        >>> print(f"Warnings: {report['warnings']}")
    """
    # Walidacja wejścia
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid input: expected pandas DataFrame")
        return {"error": "Brak danych lub nieprawidłowy typ (oczekiwano DataFrame)"}
    
    rows = int(len(df))
    cols = int(df.shape[1])
    
    # Edge case: pusty DataFrame
    if rows == 0:
        logger.warning("Empty DataFrame provided")
        return {
            "rows": 0,
            "cols": cols,
            "missing_pct": 0.0,
            "dupes": 0,
            "dupes_pct": 0.0,
            "memory_mb": 0.0,
            "quality_score": 100.0,
            "severity": "ok",
            "warnings": [],
            "recommendations": []
        }
    
    try:
        # ============================================================================
        # PODSTAWOWE METRYKI
        # ============================================================================
        
        total_cells = max(1, df.size)
        missing_all = _safe_int(df.isna().sum().sum())
        missing_pct = _safe_float(missing_all / total_cells)
        
        dupes = _safe_int(df.duplicated().sum())
        dupes_pct = _safe_float(dupes / rows if rows > 0 else 0.0)
        
        # Pamięć (deep=True dla dokładności)
        try:
            memory_mb = _safe_float(df.memory_usage(deep=True).sum() / (1024 ** 2))
        except Exception:
            memory_mb = _safe_float(df.memory_usage().sum() / (1024 ** 2))
        
        # ============================================================================
        # ANALIZA TYPÓW DANYCH
        # ============================================================================
        
        dtype_counts: Dict[str, int] = {}
        for col in df.columns:
            _, dtype_group = _classify_column_dtype(df[col])
            dtype_counts[dtype_group] = dtype_counts.get(dtype_group, 0) + 1
        
        n_numeric = dtype_counts.get("numeric", 0)
        n_object = dtype_counts.get("object", 0)
        n_category = dtype_counts.get("category", 0)
        n_bool = dtype_counts.get("bool", 0)
        n_datetime = dtype_counts.get("datetime", 0)
        
        # ============================================================================
        # ANALIZA BRAKÓW
        # ============================================================================
        
        avg_nulls_per_col = _safe_float(df.isna().sum().mean())
        avg_nulls_per_row_pct = _safe_float(df.isna().mean(axis=1).mean())
        
        top_missing_cols = _top_missing_cols(df, top_n=10)
        
        # ============================================================================
        # KOLUMNY STAŁE
        # ============================================================================
        
        constant_columns: List[str] = []
        try:
            for col in df.columns:
                if df[col].nunique(dropna=True) <= 1:
                    constant_columns.append(str(col))
        except Exception as e:
            logger.warning(f"Error detecting constant columns: {e}")
        
        # ============================================================================
        # ANALIZA KARDYNALNOŚCI
        # ============================================================================
        
        cardinality = _cardinality_analysis(df)
        
        # ============================================================================
        # ANALIZA NUMERYCZNA
        # ============================================================================
        
        numeric_analysis = _numeric_analysis(df)
        
        # ============================================================================
        # DUPLIKATY - PRZYKŁADOWE INDEKSY
        # ============================================================================
        
        sample_duplicate_indices: List[str] = []
        try:
            if dupes > 0:
                dup_mask = df.duplicated()
                dup_indices = df.index[dup_mask].tolist()[:10]
                sample_duplicate_indices = [str(idx) for idx in dup_indices]
        except Exception as e:
            logger.warning(f"Error extracting duplicate indices: {e}")
        
        # ============================================================================
        # QUALITY SCORE & SEVERITY
        # ============================================================================
        
        quality_score = _compute_quality_score(
            missing_pct=missing_pct,
            dupes_pct=dupes_pct,
            non_finite_pct=numeric_analysis.non_finite_pct / 100.0,
            n_constant=len(constant_columns),
            total_cols=cols
        )
        
        severity = _determine_severity(quality_score)
        
        # ============================================================================
        # OSTRZEŻENIA & REKOMENDACJE
        # ============================================================================
        
        warnings = _generate_warnings(
            missing_pct=missing_pct,
            dupes_pct=dupes_pct,
            constant_cols=constant_columns,
            non_finite_pct=numeric_analysis.non_finite_pct,
            cardinality=cardinality,
            numeric=numeric_analysis
        )
        
        recommendations = _generate_recommendations(
            missing_pct=missing_pct,
            dupes=dupes,
            constant_cols=constant_columns,
            cardinality=cardinality,
            numeric=numeric_analysis
        )
        
        # ============================================================================
        # BUDOWANIE RAPORTU
        # ============================================================================
        
        report = QualityReport(
            # Podstawowe
            rows=rows,
            cols=cols,
            missing_pct=round(missing_pct, 6),
            dupes=dupes,
            dupes_pct=round(dupes_pct, 6),
            memory_mb=round(memory_mb, 3),
            
            # Typy
            dtypes_summary=dtype_counts,
            n_numeric=n_numeric,
            n_object=n_object,
            n_category=n_category,
            n_bool=n_bool,
            n_datetime=n_datetime,
            
            # Braki
            avg_nulls_per_col=round(avg_nulls_per_col, 3),
            avg_nulls_per_row_pct=round(avg_nulls_per_row_pct, 6),
            top_missing_cols=top_missing_cols,
            
            # Problemowe kolumny
            constant_columns=constant_columns,
            
            # Kardynalność
            cardinality=cardinality,
            
            # Numeryczne
            numeric=numeric_analysis,
            
            # Duplikaty
            sample_duplicate_indices=sample_duplicate_indices,
            
            # Wnioski
            warnings=warnings,
            recommendations=recommendations,
            quality_score=round(quality_score, 2),
            severity=severity
        )
        
        logger.info(
            f"Quality check completed: score={quality_score:.1f}, "
            f"severity={severity}, warnings={len(warnings)}"
        )
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"Critical error in quality checks: {e}", exc_info=True)
        return {
            "error": f"Błąd podczas analizy: {str(e)}",
            "rows": rows,
            "cols": cols,
            "quality_score": 0.0,
            "severity": "critical"
        }


# ========================================================================================
# DODATKOWE FUNKCJE WALIDACYJNE
# ========================================================================================

def validate_dataframe_for_ml(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    min_rows: int = 10,
    max_missing_pct: float = 0.5
) -> Tuple[bool, List[str]]:
    """
    Waliduje DataFrame pod kątem gotowości do treningu ML.
    
    Args:
        df: DataFrame do walidacji
        target_col: Opcjonalna kolumna celu
        min_rows: Minimalna liczba wierszy
        max_missing_pct: Maksymalny akceptowalny procent braków
        
    Returns:
        Tuple (is_valid, list_of_issues)
        
    Examples:
        >>> is_valid, issues = validate_dataframe_for_ml(df, target_col="price")
        >>> if not is_valid:
        >>>     print("Issues:", issues)
    """
    issues: List[str] = []
    
    # Podstawowa walidacja
    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["DataFrame jest None lub nieprawidłowy typ"]
    
    if df.empty:
        return False, ["DataFrame jest pusty"]
    
    # Minimalna liczba wierszy
    if len(df) < min_rows:
        issues.append(f"Za mało wierszy ({len(df)} < {min_rows})")
    
    # Brak kolumn
    if df.shape[1] == 0:
        issues.append("Brak kolumn w DataFrame")
        return False, issues
    
    # Walidacja target
    if target_col:
        if target_col not in df.columns:
            issues.append(f"Kolumna celu '{target_col}' nie istnieje")
        else:
            if df[target_col].isna().all():
                issues.append(f"Kolumna celu '{target_col}' zawiera tylko NaN")
            
            target_missing_pct = df[target_col].isna().mean()
            if target_missing_pct > 0.1:
                issues.append(
                    f"Kolumna celu ma {target_missing_pct*100:.1f}% braków (>10%)"
                )
    
    # Sprawdź ogólny procent braków
    total_missing_pct = df.isna().sum().sum() / df.size
    if total_missing_pct > max_missing_pct:
        issues.append(
            f"Zbyt dużo braków ({total_missing_pct*100:.1f}% > "
            f"{max_missing_pct*100:.0f}%)"
        )
    
    # Sprawdź czy są jakiekolwiek dane numeryczne (dla większości ML)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        issues.append("Brak kolumn numerycznych - większość algorytmów ML wymaga cech numerycznych")
    
    # Sprawdź kolumny całkowicie puste
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        issues.append(f"Kolumny całkowicie puste: {empty_cols[:5]}")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def validate_column_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str]
) -> Tuple[bool, Dict[str, str]]:
    """
    Waliduje czy kolumny mają oczekiwane typy.
    
    Args:
        df: DataFrame do walidacji
        expected_types: Dict {column_name: expected_dtype_group}
                       dtype_group może być: "numeric", "object", "datetime", "bool", "category"
        
    Returns:
        Tuple (all_valid, dict_of_mismatches)
        
    Examples:
        >>> expected = {"age": "numeric", "name": "object", "date": "datetime"}
        >>> is_valid, mismatches = validate_column_types(df, expected)
    """
    mismatches: Dict[str, str] = {}
    
    for col, expected_group in expected_types.items():
        if col not in df.columns:
            mismatches[col] = f"kolumna nie istnieje (oczekiwano {expected_group})"
            continue
        
        _, actual_group = _classify_column_dtype(df[col])
        
        if actual_group != expected_group:
            mismatches[col] = f"typ {actual_group} != oczekiwany {expected_group}"
    
    all_valid = len(mismatches) == 0
    
    return all_valid, mismatches


def check_data_leakage(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.95
) -> List[str]:
    """
    Wykrywa potencjalny data leakage (kolumny zbyt skorelowane z target).
    
    Args:
        df: DataFrame do sprawdzenia
        target_col: Nazwa kolumny celu
        threshold: Próg korelacji (default 0.95)
        
    Returns:
        Lista kolumn podejrzanych o leakage
        
    Examples:
        >>> suspicious = check_data_leakage(df, "price", threshold=0.95)
        >>> if suspicious:
        >>>     print(f"Potencjalny leakage: {suspicious}")
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return []
    
    suspicious_cols: List[str] = []
    
    try:
        # Sprawdź tylko kolumny numeryczne
        numeric_df = df.select_dtypes(include=np.number)
        
        if target_col not in numeric_df.columns:
            logger.info(f"Target '{target_col}' is not numeric, skipping leakage check")
            return []
        
        # Oblicz korelacje
        correlations = numeric_df.corr()[target_col].abs()
        
        # Znajdź kolumny o wysokiej korelacji (excluding target itself)
        for col, corr in correlations.items():
            if col != target_col and np.isfinite(corr) and corr > threshold:
                suspicious_cols.append(col)
        
        if suspicious_cols:
            logger.warning(
                f"Potential data leakage detected: {suspicious_cols} "
                f"(correlation > {threshold})"
            )
        
    except Exception as e:
        logger.error(f"Error checking data leakage: {e}")
    
    return suspicious_cols


def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Szczegółowa analiza zużycia pamięci.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Dict z breakdown zużycia pamięci
        
    Examples:
        >>> mem_info = estimate_memory_usage(df)
        >>> print(f"Total: {mem_info['total_mb']:.2f} MB")
        >>> print(f"Top columns: {mem_info['top_consumers']}")
    """
    if df.empty:
        return {
            "total_mb": 0.0,
            "per_column_mb": {},
            "top_consumers": [],
            "dtypes_breakdown": {}
        }
    
    try:
        # Zużycie per kolumna
        mem_usage = df.memory_usage(deep=True)
        total_bytes = mem_usage.sum()
        total_mb = total_bytes / (1024 ** 2)
        
        # Per kolumna (bez indeksu)
        per_column_mb = {
            str(col): mem_usage[col] / (1024 ** 2)
            for col in df.columns
        }
        
        # Top consumers
        sorted_cols = sorted(
            per_column_mb.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_consumers = [
            {"column": col, "mb": round(mb, 3)}
            for col, mb in sorted_cols[:10]
        ]
        
        # Breakdown per dtype
        dtypes_breakdown: Dict[str, float] = {}
        for col in df.columns:
            _, dtype_group = _classify_column_dtype(df[col])
            dtypes_breakdown[dtype_group] = dtypes_breakdown.get(dtype_group, 0.0) + per_column_mb[col]
        
        dtypes_breakdown = {
            k: round(v, 3) for k, v in dtypes_breakdown.items()
        }
        
        return {
            "total_mb": round(total_mb, 3),
            "per_column_mb": {k: round(v, 3) for k, v in per_column_mb.items()},
            "top_consumers": top_consumers,
            "dtypes_breakdown": dtypes_breakdown
        }
        
    except Exception as e:
        logger.error(f"Error estimating memory usage: {e}")
        return {
            "total_mb": 0.0,
            "per_column_mb": {},
            "top_consumers": [],
            "dtypes_breakdown": {},
            "error": str(e)
        }


# ========================================================================================
# ALIASY DLA BACKWARD COMPATIBILITY
# ========================================================================================

def validate(df: pd.DataFrame) -> dict:
    """
    Alias dla basic_quality_checks (backward compatibility).
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Dict z raportem jakości
    """
    return basic_quality_checks(df)


# ========================================================================================
# EXPORT
# ========================================================================================

__all__ = [
    "basic_quality_checks",
    "validate",
    "validate_dataframe_for_ml",
    "validate_column_types",
    "check_data_leakage",
    "estimate_memory_usage",
    "QualityReport",
    "ColumnStats",
    "NumericAnalysis",
    "CardinalityAnalysis",
]
        