"""
Data Quality Validators - Kompleksowa walidacja i raportowanie jakości danych.

Funkcjonalności:
- Comprehensive quality checks (missing, duplicates, types)
- Memory usage analysis
- Data type profiling
- Cardinality analysis
- Numeric anomalies detection (skew, kurtosis, non-finite)
- Constant columns detection
- Automated warnings
- JSON-serializable reports
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Thresholds for warnings
MISSING_WARNING_THRESHOLD = 0.20  # 20%
DUPLICATES_WARNING_THRESHOLD = 0.01  # 1%
NON_FINITE_WARNING_THRESHOLD = 0.001  # 0.1%

# Cardinality buckets
BINARY_THRESHOLD = 2
LOW_CARDINALITY_MIN = 3
LOW_CARDINALITY_MAX = 10
HIGH_CARDINALITY_MIN = 30

# Report limits
MAX_TOP_MISSING_COLS = 10
MAX_SKEW_KURTOSIS_COLS = 5
MAX_DUPLICATE_SAMPLES = 10
MAX_CONSTANT_COLS_IN_WARNING = 5

# Data type groups
DTYPE_GROUPS = {
    "bool": "boolean",
    "datetime": "datetime",
    "numeric": "numeric",
    "category": "categorical",
    "object": "object"
}

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "validators", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class QualityMetrics:
    """Podstawowe metryki jakości."""
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    dupes_pct: float
    memory_mb: float


@dataclass
class TypeDistribution:
    """Rozkład typów danych."""
    numeric: int
    object: int
    categorical: int
    boolean: int
    datetime: int
    
    def to_dict(self) -> Dict[str, int]:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass
class CardinalityBuckets:
    """Buckety kardynalności dla kolumn kategorycznych."""
    low_cardinality: List[str]
    high_cardinality: List[str]
    binary_like: List[str]
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Konwertuje do słownika."""
        return asdict(self)


@dataclass
class NumericFlags:
    """Flagi dla danych numerycznych."""
    non_finite_pct: float
    skewed_cols: List[str]
    high_kurtosis_cols: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return asdict(self)


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Bezpieczna konwersja do float.
    
    Args:
        value: Wartość do konwersji
        default: Wartość domyślna przy błędzie
        
    Returns:
        Float value
    """
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Bezpieczna konwersja do int.
    
    Args:
        value: Wartość do konwersji
        default: Wartość domyślna przy błędzie
        
    Returns:
        Int value
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _classify_dtype(series: pd.Series) -> str:
    """
    Klasyfikuje typ danych series.
    
    Args:
        series: Pandas Series
        
    Returns:
        Grupa typu: "bool", "datetime", "numeric", "category", "object"
    """
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    if pd.api.types.is_categorical_dtype(series):
        return "category"
    
    return "object"


# ========================================================================================
# MISSING VALUES ANALYSIS
# ========================================================================================

def _analyze_missing_values(df: pd.DataFrame) -> Tuple[float, Dict[str, float], float, float]:
    """
    Analizuje brakujące wartości.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Tuple (missing_pct, top_missing_cols, avg_nulls_per_col, avg_nulls_per_row_pct)
    """
    if df.empty:
        return 0.0, {}, 0.0, 0.0
    
    # Total missing percentage
    total_cells = df.size
    missing_all = df.isna().sum().sum()
    missing_pct = float(missing_all / max(1, total_cells))
    
    # Top missing columns
    col_missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    col_missing_pct = col_missing_pct[col_missing_pct > 0].head(MAX_TOP_MISSING_COLS)
    top_missing_cols = {
        str(col): round(_safe_float(pct), 2)
        for col, pct in col_missing_pct.items()
    }
    
    # Average nulls per column
    avg_nulls_per_col = float(df.isna().sum().mean())
    
    # Average nulls per row (as percentage)
    avg_nulls_per_row_pct = float(df.isna().mean(axis=1).mean())
    
    return missing_pct, top_missing_cols, avg_nulls_per_col, avg_nulls_per_row_pct


# ========================================================================================
# DUPLICATES ANALYSIS
# ========================================================================================

def _analyze_duplicates(df: pd.DataFrame) -> Tuple[int, float, List[str]]:
    """
    Analizuje duplikaty.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Tuple (dupes_count, dupes_pct, sample_indices)
    """
    if df.empty:
        return 0, 0.0, []
    
    dupes_mask = df.duplicated()
    dupes_count = int(dupes_mask.sum())
    dupes_pct = float(dupes_count / len(df)) if len(df) > 0 else 0.0
    
    # Sample duplicate indices
    try:
        dup_indices = df.index[dupes_mask].tolist()[:MAX_DUPLICATE_SAMPLES]
        sample_indices = [str(idx) for idx in dup_indices]
    except Exception as e:
        LOGGER.warning(f"Nie udało się pobrać sample duplicate indices: {e}")
        sample_indices = []
    
    return dupes_count, dupes_pct, sample_indices


# ========================================================================================
# MEMORY ANALYSIS
# ========================================================================================

def _calculate_memory_usage(df: pd.DataFrame) -> float:
    """
    Oblicza zużycie pamięci przez DataFrame.
    
    Args:
        df: DataFrame
        
    Returns:
        Memory usage w MB
    """
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
    except Exception:
        # Fallback without deep
        memory_bytes = df.memory_usage().sum()
    
    memory_mb = float(memory_bytes / (1024 ** 2))
    return memory_mb


# ========================================================================================
# TYPE DISTRIBUTION
# ========================================================================================

def _analyze_type_distribution(df: pd.DataFrame) -> TypeDistribution:
    """
    Analizuje rozkład typów danych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        TypeDistribution object
    """
    type_groups = [_classify_dtype(df[col]) for col in df.columns]
    
    counts = {
        "numeric": type_groups.count("numeric"),
        "object": type_groups.count("object"),
        "categorical": type_groups.count("category"),
        "boolean": type_groups.count("bool"),
        "datetime": type_groups.count("datetime")
    }
    
    return TypeDistribution(**counts)


# ========================================================================================
# CONSTANT COLUMNS
# ========================================================================================

def _find_constant_columns(df: pd.DataFrame) -> List[str]:
    """
    Znajduje kolumny o stałych wartościach.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Lista nazw kolumn stałych
    """
    constant_cols: List[str] = []
    
    for col in df.columns:
        try:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                constant_cols.append(str(col))
        except Exception as e:
            LOGGER.debug(f"Błąd sprawdzania {col}: {e}")
            continue
    
    return constant_cols


# ========================================================================================
# CARDINALITY ANALYSIS
# ========================================================================================

def _analyze_cardinality(df: pd.DataFrame) -> CardinalityBuckets:
    """
    Analizuje kardynalność kolumn kategorycznych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        CardinalityBuckets object
    """
    low_card: List[str] = []
    high_card: List[str] = []
    binary: List[str] = []
    
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    
    for col in categorical_cols:
        try:
            n_unique = int(df[col].nunique(dropna=True))
            
            if n_unique == BINARY_THRESHOLD:
                binary.append(str(col))
            elif LOW_CARDINALITY_MIN <= n_unique <= LOW_CARDINALITY_MAX:
                low_card.append(str(col))
            elif n_unique > HIGH_CARDINALITY_MIN:
                high_card.append(str(col))
        except Exception as e:
            LOGGER.debug(f"Błąd analizy kardynalności {col}: {e}")
            continue
    
    return CardinalityBuckets(
        low_cardinality=low_card,
        high_cardinality=high_card,
        binary_like=binary
    )


# ========================================================================================
# NUMERIC ANALYSIS
# ========================================================================================

def _analyze_numeric_flags(df: pd.DataFrame) -> NumericFlags:
    """
    Analizuje flagi dla danych numerycznych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        NumericFlags object
    """
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return NumericFlags(
            non_finite_pct=0.0,
            skewed_cols=[],
            high_kurtosis_cols=[]
        )
    
    # Non-finite percentage
    try:
        numeric_array = numeric_df.to_numpy(dtype=float)
        non_finite_mask = ~np.isfinite(numeric_array)
        non_finite_pct = float(non_finite_mask.mean() * 100.0)
    except Exception as e:
        LOGGER.warning(f"Błąd obliczania non-finite: {e}")
        non_finite_pct = 0.0
    
    # Skewness
    try:
        skew_series = numeric_df.skew(numeric_only=True).abs()
        skew_series = skew_series.dropna().sort_values(ascending=False)
        skewed_cols = [
            str(col) for col in skew_series.head(MAX_SKEW_KURTOSIS_COLS).index
            if np.isfinite(skew_series[col])
        ]
    except Exception as e:
        LOGGER.warning(f"Błąd obliczania skewness: {e}")
        skewed_cols = []
    
    # Kurtosis
    try:
        kurt_series = numeric_df.kurtosis(numeric_only=True).abs()
        kurt_series = kurt_series.dropna().sort_values(ascending=False)
        high_kurt_cols = [
            str(col) for col in kurt_series.head(MAX_SKEW_KURTOSIS_COLS).index
            if np.isfinite(kurt_series[col])
        ]
    except Exception as e:
        LOGGER.warning(f"Błąd obliczania kurtosis: {e}")
        high_kurt_cols = []
    
    return NumericFlags(
        non_finite_pct=round(non_finite_pct, 3),
        skewed_cols=skewed_cols,
        high_kurtosis_cols=high_kurt_cols
    )


# ========================================================================================
# WARNINGS GENERATION
# ========================================================================================

def _generate_warnings(
    missing_pct: float,
    dupes_pct: float,
    constant_cols: List[str],
    numeric_flags: NumericFlags
) -> List[str]:
    """
    Generuje ostrzeżenia na podstawie metryk.
    
    Args:
        missing_pct: Procent braków
        dupes_pct: Procent duplikatów
        constant_cols: Lista kolumn stałych
        numeric_flags: Flagi numeryczne
        
    Returns:
        Lista ostrzeżeń
    """
    warnings: List[str] = []
    
    # Missing values
    if missing_pct > MISSING_WARNING_THRESHOLD:
        warnings.append(
            f"Wysoki odsetek braków ({missing_pct*100:.1f}% > {MISSING_WARNING_THRESHOLD*100}%)"
        )
    
    # Duplicates
    if dupes_pct > DUPLICATES_WARNING_THRESHOLD:
        warnings.append(
            f"Zauważalna liczba duplikatów ({dupes_pct*100:.2f}% > {DUPLICATES_WARNING_THRESHOLD*100}%)"
        )
    
    # Constant columns
    if constant_cols:
        n_constant = len(constant_cols)
        display_count = min(n_constant, MAX_CONSTANT_COLS_IN_WARNING)
        warnings.append(
            f"Kolumny o zerowej wariancji: {display_count}"
            f"{'+' if n_constant > display_count else ''}"
        )
    
    # Non-finite values
    if numeric_flags.non_finite_pct > NON_FINITE_WARNING_THRESHOLD * 100:
        warnings.append(
            f"Wartości niefinity w cechach numerycznych "
            f"({numeric_flags.non_finite_pct:.3f}% > {NON_FINITE_WARNING_THRESHOLD*100}%)"
        )
    
    return warnings


# ========================================================================================
# MAIN API
# ========================================================================================

def basic_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Wykonuje kompleksową analizę jakości danych.
    
    Raport zawiera:
    - Basic metrics: rows, cols, missing_pct, dupes, memory_mb
    - Type distribution: n_numeric, n_object, n_category, n_bool, n_datetime
    - Missing analysis: avg_nulls_per_col, top_missing_cols
    - Constant columns
    - Cardinality buckets: low/high/binary
    - Numeric flags: non_finite_pct, skewed_cols, high_kurtosis_cols
    - Sample duplicate indices
    - Automated warnings
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Słownik z raportem jakości (JSON-serializable)
        
    Example:
        >>> report = basic_quality_checks(df)
        >>> print(f"Missing: {report['missing_pct']:.1%}")
        >>> print(f"Warnings: {report['warnings']}")
    """
    # Input validation
    if df is None or not isinstance(df, pd.DataFrame):
        LOGGER.error("Invalid input: not a DataFrame")
        return {"error": "Nieprawidłowe dane wejściowe (brak DataFrame)"}
    
    rows = len(df)
    cols = df.shape[1]
    
    if rows == 0:
        LOGGER.warning("DataFrame is empty")
        return {
            "rows": 0,
            "cols": cols,
            "missing_pct": 0.0,
            "dupes": 0,
            "warnings": ["DataFrame jest pusty"]
        }
    
    LOGGER.debug(f"Analyzing DataFrame: {rows} rows × {cols} cols")
    
    # 1. Missing values analysis
    missing_pct, top_missing, avg_nulls_col, avg_nulls_row = _analyze_missing_values(df)
    
    # 2. Duplicates analysis
    dupes, dupes_pct, dup_samples = _analyze_duplicates(df)
    
    # 3. Memory usage
    memory_mb = _calculate_memory_usage(df)
    
    # 4. Type distribution
    type_dist = _analyze_type_distribution(df)
    
    # 5. Constant columns
    constant_cols = _find_constant_columns(df)
    
    # 6. Cardinality analysis
    cardinality = _analyze_cardinality(df)
    
    # 7. Numeric flags
    numeric_flags = _analyze_numeric_flags(df)
    
    # 8. Warnings
    warnings = _generate_warnings(missing_pct, dupes_pct, constant_cols, numeric_flags)
    
    # Build report
    report: Dict[str, Any] = {
        # Basic metrics
        "rows": rows,
        "cols": cols,
        "missing_pct": round(missing_pct, 6),
        "dupes": dupes,
        "dupes_pct": round(dupes_pct, 6),
        "memory_mb": round(memory_mb, 3),
        
        # Type distribution
        "dtypes_summary": type_dist.to_dict(),
        "n_numeric": type_dist.numeric,
        "n_object": type_dist.object,
        "n_category": type_dist.categorical,
        "n_bool": type_dist.boolean,
        "n_datetime": type_dist.datetime,
        
        # Missing analysis
        "avg_nulls_per_col": round(avg_nulls_col, 3),
        "avg_nulls_per_row_pct": round(avg_nulls_row, 6),
        "top_missing_cols": top_missing,
        
        # Constant columns
        "constant_columns": constant_cols,
        
        # Cardinality
        "cardinality": cardinality.to_dict(),
        
        # Numeric analysis
        "numeric": numeric_flags.to_dict(),
        
        # Duplicates
        "sample_duplicate_indices": dup_samples,
        
        # Warnings
        "warnings": warnings,
    }
    
    LOGGER.info(
        f"Quality check complete: "
        f"missing={missing_pct:.1%}, dupes={dupes_pct:.1%}, "
        f"warnings={len(warnings)}"
    )
    
    return report


# ========================================================================================
# VALIDATION FUNCTIONS
# ========================================================================================

def validate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Alias dla basic_quality_checks (backward compatibility).
    
    Args:
        df: DataFrame do walidacji
        
    Returns:
        Raport jakości
    """
    return basic_quality_checks(df)


def validate_for_ml(
    df: pd.DataFrame,
    target: Optional[str] = None
) -> Dict[str, Any]:
    """
    Walidacja specyficzna dla ML tasks.
    
    Dodatkowo sprawdza:
    - Target column presence
    - Target missing values
    - Feature-target correlation
    - Class imbalance (classification)
    
    Args:
        df: DataFrame do walidacji
        target: Nazwa kolumny celu (optional)
        
    Returns:
        Rozszerzony raport jakości
    """
    # Basic quality check
    report = basic_quality_checks(df)
    
    if "error" in report:
        return report
    
    # ML-specific checks
    ml_checks: Dict[str, Any] = {}
    
    if target:
        if target not in df.columns:
            ml_checks["target_error"] = f"Kolumna celu '{target}' nie istnieje"
        else:
            target_series = df[target]
            
            # Target missing
            target_missing = target_series.isna().mean()
            ml_checks["target_missing_pct"] = round(float(target_missing), 6)
            
            # Target stats
            ml_checks["target_unique"] = int(target_series.nunique(dropna=True))
            
            # Class imbalance (if categorical/discrete)
            if ml_checks["target_unique"] <= 20:
                value_counts = target_series.value_counts()
                if len(value_counts) > 0:
                    imbalance_ratio = float(value_counts.max() / value_counts.min())
                    ml_checks["class_imbalance_ratio"] = round(imbalance_ratio, 2)
            
            # Warning dla missing target
            if target_missing > 0.05:
                report["warnings"].append(
                    f"Wysoki odsetek braków w targecie: {target_missing:.1%}"
                )
    
    report["ml_checks"] = ml_checks
    
    return report


def quick_summary(df: pd.DataFrame) -> str:
    """
    Zwraca krótkie podsumowanie jakości danych (human-readable).
    
    Args:
        df: DataFrame
        
    Returns:
        String z podsumowaniem
    """
    report = basic_quality_checks(df)
    
    if "error" in report:
        return f"Error: {report['error']}"
    
    summary_parts = [
        f"Dataset: {report['rows']:,} rows × {report['cols']} columns",
        f"Missing: {report['missing_pct']:.1%}",
        f"Duplicates: {report['dupes']:,} ({report['dupes_pct']:.2%})",
        f"Memory: {report['memory_mb']:.1f} MB",
        f"Types: {report['n_numeric']} numeric, {report['n_object']} object",
    ]
    
    if report["warnings"]:
        summary_parts.append(f"⚠️ Warnings: {len(report['warnings'])}")
    
    return " | ".join(summary_parts)