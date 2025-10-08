"""
Data Cleaner PRO - Inteligentne czyszczenie i normalizacja danych.

Funkcjonalności:
- Automatyczna detekcja i konwersja typów danych
- Wykrywanie i parsowanie dat (wiele formatów)
- Konwersja kolumn binarnych (bool/int)
- Normalizacja stringów i usuwanie duplikatów
- Inteligentna imputacja braków
- Zaawansowane wykrywanie outlierów
- Raportowanie zmian
- Thread-safe operations
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

logger = logging.getLogger(__name__)

# Limity bezpieczeństwa
MAX_SAMPLE_SIZE = 10_000
DEFAULT_DATE_SAMPLE = 200
DEFAULT_DATE_THRESHOLD = 0.65

# Strategie imputacji
ImputeStrategy = Literal["median", "mean", "mode", "ffill", "bfill"]


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class CleanOptions:
    """Opcje czyszczenia danych."""
    # Detekcja dat
    sample_size_dates: int = DEFAULT_DATE_SAMPLE
    date_hit_threshold: float = DEFAULT_DATE_THRESHOLD
    
    # Binarne
    detect_binary: bool = True
    binary_bool_preferred: bool = True  # True -> BooleanDtype, False -> Int8
    
    # Podstawowe czyszczenie
    cast_numeric: bool = True
    trim_strings: bool = True
    drop_duplicates: bool = True
    replace_inf_with_nan: bool = True
    
    # Imputacja
    impute_missing: bool = True
    impute_numeric_strategy: ImputeStrategy = "median"
    impute_object_token: str = "<missing>"
    impute_categorical_token: str = "<unknown>"
    impute_bool_value: bool = False
    
    # Outliers (opcjonalne)
    detect_outliers: bool = False
    outlier_std_threshold: float = 3.0
    
    # Performance
    chunk_size: Optional[int] = None  # dla dużych DataFrame


@dataclass(frozen=True)
class CleanReport:
    """Raport z czyszczenia danych."""
    shape_initial: Tuple[int, int]
    shape_final: Tuple[int, int]
    duplicates_removed: int
    date_columns_detected: List[str]
    binary_columns_converted: List[str]
    numeric_columns_casted: List[str]
    imputed_columns: Dict[str, str]
    outliers_detected: Dict[str, int]
    na_total_before: int
    na_total_after: int
    notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje raport do słownika."""
        return asdict(self)
    
    def summary(self) -> str:
        """Zwraca tekstowe podsumowanie."""
        lines = [
            f"📊 Clean Report Summary",
            f"Shape: {self.shape_initial} → {self.shape_final}",
            f"Duplicates removed: {self.duplicates_removed}",
            f"Dates detected: {len(self.date_columns_detected)}",
            f"Binary converted: {len(self.binary_columns_converted)}",
            f"Numeric casted: {len(self.numeric_columns_casted)}",
            f"Columns imputed: {len(self.imputed_columns)}",
            f"Missing values: {self.na_total_before} → {self.na_total_after}",
        ]
        
        if self.notes:
            lines.append(f"\nNotes ({len(self.notes)}):")
            for note in self.notes[:5]:  # Max 5 notes
                lines.append(f"  - {note}")
        
        return "\n".join(lines)


# ========================================================================================
# DATE DETECTION
# ========================================================================================

# Wzorce dat (od najbardziej do najmniej specyficznych)
_DATE_PATTERNS = [
    # ISO formats
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",      # 2024-09-01T12:30:45
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",      # 2024-09-01 12:30:45
    r"^\d{4}-\d{2}-\d{2}",                         # 2024-09-01
    r"^\d{4}/\d{2}/\d{2}",                         # 2024/09/01
    
    # European formats
    r"^\d{2}[-/.]\d{2}[-/.]\d{4}",                 # 01-09-2024, 01.09.2024
    r"^\d{2}[-/.]\d{2}[-/.]\d{2}",                 # 01-09-24
    
    # American formats
    r"^\d{2}/\d{2}/\d{4}",                         # 09/01/2024
    
    # Compact formats
    r"^\d{8}$",                                    # 20240901
    r"^\d{14}$",                                   # 20240901123045
    
    # Timestamps
    r"^\d{10}$",                                   # Unix timestamp (seconds)
    r"^\d{13}$",                                   # Unix timestamp (milliseconds)
    
    # ISO week dates
    r"^\d{4}-W\d{2}-\d$",                          # 2024-W35-1
    
    # Year-Month only
    r"^\d{4}-\d{2}$",                              # 2024-09
    r"^\d{4}/\d{2}$",                              # 2024/09
]

# Kompilacja patterns dla wydajności
_COMPILED_DATE_PATTERNS = [re.compile(p) for p in _DATE_PATTERNS]


def _is_likely_date_string(value: str) -> bool:
    """
    Szybkie sprawdzenie czy string wygląda jak data.
    
    Args:
        value: String do sprawdzenia
        
    Returns:
        True jeśli prawdopodobnie data
    """
    if not value or len(value) > 30:  # Dates rzadko > 30 znaków
        return False
    
    # Szybkie sprawdzenie: czy zawiera cyfry i separatory
    if not any(c.isdigit() for c in value):
        return False
    
    for pattern in _COMPILED_DATE_PATTERNS:
        if pattern.match(value):
            return True
    
    return False


def _detect_date_columns(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_DATE_SAMPLE,
    hit_threshold: float = DEFAULT_DATE_THRESHOLD
) -> List[str]:
    """
    Wykrywa kolumny zawierające daty.
    
    Args:
        df: DataFrame do analizy
        sample_size: Rozmiar próbki do testowania
        hit_threshold: Próg trafień (0.0-1.0)
        
    Returns:
        Lista nazw kolumn z datami
    """
    if df.empty:
        return []
    
    candidates: List[str] = []
    
    # Tylko kolumny object/string
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    
    for col in obj_cols:
        series = df[col].dropna()
        
        if series.empty:
            continue
        
        # Próbka (deterministyczna - head)
        actual_sample_size = min(sample_size, len(series))
        sample = series.head(actual_sample_size).astype(str)
        
        if sample.empty:
            continue
        
        # Zlicz trafienia
        hits = sum(1 for val in sample if _is_likely_date_string(val.strip()))
        
        hit_rate = hits / len(sample)
        
        if hit_rate >= hit_threshold:
            candidates.append(col)
            logger.debug(f"Date column detected: '{col}' ({hit_rate:.1%} match rate)")
    
    return candidates


# ========================================================================================
# BINARY DETECTION
# ========================================================================================

# Mapowania wartości binarnych
_TRUE_TOKENS = {"1", "true", "t", "y", "yes", "tak", "prawda", "on", "enabled"}
_FALSE_TOKENS = {"0", "false", "f", "n", "no", "nie", "fałsz", "off", "disabled"}
_ALL_BINARY_TOKENS = _TRUE_TOKENS | _FALSE_TOKENS


def _is_binary_column(series: pd.Series) -> Tuple[bool, Optional[str]]:
    """
    Sprawdza czy kolumna jest binarna.
    
    Args:
        series: Seria do sprawdzenia
        
    Returns:
        Tuple (is_binary, binary_type)
        binary_type: "numeric" | "boolean" | "text" | None
    """
    # Skip już numeric/bool
    if ptypes.is_bool_dtype(series):
        return False, None
    
    if ptypes.is_integer_dtype(series) or ptypes.is_float_dtype(series):
        # Sprawdź czy tylko 0/1
        unique_vals = set(series.dropna().unique())
        if unique_vals and unique_vals <= {0, 1, 0.0, 1.0}:
            return True, "numeric"
        return False, None
    
    # String/object
    if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
        lower_vals = series.astype(str).str.strip().str.lower()
        unique_vals = set(lower_vals.dropna().unique())
        
        if not unique_vals:
            return False, None
        
        # Czysty 0/1?
        if unique_vals <= {"0", "1"}:
            return True, "numeric"
        
        # Boolean tokens?
        if unique_vals <= _ALL_BINARY_TOKENS:
            return True, "boolean"
    
    return False, None


def _convert_binary_column(
    series: pd.Series,
    binary_type: str,
    prefer_boolean: bool
) -> pd.Series:
    """
    Konwertuje kolumnę binarną.
    
    Args:
        series: Seria do konwersji
        binary_type: Typ binarny ("numeric" lub "boolean")
        prefer_boolean: Czy preferować BooleanDtype
        
    Returns:
        Skonwertowana seria
    """
    if binary_type == "numeric":
        lower_vals = series.astype(str).str.strip().str.lower()
        
        if prefer_boolean:
            return lower_vals.map({
                "0": False, "0.0": False,
                "1": True, "1.0": True
            }).astype("boolean")
        else:
            return lower_vals.map({
                "0": 0, "0.0": 0,
                "1": 1, "1.0": 1
            }).astype("Int8")
    
    elif binary_type == "boolean":
        lower_vals = series.astype(str).str.strip().str.lower()
        
        def map_bool(val):
            if pd.isna(val) or val == "nan":
                return pd.NA
            if val in _TRUE_TOKENS:
                return True
            if val in _FALSE_TOKENS:
                return False
            return pd.NA
        
        return lower_vals.map(map_bool).astype("boolean")
    
    return series


def _convert_binaries(
    df: pd.DataFrame,
    prefer_boolean: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Konwertuje wszystkie kolumny binarne.
    
    Args:
        df: DataFrame
        prefer_boolean: Czy preferować BooleanDtype
        
    Returns:
        Tuple (converted_df, list_of_converted_columns)
    """
    if df.empty:
        return df, []
    
    converted = df.copy()
    converted_cols: List[str] = []
    
    for col in converted.columns:
        is_binary, binary_type = _is_binary_column(converted[col])
        
        if is_binary and binary_type:
            try:
                converted[col] = _convert_binary_column(
                    converted[col],
                    binary_type,
                    prefer_boolean
                )
                converted_cols.append(col)
                logger.debug(f"Binary conversion: '{col}' → {binary_type}")
                
            except Exception as e:
                logger.warning(f"Failed to convert binary column '{col}': {e}")
    
    return converted, converted_cols


# ========================================================================================
# NUMERIC CASTING
# ========================================================================================

def _safe_cast_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    """
    Bezpiecznie próbuje scastować do numeric.
    
    Args:
        series: Seria do konwersji
        
    Returns:
        Tuple (converted_series, was_converted)
    """
    # Skip już numeric
    if ptypes.is_numeric_dtype(series):
        return series, False
    
    # Skip datetime
    if ptypes.is_datetime64_any_dtype(series):
        return series, False
    
    # Skip categorical (może mieć numeric codes ale to nie to samo)
    if ptypes.is_categorical_dtype(series):
        return series, False
    
    # Tylko object/string
    if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)):
        return series, False
    
    try:
        # Próba konwersji z pd.to_numeric
        converted = pd.to_numeric(series, errors="coerce")
        
        # Sprawdź czy konwersja ma sens (nie wszystko NaN)
        if converted.notna().sum() == 0:
            return series, False
        
        # Sprawdź czy > 50% wartości się skonwertowało
        conversion_rate = converted.notna().sum() / len(series)
        
        if conversion_rate < 0.5:
            return series, False
        
        # Optymalizuj typ (int jeśli możliwe)
        if converted.dropna().apply(lambda x: x == int(x)).all():
            # Wszystkie wartości to inty
            try:
                converted = converted.astype("Int64")
            except Exception:
                pass
        
        return converted, True
        
    except Exception as e:
        logger.debug(f"Numeric cast failed: {e}")
        return series, False


def _cast_numerics(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Próbuje scastować kolumny do numeric.
    
    Args:
        df: DataFrame
        
    Returns:
        Tuple (converted_df, list_of_converted_columns)
    """
    converted = df.copy()
    converted_cols: List[str] = []
    
    for col in converted.columns:
        series, was_converted = _safe_cast_numeric(converted[col])
        
        if was_converted:
            converted[col] = series
            converted_cols.append(col)
            logger.debug(f"Numeric cast: '{col}' → {series.dtype}")
    
    return converted, converted_cols


# ========================================================================================
# OUTLIER DETECTION
# ========================================================================================

def _detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Wykrywa outliery metodą Z-score.
    
    Args:
        series: Seria numeryczna
        threshold: Próg (typowo 3.0)
        
    Returns:
        Boolean series (True = outlier)
    """
    if not ptypes.is_numeric_dtype(series):
        return pd.Series([False] * len(series), index=series.index)
    
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series([False] * len(series), index=series.index)
    
    z_scores = np.abs((series - mean) / std)
    
    return z_scores > threshold


def _detect_all_outliers(
    df: pd.DataFrame,
    threshold: float = 3.0
) -> Dict[str, int]:
    """
    Wykrywa outliery we wszystkich kolumnach numerycznych.
    
    Args:
        df: DataFrame
        threshold: Próg Z-score
        
    Returns:
        Dict {column_name: outlier_count}
    """
    outliers: Dict[str, int] = {}
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        outlier_mask = _detect_outliers_zscore(df[col], threshold)
        count = int(outlier_mask.sum())
        
        if count > 0:
            outliers[col] = count
    
    return outliers


# ========================================================================================
# IMPUTACJA
# ========================================================================================

def _impute_numeric(
    series: pd.Series,
    strategy: ImputeStrategy
) -> Tuple[pd.Series, str]:
    """
    Imputuje braki w kolumnie numerycznej.
    
    Args:
        series: Seria numeryczna z brakami
        strategy: Strategia imputacji
        
    Returns:
        Tuple (imputed_series, description)
    """
    if strategy == "median":
        fill_val = series.median(skipna=True)
        return series.fillna(fill_val), f"median={fill_val:.2f}"
    
    elif strategy == "mean":
        fill_val = series.mean(skipna=True)
        return series.fillna(fill_val), f"mean={fill_val:.2f}"
    
    elif strategy == "mode":
        mode_vals = series.mode()
        if len(mode_vals) > 0:
            fill_val = mode_vals.iloc[0]
            return series.fillna(fill_val), f"mode={fill_val:.2f}"
        else:
            return series, "mode=N/A"
    
    elif strategy == "ffill":
        return series.fillna(method="ffill"), "ffill"
    
    elif strategy == "bfill":
        return series.fillna(method="bfill"), "bfill"
    
    else:
        # Default: median
        fill_val = series.median(skipna=True)
        return series.fillna(fill_val), f"median={fill_val:.2f}"


def _impute_missing(
    df: pd.DataFrame,
    opts: CleanOptions
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Imputuje braki we wszystkich kolumnach.
    
    Args:
        df: DataFrame z brakami
        opts: Opcje czyszczenia
        
    Returns:
        Tuple (imputed_df, dict_of_imputed_columns)
    """
    imputed = df.copy()
    imputed_cols: Dict[str, str] = {}
    
    for col in imputed.columns:
        series = imputed[col]
        
        if not series.isna().any():
            continue
        
        try:
            # Numeric
            if ptypes.is_numeric_dtype(series):
                imputed[col], desc = _impute_numeric(series, opts.impute_numeric_strategy)
                imputed_cols[col] = desc
            
            # Categorical
            elif ptypes.is_categorical_dtype(series):
                token = opts.impute_categorical_token
                
                # Dodaj kategorię jeśli nie istnieje
                if token not in series.cat.categories:
                    imputed[col] = series.cat.add_categories([token])
                
                imputed[col] = imputed[col].fillna(token)
                imputed_cols[col] = token
            
            # Boolean
            elif ptypes.is_bool_dtype(series):
                imputed[col] = series.fillna(opts.impute_bool_value)
                imputed_cols[col] = str(opts.impute_bool_value)
            
            # String/Object
            elif ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series):
                token = opts.impute_object_token
                imputed[col] = series.fillna(token)
                imputed_cols[col] = token
            
            # Datetime - skip imputacji
            else:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to impute column '{col}': {e}")
    
    return imputed, imputed_cols


# ========================================================================================
# PUBLIC API
# ========================================================================================

def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inteligentne czyszczenie danych (backward compatible API).
    
    Zachowuje podpis funkcji z oryginalnego kodu.
    Raport dostępny jako df.clean_report po zwróceniu.
    
    Args:
        df: DataFrame do wyczyszczenia
        
    Returns:
        Wyczyszczony DataFrame z atrybutem .clean_report
    """
    df_clean, report = quick_clean_pro(df)
    
    # Dołącz raport jako atrybut
    try:
        setattr(df_clean, "clean_report", report.to_dict())
    except Exception as e:
        logger.warning(f"Failed to attach clean_report: {e}")
    
    return df_clean


def quick_clean_pro(
    df: pd.DataFrame,
    opts: Optional[CleanOptions] = None
) -> Tuple[pd.DataFrame, CleanReport]:
    """
    Wersja PRO czyszczenia danych z pełnym raportem.
    
    Args:
        df: DataFrame do wyczyszczenia
        opts: Opcje czyszczenia (domyślne jeśli None)
        
    Returns:
        Tuple (cleaned_df, CleanReport)
        
    Raises:
        TypeError: Jeśli df nie jest DataFrame
        ValueError: Jeśli df jest pusty
    """
    # Walidacja
    if not isinstance(df, pd.DataFrame):
        raise TypeError("quick_clean_pro: df must be a pandas.DataFrame")
    
    if df.empty:
        logger.warning("quick_clean_pro: DataFrame is empty")
        empty_report = CleanReport(
            shape_initial=(0, 0),
            shape_final=(0, 0),
            duplicates_removed=0,
            date_columns_detected=[],
            binary_columns_converted=[],
            numeric_columns_casted=[],
            imputed_columns={},
            outliers_detected={},
            na_total_before=0,
            na_total_after=0,
            notes=["DataFrame was empty"]
        )
        return df.copy(), empty_report
    
    # Opcje
    if opts is None:
        opts = CleanOptions()
    
    # Inicjalizacja
    notes: List[str] = []
    shape_initial = df.shape
    na_total_before = int(df.isna().sum().sum())
    
    # Kopia (deep copy dla bezpieczeństwa)
    df_clean = df.copy(deep=True)
    
    logger.info(f"Starting quick_clean_pro on DataFrame {shape_initial}")
    
    # ============================================================================
    # ETAP 1: DUPLIKATY
    # ============================================================================
    
    duplicates_removed = 0
    
    if opts.drop_duplicates:
        dup_count = int(df_clean.duplicated().sum())
        
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = dup_count
            notes.append(f"Removed {dup_count:,} duplicate rows")
            logger.debug(f"Duplicates removed: {dup_count}")
    
    # ============================================================================
    # ETAP 2: NORMALIZACJA STRINGÓW
    # ============================================================================
    
    if opts.trim_strings:
        obj_cols = df_clean.select_dtypes(include=["object", "string"]).columns
        
        for col in obj_cols:
            try:
                # Konwersja do string type
                series = df_clean[col].astype("string")
                
                # Strip whitespace
                series = series.str.strip()
                
                # Replace empty strings i common null representations
                series = series.replace({
                    "": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "none": pd.NA,
                    "None": pd.NA,
                    "null": pd.NA,
                    "NULL": pd.NA,
                }, regex=False)
                
                df_clean[col] = series
                
            except Exception as e:
                logger.warning(f"Failed to trim strings in '{col}': {e}")
        
        if len(obj_cols) > 0:
            notes.append(f"Trimmed strings in {len(obj_cols)} columns")
    
    # ============================================================================
    # ETAP 3: KONWERSJA NUMERIC
    # ============================================================================
    
    numeric_casted: List[str] = []
    
    if opts.cast_numeric:
        try:
            df_clean, numeric_casted = _cast_numerics(df_clean)
            
            if numeric_casted:
                notes.append(f"Casted {len(numeric_casted)} columns to numeric")
                logger.debug(f"Numeric columns casted: {numeric_casted}")
                
        except Exception as e:
            logger.error(f"Numeric casting failed: {e}")
            notes.append("Numeric casting encountered errors")
    
    # ============================================================================
    # ETAP 4: DETEKCJA I KONWERSJA DAT
    # ============================================================================
    
    date_cols: List[str] = []
    
    try:
        date_cols = _detect_date_columns(
            df_clean,
            sample_size=min(opts.sample_size_dates, MAX_SAMPLE_SIZE),
            hit_threshold=opts.date_hit_threshold
        )
        
        for col in date_cols:
            try:
                # Konwersja z errors='coerce' (invalid → NaT)
                df_clean[col] = pd.to_datetime(
                    df_clean[col],
                    errors="coerce",
                    utc=False
                )
                logger.debug(f"Date conversion: '{col}'")
                
            except Exception as e:
                logger.warning(f"Failed to convert '{col}' to datetime: {e}")
                date_cols.remove(col)
        
        if date_cols:
            notes.append(f"Detected and converted {len(date_cols)} date columns")
            
    except Exception as e:
        logger.error(f"Date detection failed: {e}")
        notes.append("Date detection encountered errors")
    
    # ============================================================================
    # ETAP 5: KONWERSJA BINARNE
    # ============================================================================
    
    binary_converted: List[str] = []
    
    if opts.detect_binary:
        try:
            df_clean, binary_converted = _convert_binaries(
                df_clean,
                prefer_boolean=opts.binary_bool_preferred
            )
            
            if binary_converted:
                notes.append(f"Converted {len(binary_converted)} binary columns")
                logger.debug(f"Binary columns: {binary_converted}")
                
        except Exception as e:
            logger.error(f"Binary conversion failed: {e}")
            notes.append("Binary conversion encountered errors")
    
    # ============================================================================
    # ETAP 6: INF → NaN
    # ============================================================================
    
    if opts.replace_inf_with_nan:
        try:
            # Replace inf w numeric columns
            numeric_cols = df_clean.select_dtypes(include=np.number).columns
            
            for col in numeric_cols:
                inf_count = np.isinf(df_clean[col]).sum()
                
                if inf_count > 0:
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], pd.NA)
                    logger.debug(f"Replaced {inf_count} inf values in '{col}'")
            
        except Exception as e:
            logger.warning(f"Inf replacement failed: {e}")
    
    # ============================================================================
    # ETAP 7: DETEKCJA OUTLIERÓW (opcjonalne)
    # ============================================================================
    
    outliers_detected: Dict[str, int] = {}
    
    if opts.detect_outliers:
        try:
            outliers_detected = _detect_all_outliers(
                df_clean,
                threshold=opts.outlier_std_threshold
            )
            
            if outliers_detected:
                total_outliers = sum(outliers_detected.values())
                notes.append(f"Detected {total_outliers} outliers across {len(outliers_detected)} columns")
                logger.debug(f"Outliers: {outliers_detected}")
                
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            notes.append("Outlier detection encountered errors")
    
    # ============================================================================
    # ETAP 8: IMPUTACJA
    # ============================================================================
    
    imputed_cols: Dict[str, str] = {}
    
    if opts.impute_missing:
        try:
            df_clean, imputed_cols = _impute_missing(df_clean, opts)
            
            if imputed_cols:
                notes.append(f"Imputed missing values in {len(imputed_cols)} columns")
                logger.debug(f"Imputed columns: {list(imputed_cols.keys())}")
                
        except Exception as e:
            logger.error(f"Imputation failed: {e}")
            notes.append("Imputation encountered errors")
    
    # ============================================================================
    # FINALIZACJA
    # ============================================================================
    
    shape_final = df_clean.shape
    na_total_after = int(df_clean.isna().sum().sum())
    
    # Raport
    report = CleanReport(
        shape_initial=shape_initial,
        shape_final=shape_final,
        duplicates_removed=duplicates_removed,
        date_columns_detected=date_cols,
        binary_columns_converted=binary_converted,
        numeric_columns_casted=numeric_casted,
        imputed_columns=imputed_cols,
        outliers_detected=outliers_detected,
        na_total_before=na_total_before,
        na_total_after=na_total_after=na_total_after,
        notes=notes
    )
    
    logger.info(
        f"quick_clean_pro completed: {shape_initial} → {shape_final}, "
        f"NA: {na_total_before} → {na_total_after}"
    )
    
    # Dołącz raport jako atrybut (dla kompatybilności)
    try:
        setattr(df_clean, "clean_report", report.to_dict())
    except Exception as e:
        logger.debug(f"Failed to attach clean_report attribute: {e}")
    
    return df_clean, report


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def validate_clean_options(opts: CleanOptions) -> None:
    """
    Waliduje opcje czyszczenia.
    
    Args:
        opts: Opcje do walidacji
        
    Raises:
        ValueError: Jeśli opcje są nieprawidłowe
    """
    if opts.sample_size_dates < 1:
        raise ValueError("sample_size_dates must be >= 1")
    
    if not 0.0 <= opts.date_hit_threshold <= 1.0:
        raise ValueError("date_hit_threshold must be between 0.0 and 1.0")
    
    if opts.impute_numeric_strategy not in ("median", "mean", "mode", "ffill", "bfill"):
        raise ValueError(
            f"Invalid impute_numeric_strategy: {opts.impute_numeric_strategy}. "
            "Must be one of: median, mean, mode, ffill, bfill"
        )
    
    if opts.outlier_std_threshold <= 0:
        raise ValueError("outlier_std_threshold must be > 0")


def get_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca podsumowanie jakości danych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Słownik z metrykami jakości
    """
    if df.empty:
        return {
            "rows": 0,
            "cols": 0,
            "missing_pct": 0.0,
            "duplicates": 0,
            "memory_mb": 0.0,
            "dtypes": {}
        }
    
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing_pct": float(missing_cells / total_cells * 100) if total_cells > 0 else 0.0,
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict()
    }


def compare_dataframes(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> Dict[str, Any]:
    """
    Porównuje dwa DataFrame (przed i po czyszczeniu).
    
    Args:
        df_before: DataFrame przed czyszczeniem
        df_after: DataFrame po czyszczeniu
        
    Returns:
        Słownik z porównaniem
    """
    before_stats = get_data_quality_summary(df_before)
    after_stats = get_data_quality_summary(df_after)
    
    return {
        "rows_change": after_stats["rows"] - before_stats["rows"],
        "cols_change": after_stats["cols"] - before_stats["cols"],
        "missing_reduction_pct": before_stats["missing_pct"] - after_stats["missing_pct"],
        "duplicates_removed": before_stats["duplicates"] - after_stats["duplicates"],
        "memory_change_mb": after_stats["memory_mb"] - before_stats["memory_mb"],
        "before": before_stats,
        "after": after_stats
    }


def clean_with_validation(
    df: pd.DataFrame,
    opts: Optional[CleanOptions] = None,
    validate_opts: bool = True
) -> Tuple[pd.DataFrame, CleanReport]:
    """
    Czyści dane z walidacją opcji.
    
    Args:
        df: DataFrame do wyczyszczenia
        opts: Opcje czyszczenia
        validate_opts: Czy walidować opcje przed czyszczeniem
        
    Returns:
        Tuple (cleaned_df, CleanReport)
        
    Raises:
        ValueError: Jeśli opcje są nieprawidłowe
    """
    if opts is None:
        opts = CleanOptions()
    
    if validate_opts:
        validate_clean_options(opts)
    
    return quick_clean_pro(df, opts)


# ========================================================================================
# ADVANCED FEATURES
# ========================================================================================

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Wykrywa semantyczne typy kolumn (nie tylko pandas dtype).
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Dict {column_name: semantic_type}
        Typy: "numeric", "categorical", "datetime", "boolean", "text", "id", "unknown"
    """
    type_map: Dict[str, str] = {}
    
    for col in df.columns:
        series = df[col]
        
        # Datetime
        if ptypes.is_datetime64_any_dtype(series):
            type_map[col] = "datetime"
            continue
        
        # Boolean
        if ptypes.is_bool_dtype(series):
            type_map[col] = "boolean"
            continue
        
        # Numeric
        if ptypes.is_numeric_dtype(series):
            # ID detection (wysokie nunique, sekwencyjne)
            nunique = series.nunique()
            if nunique == len(series) and nunique > 100:
                # Potencjalnie ID
                if ptypes.is_integer_dtype(series):
                    # Sprawdź czy sekwencyjne
                    sorted_vals = series.dropna().sort_values()
                    if len(sorted_vals) > 1:
                        diffs = sorted_vals.diff().dropna()
                        if diffs.median() == 1:
                            type_map[col] = "id"
                            continue
            
            type_map[col] = "numeric"
            continue
        
        # Categorical
        if ptypes.is_categorical_dtype(series):
            type_map[col] = "categorical"
            continue
        
        # String/Object - dalsze badanie
        if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
            nunique = series.nunique()
            total = len(series)
            
            # ID candidate (wszystkie unique)
            if nunique == total and total > 100:
                type_map[col] = "id"
                continue
            
            # Low cardinality → categorical
            if nunique < total * 0.05 and nunique < 50:
                type_map[col] = "categorical"
                continue
            
            # Date detection
            is_date, _ = _is_binary_column(series)
            if not is_date:
                sample = series.dropna().head(100).astype(str)
                if len(sample) > 0:
                    date_matches = sum(1 for v in sample if _is_likely_date_string(v))
                    if date_matches / len(sample) > 0.6:
                        type_map[col] = "datetime"
                        continue
            
            # Default: text
            type_map[col] = "text"
            continue
        
        # Unknown
        type_map[col] = "unknown"
    
    return type_map


def suggest_cleaning_strategy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Sugeruje strategię czyszczenia na podstawie analizy danych.
    
    Args:
        df: DataFrame do analizy
        
    Returns:
        Słownik z sugestiami
    """
    quality = get_data_quality_summary(df)
    column_types = detect_column_types(df)
    
    suggestions = {
        "recommended_options": {},
        "warnings": [],
        "optimizations": []
    }
    
    # Duplikaty
    if quality["duplicates"] > 0:
        dup_pct = quality["duplicates"] / quality["rows"] * 100
        suggestions["recommended_options"]["drop_duplicates"] = True
        suggestions["warnings"].append(
            f"Found {quality['duplicates']:,} duplicates ({dup_pct:.1f}%)"
        )
    
    # Braki
    if quality["missing_pct"] > 5:
        suggestions["recommended_options"]["impute_missing"] = True
        suggestions["warnings"].append(
            f"High missing data rate: {quality['missing_pct']:.1f}%"
        )
        
        if quality["missing_pct"] > 30:
            suggestions["recommended_options"]["impute_numeric_strategy"] = "median"
        else:
            suggestions["recommended_options"]["impute_numeric_strategy"] = "mean"
    
    # Detekcja dat
    date_candidates = sum(1 for t in column_types.values() if t == "datetime")
    if date_candidates > 0:
        suggestions["optimizations"].append(
            f"Enable date detection (found {date_candidates} potential date columns)"
        )
    
    # Binarne
    binary_candidates = sum(
        1 for col in df.columns
        if _is_binary_column(df[col])[0]
    )
    if binary_candidates > 0:
        suggestions["recommended_options"]["detect_binary"] = True
        suggestions["optimizations"].append(
            f"Found {binary_candidates} binary columns - enable conversion"
        )
    
    # Pamięć
    if quality["memory_mb"] > 1000:
        suggestions["optimizations"].append(
            "Large dataset - consider enabling chunk processing"
        )
        suggestions["recommended_options"]["chunk_size"] = 10000
    
    # Outliery (jeśli dużo numeric)
    numeric_count = sum(1 for t in column_types.values() if t == "numeric")
    if numeric_count > 5:
        suggestions["optimizations"].append(
            f"Many numeric columns ({numeric_count}) - consider outlier detection"
        )
    
    return suggestions


def auto_clean(
    df: pd.DataFrame,
    aggressive: bool = False
) -> Tuple[pd.DataFrame, CleanReport]:
    """
    Automatyczne czyszczenie z inteligentnymi sugestiami.
    
    Args:
        df: DataFrame do wyczyszczenia
        aggressive: Czy używać agresywnych opcji (outliery, więcej imputacji)
        
    Returns:
        Tuple (cleaned_df, CleanReport)
    """
    # Analiza i sugestie
    suggestions = suggest_cleaning_strategy(df)
    
    # Buduj opcje na podstawie sugestii
    opts_dict = suggestions.get("recommended_options", {})
    
    # Dodaj agresywne opcje
    if aggressive:
        opts_dict["detect_outliers"] = True
        opts_dict["outlier_std_threshold"] = 2.5  # Bardziej czuły
    
    # Przekonwertuj na CleanOptions
    opts = CleanOptions(**opts_dict)
    
    logger.info(f"Auto-cleaning with options: {opts_dict}")
    
    return quick_clean_pro(df, opts)


# ========================================================================================
# EXPORT & REPORTING
# ========================================================================================

def export_clean_report(
    report: CleanReport,
    format: Literal["json", "markdown", "html"] = "json"
) -> str:
    """
    Eksportuje raport czyszczenia do różnych formatów.
    
    Args:
        report: CleanReport do eksportu
        format: Format wyjściowy
        
    Returns:
        Sformatowany string
    """
    if format == "json":
        import json
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    
    elif format == "markdown":
        lines = [
            "# Data Cleaning Report",
            "",
            "## Summary",
            f"- **Initial Shape**: {report.shape_initial[0]:,} rows × {report.shape_initial[1]} cols",
            f"- **Final Shape**: {report.shape_final[0]:,} rows × {report.shape_final[1]} cols",
            f"- **Duplicates Removed**: {report.duplicates_removed:,}",
            f"- **Missing Values**: {report.na_total_before:,} → {report.na_total_after:,}",
            "",
            "## Transformations",
            f"- **Date Columns Detected**: {len(report.date_columns_detected)}",
        ]
        
        if report.date_columns_detected:
            for col in report.date_columns_detected:
                lines.append(f"  - `{col}`")
        
        lines.append(f"- **Binary Columns Converted**: {len(report.binary_columns_converted)}")
        
        if report.binary_columns_converted:
            for col in report.binary_columns_converted:
                lines.append(f"  - `{col}`")
        
        lines.append(f"- **Numeric Columns Casted**: {len(report.numeric_columns_casted)}")
        
        if report.numeric_columns_casted:
            for col in report.numeric_columns_casted:
                lines.append(f"  - `{col}`")
        
        lines.append(f"- **Columns Imputed**: {len(report.imputed_columns)}")
        
        if report.imputed_columns:
            for col, strategy in report.imputed_columns.items():
                lines.append(f"  - `{col}`: {strategy}")
        
        if report.outliers_detected:
            lines.append("")
            lines.append("## Outliers Detected")
            for col, count in report.outliers_detected.items():
                lines.append(f"- `{col}`: {count:,} outliers")
        
        if report.notes:
            lines.append("")
            lines.append("## Notes")
            for note in report.notes:
                lines.append(f"- {note}")
        
        return "\n".join(lines)
    
    elif format == "html":
        html_parts = [
            "<html><head>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #2c3e50; }",
            "h2 { color: #34495e; margin-top: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #3498db; color: white; }",
            "code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            ".metric { display: inline-block; margin: 10px 20px 10px 0; }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }",
            ".metric-label { font-size: 12px; color: #7f8c8d; }",
            "</style>",
            "</head><body>",
            "<h1>📊 Data Cleaning Report</h1>",
            "<div class='metrics'>",
            f"<div class='metric'><div class='metric-value'>{report.shape_initial[0]:,}</div><div class='metric-label'>Initial Rows</div></div>",
            f"<div class='metric'><div class='metric-value'>{report.shape_final[0]:,}</div><div class='metric-label'>Final Rows</div></div>",
            f"<div class='metric'><div class='metric-value'>{report.duplicates_removed:,}</div><div class='metric-label'>Duplicates Removed</div></div>",
            f"<div class='metric'><div class='metric-value'>{report.na_total_after:,}</div><div class='metric-label'>Missing Values</div></div>",
            "</div>",
            "<h2>Transformations</h2>",
            "<table>",
            "<tr><th>Category</th><th>Count</th><th>Details</th></tr>",
            f"<tr><td>Date Columns</td><td>{len(report.date_columns_detected)}</td><td>{', '.join(report.date_columns_detected) if report.date_columns_detected else 'None'}</td></tr>",
            f"<tr><td>Binary Conversions</td><td>{len(report.binary_columns_converted)}</td><td>{', '.join(report.binary_columns_converted) if report.binary_columns_converted else 'None'}</td></tr>",
            f"<tr><td>Numeric Casts</td><td>{len(report.numeric_columns_casted)}</td><td>{', '.join(report.numeric_columns_casted) if report.numeric_columns_casted else 'None'}</td></tr>",
            f"<tr><td>Imputed Columns</td><td>{len(report.imputed_columns)}</td><td>{len(report.imputed_columns)} columns</td></tr>",
            "</table>",
        ]
        
        if report.notes:
            html_parts.append("<h2>Notes</h2><ul>")
            for note in report.notes:
                html_parts.append(f"<li>{note}</li>")
            html_parts.append("</ul>")
        
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


# ========================================================================================
# BACKWARDS COMPATIBILITY
# ========================================================================================

# Alias dla kompatybilności z poprzednim kodem
def smart_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias dla _cast_numerics (backward compatibility).
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame z scastowanymi kolumnami
    """
    converted, _ = _cast_numerics(df)
    return converted


# ========================================================================================
# MODULE INFO
# ========================================================================================

__all__ = [
    # Main API
    "quick_clean",
    "quick_clean_pro",
    "clean_with_validation",
    "auto_clean",
    
    # Options & Reports
    "CleanOptions",
    "CleanReport",
    
    # Utilities
    "validate_clean_options",
    "get_data_quality_summary",
    "compare_dataframes",
    "detect_column_types",
    "suggest_cleaning_strategy",
    "export_clean_report",
    
    # Backward compatibility
    "smart_cast_numeric",
]

__version__ = "2.0.0"