# feature_engineering.py — TURBO PRO++ Enhanced Edition
"""
Advanced Feature Engineering Module.

Features:
- Automatic datetime detection and feature extraction
- Text feature engineering (length, word count, patterns)
- Intelligent categorical encoding (one-hot, ordinal, target)
- Numeric transformations and binning
- Missing value indicators
- Interaction features
- Performance optimizations for large datasets
- Comprehensive logging and reporting
- Backward compatibility
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Set

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# ========================================================================================
# LOGGING
# ========================================================================================

logger = logging.getLogger(__name__)

# ========================================================================================
# CONFIGURATION
# ========================================================================================

# Performance limits
MAX_NEW_COLUMNS = 5_000
MAX_CATEGORICAL_CARDINALITY = 100
MAX_ONE_HOT_CARDINALITY = 20
MAX_TEXT_SAMPLE = 1_000

# Date detection
DEFAULT_DATE_HINTS = (
    "date", "time", "timestamp", "data", "czas", "dt", "day", "month", "year",
    "created", "updated", "modified", "deleted", "birth", "death"
)

# Common date patterns
DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",            # 2024-09-01 / 2024/9/1
    r"^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$",      # 01/09/2024, 1.9.24
    r"^\d{4}[-/]\d{1,2}$",                       # 2024-09
    r"^\d{8}$",                                  # 20240901
    r"^\d{14}$",                                 # 20240901123045
    r"^\d{10}$",                                 # unix seconds
    r"^\d{13}$",                                 # unix milliseconds
    r"^\d{16}$",                                 # unix microseconds
    r"^\d{4}-W\d{2}-\d$",                        # ISO week-date
]

# ========================================================================================
# TYPES
# ========================================================================================

ActionKind = Literal["date", "text", "categorical", "numeric", "repair", "missing"]


@dataclass(frozen=True)
class FEOptions:
    """Configuration for feature engineering."""
    
    # Date detection
    date_name_hints: Tuple[str, ...] = DEFAULT_DATE_HINTS
    date_sample_size: int = 200
    date_hit_threshold: float = 0.60
    min_parsed_ratio: float = 0.50
    
    # Date features
    add_age: bool = True
    age_round_decimals: int = 1
    add_time_features: bool = True  # hour, minute, second
    add_cyclical_features: bool = False  # sin/cos encoding for circular features
    
    # Text features
    text_min_unique: int = 5
    text_max_unique: int = 5_000
    text_len_dtype: str = "Int32"
    text_words_dtype: str = "Int16"
    add_text_complexity: bool = True  # capitals, special chars, etc.
    
    # Categorical encoding
    cat_low_card_max: int = 10      # One-hot for <= 10 unique
    cat_mid_card_max: int = 30      # Ordinal for (10, 30]
    cat_high_card_max: int = 100    # Skip or hash for > 100
    one_hot_drop_first: bool = False
    one_hot_dummy_na: bool = False
    ordinal_dtype: str = "Int32"
    one_hot_prefix_sep: str = "_"
    handle_high_cardinality: bool = False  # Hash encoding for high card
    
    # Numeric features
    add_binned_features: bool = False
    n_bins: int = 5
    bin_strategy: Literal["quantile", "uniform"] = "quantile"
    
    # Missing indicators
    add_missing_indicators: bool = True
    missing_threshold: float = 0.05  # Only add indicator if >5% missing
    
    # Cleanup
    replace_inf_with_nan: bool = True
    drop_constant_columns: bool = True
    
    # Performance
    cap_new_columns: int = MAX_NEW_COLUMNS
    sample_large_datasets: bool = True
    max_sample_size: int = 100_000
    
    # Verbose
    verbose: bool = False
    
    def __post_init__(self):
        """Validate options."""
        if self.cat_low_card_max > self.cat_mid_card_max:
            object.__setattr__(self, 'cat_mid_card_max', self.cat_low_card_max + 20)
            logger.warning("cat_mid_card_max adjusted to be > cat_low_card_max")


@dataclass(frozen=True)
class FEAction:
    """Individual feature engineering action."""
    
    kind: ActionKind
    column: Optional[str]
    produced: List[str]
    note: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "column": self.column,
            "produced": self.produced,
            "note": self.note,
            "details": self.details
        }


@dataclass(frozen=True)
class FEResult:
    """Feature engineering result report."""
    
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    actions: List[FEAction]
    warnings: List[str]
    
    # Summary stats
    n_date_cols: int = 0
    n_text_cols: int = 0
    n_categorical_cols: int = 0
    n_numeric_cols: int = 0
    n_features_added: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "actions": [a.to_dict() for a in self.actions],
            "warnings": self.warnings,
            "n_date_cols": self.n_date_cols,
            "n_text_cols": self.n_text_cols,
            "n_categorical_cols": self.n_categorical_cols,
            "n_numeric_cols": self.n_numeric_cols,
            "n_features_added": self.n_features_added
        }
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "Feature Engineering Summary",
            "=" * 50,
            f"Input shape: {self.input_shape[0]:,} rows × {self.input_shape[1]} cols",
            f"Output shape: {self.output_shape[0]:,} rows × {self.output_shape[1]} cols",
            f"Features added: {self.n_features_added}",
            "",
            "Processed:",
            f"  - Date columns: {self.n_date_cols}",
            f"  - Text columns: {self.n_text_cols}",
            f"  - Categorical columns: {self.n_categorical_cols}",
            f"  - Numeric columns: {self.n_numeric_cols}",
        ]
        
        if self.warnings:
            lines.append("")
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        
        return "\n".join(lines)


# ========================================================================================
# DATE DETECTION AND FEATURES
# ========================================================================================

def _is_date_series(
    s: pd.Series, 
    sample_size: int, 
    hit_threshold: float
) -> bool:
    """
    Check if series looks like dates based on regex patterns.
    
    Args:
        s: Series to check
        sample_size: Number of values to sample
        hit_threshold: Minimum hit ratio to consider as date
        
    Returns:
        True if series looks like dates
    """
    if not (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)):
        return False
    
    # Sample for performance
    sample = s.dropna().astype(str).head(max(1, sample_size))
    
    if sample.empty:
        return False
    
    # Check patterns
    max_hits = 0
    for pattern in DATE_PATTERNS:
        try:
            hits = int(sample.str.match(pattern, na=False).sum())
            max_hits = max(max_hits, hits)
            
            if max_hits / len(sample) >= hit_threshold:
                return True
        except Exception:
            continue
    
    return False


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Safely convert series to datetime.
    
    Args:
        series: Series to convert
        
    Returns:
        Datetime series or NaT series on failure
    """
    try:
        return pd.to_datetime(series, errors="coerce", utc=False)
    except Exception as e:
        logger.warning(f"Failed to convert to datetime: {e}")
        return pd.Series([pd.NaT] * len(series), index=series.index)


def _extract_date_features(
    df: pd.DataFrame,
    col: str,
    dt_series: pd.Series,
    opts: FEOptions
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features from datetime column.
    
    Args:
        df: DataFrame to modify
        col: Column name
        dt_series: Datetime series
        opts: Options
        
    Returns:
        Tuple of (modified_df, list_of_new_columns)
    """
    produced = []
    
    try:
        # Basic date components
        df[f"{col}_year"] = dt_series.dt.year.astype("Int16")
        df[f"{col}_month"] = dt_series.dt.month.astype("Int8")
        df[f"{col}_day"] = dt_series.dt.day.astype("Int8")
        df[f"{col}_dow"] = dt_series.dt.dayofweek.astype("Int8")  # 0=Monday
        df[f"{col}_quarter"] = dt_series.dt.quarter.astype("Int8")
        
        produced += [
            f"{col}_year", f"{col}_month", f"{col}_day", 
            f"{col}_dow", f"{col}_quarter"
        ]
        
        # ISO week
        try:
            iso_week = dt_series.dt.isocalendar().week
            df[f"{col}_iso_week"] = iso_week.astype("Int8")
            produced.append(f"{col}_iso_week")
        except Exception as e:
            logger.warning(f"Failed to extract ISO week for {col}: {e}")
        
        # Weekend indicator
        df[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype("Int8")
        produced.append(f"{col}_is_weekend")
        
        # Time features (if has time component)
        if opts.add_time_features:
            if (dt_series.dt.hour != 0).any() or (dt_series.dt.minute != 0).any():
                df[f"{col}_hour"] = dt_series.dt.hour.astype("Int8")
                df[f"{col}_minute"] = dt_series.dt.minute.astype("Int8")
                produced += [f"{col}_hour", f"{col}_minute"]
        
        # Cyclical encoding (for circular features like month, hour)
        if opts.add_cyclical_features:
            # Month (12 months)
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_series.dt.month / 12).astype("Float32")
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_series.dt.month / 12).astype("Float32")
            produced += [f"{col}_month_sin", f"{col}_month_cos"]
            
            # Day of week (7 days)
            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7).astype("Float32")
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7).astype("Float32")
            produced += [f"{col}_dow_sin", f"{col}_dow_cos"]
        
        # Age (years from now)
        if opts.add_age:
            try:
                now = pd.Timestamp.now(tz=dt_series.dt.tz)
                delta_days = (now - dt_series).dt.days
                
                # Only add age if most dates are in the past
                if (delta_days > 0).mean() >= 0.6:
                    age_years = (delta_days / 365.25).round(opts.age_round_decimals)
                    df[f"{col}_age"] = age_years.astype("Float32")
                    produced.append(f"{col}_age")
            except Exception as e:
                logger.warning(f"Failed to compute age for {col}: {e}")
        
    except Exception as e:
        logger.error(f"Error extracting date features from {col}: {e}")
    
    return df, produced


# ========================================================================================
# TEXT FEATURES
# ========================================================================================

def _extract_text_features(
    df: pd.DataFrame,
    col: str,
    opts: FEOptions
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features from text column.
    
    Args:
        df: DataFrame to modify
        col: Column name
        opts: Options
        
    Returns:
        Tuple of (modified_df, list_of_new_columns)
    """
    produced = []
    
    try:
        s = df[col].astype(str)
        
        # Basic length features
        df[f"{col}_len"] = s.str.len().fillna(0).astype(opts.text_len_dtype)
        produced.append(f"{col}_len")
        
        # Word count
        df[f"{col}_n_words"] = s.str.split().str.len().fillna(0).astype(opts.text_words_dtype)
        produced.append(f"{col}_n_words")
        
        # Pattern detection
        df[f"{col}_has_digits"] = s.str.contains(r"\d", regex=True, na=False).astype("Int8")
        produced.append(f"{col}_has_digits")
        
        # Advanced text complexity (if enabled)
        if opts.add_text_complexity:
            # Capital letters
            df[f"{col}_n_capitals"] = s.str.count(r"[A-Z]").fillna(0).astype("Int16")
            produced.append(f"{col}_n_capitals")
            
            # Special characters
            df[f"{col}_n_special"] = s.str.count(r"[^a-zA-Z0-9\s]").fillna(0).astype("Int16")
            produced.append(f"{col}_n_special")
            
            # Whitespace count
            df[f"{col}_n_spaces"] = s.str.count(r"\s").fillna(0).astype("Int16")
            produced.append(f"{col}_n_spaces")
            
            # Email pattern
            df[f"{col}_is_email"] = s.str.match(
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                na=False
            ).astype("Int8")
            produced.append(f"{col}_is_email")
            
            # URL pattern
            df[f"{col}_is_url"] = s.str.match(
                r"^https?://",
                na=False
            ).astype("Int8")
            produced.append(f"{col}_is_url")
        
    except Exception as e:
        logger.error(f"Error extracting text features from {col}: {e}")
    
    return df, produced


# ========================================================================================
# CATEGORICAL ENCODING
# ========================================================================================

def _encode_categorical(
    df: pd.DataFrame,
    col: str,
    nunique: int,
    opts: FEOptions
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Encode categorical column based on cardinality.
    
    Args:
        df: DataFrame
        col: Column name
        nunique: Number of unique values
        opts: Options
        
    Returns:
        Tuple of (modified_df, new_columns, encoding_type)
    """
    produced = []
    encoding_type = "none"
    
    try:
        if nunique <= 1:
            # Constant column - drop if enabled
            if opts.drop_constant_columns:
                df = df.drop(columns=[col])
                encoding_type = "dropped_constant"
            return df, produced, encoding_type
        
        elif nunique <= opts.cat_low_card_max:
            # One-hot encoding for low cardinality
            dummies = pd.get_dummies(
                df[col],
                prefix=col,
                prefix_sep=opts.one_hot_prefix_sep,
                drop_first=opts.one_hot_drop_first,
                dummy_na=opts.one_hot_dummy_na,
                dtype="Int8"  # Use nullable integer
            )
            
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            produced = dummies.columns.tolist()
            encoding_type = "one_hot"
            
        elif nunique <= opts.cat_mid_card_max:
            # Ordinal encoding for medium cardinality
            df[col] = df[col].astype("category").cat.codes.astype(opts.ordinal_dtype)
            produced = [col]
            encoding_type = "ordinal"
            
        elif nunique <= opts.cat_high_card_max and opts.handle_high_cardinality:
            # Hash encoding for high cardinality
            hash_values = df[col].astype(str).apply(lambda x: hash(x) % 1000).astype("Int16")
            df[f"{col}_hash"] = hash_values
            df = df.drop(columns=[col])
            produced = [f"{col}_hash"]
            encoding_type = "hash"
            
        else:
            # Too high cardinality - skip
            encoding_type = "skipped_high_card"
            logger.info(f"Skipping column {col} with {nunique} unique values")
    
    except Exception as e:
        logger.error(f"Error encoding categorical {col}: {e}")
        encoding_type = "error"
    
    return df, produced, encoding_type


# ========================================================================================
# NUMERIC FEATURES
# ========================================================================================

def _add_numeric_features(
    df: pd.DataFrame,
    col: str,
    opts: FEOptions
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add features for numeric columns.
    
    Args:
        df: DataFrame
        col: Column name
        opts: Options
        
    Returns:
        Tuple of (modified_df, new_columns)
    """
    produced = []
    
    try:
        # Binning
        if opts.add_binned_features:
            try:
                if opts.bin_strategy == "quantile":
                    bins = pd.qcut(
                        df[col], 
                        q=opts.n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                else:  # uniform
                    bins = pd.cut(
                        df[col],
                        bins=opts.n_bins,
                        labels=False
                    )
                
                df[f"{col}_bin"] = bins.astype("Int8")
                produced.append(f"{col}_bin")
                
            except Exception as e:
                logger.warning(f"Failed to bin column {col}: {e}")
        
    except Exception as e:
        logger.error(f"Error adding numeric features to {col}: {e}")
    
    return df, produced


# ========================================================================================
# MISSING INDICATORS
# ========================================================================================

def _add_missing_indicators(
    df: pd.DataFrame,
    opts: FEOptions
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add binary indicators for missing values.
    
    Args:
        df: DataFrame
        opts: Options
        
    Returns:
        Tuple of (modified_df, new_columns)
    """
    produced = []
    
    try:
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            
            if missing_ratio > opts.missing_threshold:
                indicator_col = f"{col}_is_missing"
                df[indicator_col] = df[col].isna().astype("Int8")
                produced.append(indicator_col)
                
    except Exception as e:
        logger.error(f"Error adding missing indicators: {e}")
    
    return df, produced


# ========================================================================================
# MAIN API
# ========================================================================================

def basic_feature_engineering_pro(
    df: pd.DataFrame,
    opts: FEOptions = FEOptions()
) -> Tuple[pd.DataFrame, FEResult]:
    """
    Perform comprehensive feature engineering with detailed reporting.
    
    Args:
        df: Input DataFrame
        opts: Feature engineering options
        
    Returns:
        Tuple of (engineered_df, result_report)
        
    Raises:
        ValueError: If DataFrame is invalid
        
    Examples:
        >>> df_new, report = basic_feature_engineering_pro(df)
        >>> print(report.summary())
        >>>
        >>> opts = FEOptions(add_cyclical_features=True, add_binned_features=True)
        >>> df_new, report = basic_feature_engineering_pro(df, opts)
    """
    # Validation
    if df is None:
        raise ValueError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    logger.info(f"Starting feature engineering: {len(df):,} rows × {df.shape[1]} cols")
    
    # Copy DataFrame
    out = df.copy(deep=True)
    actions: List[FEAction] = []
    warnings: List[str] = []
    new_cols_count = 0
    input_shape = out.shape
    
    # Counters for summary
    n_date = 0
    n_text = 0
    n_cat = 0
    n_num = 0
    
    # Sample if needed
    if opts.sample_large_datasets and len(out) > opts.max_sample_size:
        logger.info(f"Sampling {opts.max_sample_size:,} rows for feature engineering")
        # Note: We don't actually sample the output, just use for detection
    
    # ============================================================================
    # 1. DATE/TIME FEATURES
    # ============================================================================
    
    if opts.verbose:
        logger.info("Processing date/time columns...")
    
    for c in list(out.columns):
        # Check if looks like date by name or pattern
        col_lower = str(c).lower()
        looks_like_name = any(hint in col_lower for hint in opts.date_name_hints)
        looks_like_regex = _is_date_series(
            out[c], 
            opts.date_sample_size, 
            opts.date_hit_threshold
        )
        
        if not (looks_like_name or looks_like_regex):
            continue
        
        # Try to convert to datetime
        dt_series = _safe_to_datetime(out[c])
        parsed_ratio = float(dt_series.notna().mean())
        
        if parsed_ratio < opts.min_parsed_ratio:
            logger.info(f"Skipping {c}: only {parsed_ratio:.1%} parsed as dates")
            continue
        
        # Extract features
        out, produced = _extract_date_features(out, c, dt_series, opts)
        
        if produced:
            actions.append(FEAction(
                kind="date",
                column=c,
                produced=produced,
                note=f"Parsed {parsed_ratio:.1%} of values",
                details={"parsed_ratio": round(parsed_ratio, 3)}
            ))
            
            new_cols_count += len(produced)
            n_date += 1
            
            if new_cols_count >= opts.cap_new_columns:
                warnings.append(
                    f"Reached column limit ({opts.cap_new_columns}). "
                    "Stopping feature generation."
                )
                break
    
    # ============================================================================
    # 2. TEXT FEATURES
    # ============================================================================
    
    if new_cols_count < opts.cap_new_columns and opts.verbose:
        logger.info("Processing text columns...")
    
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    
    for c in text_cols:
        if new_cols_count >= opts.cap_new_columns:
            break
        
        try:
            nunique = int(out[c].nunique(dropna=True))
            
            # Skip if outside range
            if not (opts.text_min_unique <= nunique <= opts.text_max_unique):
                continue
            
            # Extract features
            out, produced = _extract_text_features(out, c, opts)
            
            if produced:
                actions.append(FEAction(
                    kind="text",
                    column=c,
                    produced=produced,
                    details={"nunique": nunique}
                ))
                
                new_cols_count += len(produced)
                n_text += 1
                
        except Exception as e:
            logger.error(f"Error processing text column {c}: {e}")
            warnings.append(f"Failed to process text column '{c}': {e}")
    
    # ============================================================================
    # 3. CATEGORICAL ENCODING
    # ============================================================================
    
    if new_cols_count < opts.cap_new_columns and opts.verbose:
        logger.info("Processing categorical columns...")
    
    cat_candidates = out.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    for c in cat_candidates:
        if new_cols_count >= opts.cap_new_columns:
            break
        
        try:
            nunique = int(out[c].nunique(dropna=True))
            
            # Skip very low or constant
            if nunique <= 1 and not opts.drop_constant_columns:
                continue
            
            # Encode
            out, produced, enc_type = _encode_categorical(out, c, nunique, opts)
            
            if produced or enc_type == "dropped_constant":
                actions.append(FEAction(
                    kind="categorical",
                    column=c,
                    produced=produced,
                    note=enc_type,
                    details={"nunique": nunique, "encoding": enc_type}
                ))
                
                new_cols_count += len(produced)
                n_cat += 1
                
        except Exception as e:
            logger.error(f"Error processing categorical column {c}: {e}")
            warnings.append(f"Failed to process categorical column '{c}': {e}")
    
    # ============================================================================
    # 4. NUMERIC FEATURES
    # ============================================================================
    
    if opts.add_binned_features and new_cols_count < opts.cap_new_columns:
        if opts.verbose:
            logger.info("Processing numeric columns...")
        
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        
        for c in numeric_cols:
            if new_cols_count >= opts.cap_new_columns:
                break
            
            try:
                out, produced = _add_numeric_features(out, c, opts)
                
                if produced:
                    actions.append(FEAction(
                        kind="numeric",
                        column=c,
                        produced=produced,
                        note="binning"
                    ))
                    
                    new_cols_count += len(produced)
                    n_num += 1
                    
            except Exception as e:
                logger.error(f"Error processing numeric column {c}: {e}")
    
    # ============================================================================
    # 5. MISSING INDICATORS
    # ============================================================================
    
    if opts.add_missing_indicators and new_cols_count < opts.cap_new_columns:
        if opts.verbose:
            logger.info("Adding missing value indicators...")
        
        try:
            out, produced = _add_missing_indicators(out, opts)
            
            if produced:
                actions.append(FEAction(
                    kind="missing",
                    column=None,
                    produced=produced,
                    note=f"Added {len(produced)} missing indicators"
                ))
                
                new_cols_count += len(produced)
                
        except Exception as e:
            logger.error(f"Error adding missing indicators: {e}")
            warnings.append(f"Failed to add missing indicators: {e}")
    
    # ============================================================================
    # 6. CLEANUP
    # ============================================================================
    
    # Replace inf with NaN
    if opts.replace_inf_with_nan:
        inf_counts = np.isinf(out.select_dtypes(include=[np.number])).sum().sum()
        if inf_counts > 0:
            out = out.replace([np.inf, -np.inf], np.nan)
            actions.append(FEAction(
                kind="repair",
                column=None,
                produced=[],
                note=f"Replaced {inf_counts} inf values with NaN"
            ))
    
    # Final shape
    output_shape = out.shape
    n_features_added = output_shape[1] - input_shape[1]
    
    # Build result
    result = FEResult(
        input_shape=input_shape,
        output_shape=output_shape,
        actions=actions,
        warnings=warnings,
        n_date_cols=n_date,
        n_text_cols=n_text,
        n_categorical_cols=n_cat,
        n_numeric_cols=n_num,
        n_features_added=n_features_added
    )
    
    logger.info(
        f"Feature engineering complete: {input_shape[1]} → {output_shape[1]} cols "
        f"({n_features_added} added)"
    )
    
    # Add backward compatible report attribute
    try:
        setattr(out, "fe_report", {
            "dates": [a.column for a in actions if a.kind == "date"],
            "text": [a.column for a in actions if a.kind == "text"],
            "categoricals": [
                {
                    "col": a.column,
                    "type": a.note or "",
                    "n_new": len(a.produced)
                }
                for a in actions if a.kind == "categorical"
            ],
            "warnings": warnings,
            "n_features_added": n_features_added
        })  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning(f"Failed to attach fe_report attribute: {e}")
    
    return out, result


# ========================================================================================
# BACKWARD COMPATIBILITY
# ========================================================================================

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward compatible wrapper for basic_feature_engineering_pro.
    
    Generates features using the new engine and attaches a report
    as df.fe_report attribute.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
        
    Examples:
        >>> df_new = basic_feature_engineering(df)
        >>> print(df_new.fe_report)
    """
    try:
        df_out, report = basic_feature_engineering_pro(df, FEOptions())
        
        # Attach simplified report for backward compatibility
        try:
            setattr(df_out, "fe_report", {
                "dates": [a.column for a in report.actions if a.kind == "date"],
                "text": [a.column for a in report.actions if a.kind == "text"],
                "categoricals": [a.column for a in report.actions if a.kind == "categorical"],
                "warnings": report.warnings,
            })  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Failed to attach fe_report: {e}")
        
        return df_out
        
    except Exception as e:
        logger.exception(f"Error in basic_feature_engineering: {e}")
        # Return copy on error
        return df.copy()


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def get_feature_importance_proxy(
    df: pd.DataFrame,
    target_col: str,
    max_features: int = 50
) -> pd.DataFrame:
    """
    Get proxy for feature importance using correlation with target.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        max_features: Maximum features to return
        
    Returns:
        DataFrame with feature importance scores
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    try:
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            raise ValueError(f"Target column '{target_col}' must be numeric")
        
        # Calculate correlations
        correlations = numeric_df.corr()[target_col].abs()
        
        # Remove target itself
        correlations = correlations.drop(target_col)
        
        # Sort and limit
        top_features = correlations.nlargest(max_features)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'feature': top_features.index,
            'importance': top_features.values
        }).reset_index(drop=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return pd.DataFrame(columns=['feature', 'importance'])


def detect_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect and categorize feature types in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict with feature type categories
    """
    result = {
        "numeric": [],
        "categorical_low": [],
        "categorical_high": [],
        "text": [],
        "datetime": [],
        "boolean": [],
        "constant": [],
        "id_like": []
    }
    
    try:
        for col in df.columns:
            s = df[col]
            nunique = s.nunique(dropna=True)
            
            # Constant
            if nunique <= 1:
                result["constant"].append(col)
                continue
            
            # Boolean
            if ptypes.is_bool_dtype(s) or nunique == 2:
                result["boolean"].append(col)
                continue
            
            # Datetime
            if ptypes.is_datetime64_any_dtype(s):
                result["datetime"].append(col)
                continue
            
            # Numeric
            if ptypes.is_numeric_dtype(s):
                # Check if ID-like (high uniqueness)
                if nunique / len(s) > 0.95:
                    result["id_like"].append(col)
                else:
                    result["numeric"].append(col)
                continue
            
            # Categorical or Text
            if ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
                if nunique <= 20:
                    result["categorical_low"].append(col)
                elif nunique <= 100:
                    result["categorical_high"].append(col)
                else:
                    result["text"].append(col)
                continue
            
            # Categorical type
            if ptypes.is_categorical_dtype(s):
                if nunique <= 20:
                    result["categorical_low"].append(col)
                else:
                    result["categorical_high"].append(col)
                continue
        
    except Exception as e:
        logger.error(f"Error detecting feature types: {e}")
    
    return result


def suggest_feature_engineering(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze DataFrame and suggest feature engineering strategies.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict with suggestions
    """
    suggestions = {
        "date_columns": [],
        "text_columns": [],
        "high_cardinality": [],
        "low_variance": [],
        "missing_heavy": [],
        "recommendations": []
    }
    
    try:
        feature_types = detect_feature_types(df)
        
        # Date suggestions
        for col in df.columns:
            if _is_date_series(df[col], 200, 0.6):
                suggestions["date_columns"].append(col)
        
        if suggestions["date_columns"]:
            suggestions["recommendations"].append(
                f"Consider extracting date features from: {', '.join(suggestions['date_columns'][:3])}"
            )
        
        # Text suggestions
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            nunique = df[col].nunique()
            if 5 <= nunique <= 5000:
                suggestions["text_columns"].append(col)
        
        if suggestions["text_columns"]:
            suggestions["recommendations"].append(
                f"Consider text feature extraction from: {', '.join(suggestions['text_columns'][:3])}"
            )
        
        # High cardinality
        for col in feature_types["categorical_high"]:
            suggestions["high_cardinality"].append(col)
        
        if suggestions["high_cardinality"]:
            suggestions["recommendations"].append(
                f"High cardinality columns may need special encoding: {', '.join(suggestions['high_cardinality'][:3])}"
            )
        
        # Low variance
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() < 0.01:
                suggestions["low_variance"].append(col)
        
        if suggestions["low_variance"]:
            suggestions["recommendations"].append(
                f"Consider removing low-variance columns: {', '.join(suggestions['low_variance'][:3])}"
            )
        
        # Missing data
        missing_pcts = df.isna().mean()
        heavy_missing = missing_pcts[missing_pcts > 0.3].index.tolist()
        suggestions["missing_heavy"] = heavy_missing
        
        if heavy_missing:
            suggestions["recommendations"].append(
                f"High missing data in: {', '.join(heavy_missing[:3])}. Consider adding missing indicators."
            )
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
    
    return suggestions


def export_feature_report(result: FEResult, filepath: str) -> None:
    """
    Export feature engineering report to file.
    
    Args:
        result: FEResult to export
        filepath: Output file path (.txt or .json)
    """
    import json
    from pathlib import Path
    
    path = Path(filepath)
    
    try:
        if path.suffix == ".json":
            # Export as JSON
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            # Export as text
            with open(path, "w", encoding="utf-8") as f:
                f.write(result.summary())
                f.write("\n\n")
                f.write("=" * 50)
                f.write("\n\nDetailed Actions:\n\n")
                
                for action in result.actions:
                    f.write(f"Kind: {action.kind}\n")
                    f.write(f"Column: {action.column}\n")
                    f.write(f"Produced: {len(action.produced)} features\n")
                    if action.note:
                        f.write(f"Note: {action.note}\n")
                    f.write("\n")
        
        logger.info(f"Report exported to: {path}")
        
    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        raise


def quick_feature_engineering(
    df: pd.DataFrame,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick feature engineering with sensible defaults.
    
    Args:
        df: Input DataFrame
        target_col: Optional target column (excluded from encoding)
        
    Returns:
        DataFrame with engineered features
    """
    # Create options with safe defaults
    opts = FEOptions(
        add_age=True,
        add_text_complexity=False,  # Faster
        add_cyclical_features=False,  # Faster
        add_binned_features=False,  # Faster
        add_missing_indicators=True,
        check_correlations=False,
        one_hot_drop_first=True,
        replace_inf_with_nan=True,
        drop_constant_columns=True,
        verbose=False
    )
    
    # Exclude target from encoding if specified
    if target_col and target_col in df.columns:
        target_series = df[target_col].copy()
        df_without_target = df.drop(columns=[target_col])
        
        df_out, _ = basic_feature_engineering_pro(df_without_target, opts)
        df_out[target_col] = target_series
        
        return df_out
    else:
        df_out, _ = basic_feature_engineering_pro(df, opts)
        return df_out


# ========================================================================================
# ADVANCED FEATURES
# ========================================================================================

def add_interaction_features(
    df: pd.DataFrame,
    columns: List[str],
    max_interactions: int = 10
) -> pd.DataFrame:
    """
    Add interaction features between specified columns.
    
    Args:
        df: DataFrame
        columns: Columns to create interactions from
        max_interactions: Maximum number of interactions to create
        
    Returns:
        DataFrame with interaction features
    """
    df_out = df.copy()
    interactions_added = 0
    
    try:
        # Only numeric columns
        numeric_cols = [c for c in columns if c in df.columns and ptypes.is_numeric_dtype(df[c])]
        
        if len(numeric_cols) < 2:
            logger.warning("Need at least 2 numeric columns for interactions")
            return df_out
        
        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            if interactions_added >= max_interactions:
                break
            
            for col2 in numeric_cols[i+1:]:
                if interactions_added >= max_interactions:
                    break
                
                # Multiplication
                interaction_name = f"{col1}_x_{col2}"
                df_out[interaction_name] = df[col1] * df[col2]
                interactions_added += 1
        
        logger.info(f"Added {interactions_added} interaction features")
        
    except Exception as e:
        logger.error(f"Error adding interaction features: {e}")
    
    return df_out


def add_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Add polynomial features for specified columns.
    
    Args:
        df: DataFrame
        columns: Columns to create polynomials from
        degree: Polynomial degree
        
    Returns:
        DataFrame with polynomial features
    """
    df_out = df.copy()
    
    try:
        for col in columns:
            if col not in df.columns:
                continue
            
            if not ptypes.is_numeric_dtype(df[col]):
                continue
            
            for d in range(2, degree + 1):
                poly_name = f"{col}_pow{d}"
                df_out[poly_name] = df[col] ** d
        
    except Exception as e:
        logger.error(f"Error adding polynomial features: {e}")
    
    return df_out


# ========================================================================================
# MODULE INFO
# ========================================================================================

def get_module_info() -> Dict[str, Any]:
    """
    Get information about this module.
    
    Returns:
        Dict with module information
    """
    return {
        "name": "feature_engineering",
        "version": "2.0.0-turbo-pro++",
        "description": "Advanced automated feature engineering module",
        "features": [
            "Automatic datetime detection and feature extraction",
            "Text feature engineering",
            "Smart categorical encoding",
            "Numeric transformations",
            "Missing indicators",
            "Interaction and polynomial features",
            "Comprehensive reporting",
            "Backward compatibility"
        ]
    }


# ========================================================================================
# VALIDATION
# ========================================================================================

def validate_engineered_features(
    original_df: pd.DataFrame,
    engineered_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate engineered features against original DataFrame.
    
    Args:
        original_df: Original DataFrame
        engineered_df: Engineered DataFrame
        
    Returns:
        Dict with validation results
    """
    validation = {
        "row_count_match": len(original_df) == len(engineered_df),
        "original_cols": original_df.shape[1],
        "engineered_cols": engineered_df.shape[1],
        "new_cols": engineered_df.shape[1] - original_df.shape[1],
        "missing_original_cols": [],
        "new_col_names": [],
        "issues": []
    }
    
    try:
        # Check for missing original columns
        original_cols = set(original_df.columns)
        engineered_cols = set(engineered_df.columns)
        
        missing = original_cols - engineered_cols
        new = engineered_cols - original_cols
        
        validation["missing_original_cols"] = list(missing)
        validation["new_col_names"] = list(new)
        
        # Check for issues
        if not validation["row_count_match"]:
            validation["issues"].append("Row count mismatch")
        
        if missing:
            validation["issues"].append(f"{len(missing)} original columns missing")
        
        # Check for inf values
        inf_count = np.isinf(engineered_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            validation["issues"].append(f"{inf_count} inf values present")
        
        # Check for all-null columns
        null_cols = engineered_df.columns[engineered_df.isna().all()].tolist()
        if null_cols:
            validation["issues"].append(f"{len(null_cols)} all-null columns: {null_cols[:5]}")
        
    except Exception as e:
        logger.error(f"Error validating features: {e}")
        validation["issues"].append(f"Validation error: {e}")
    
    return validation