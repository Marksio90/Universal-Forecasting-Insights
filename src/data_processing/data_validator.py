# data_quality.py — TURBO PRO++ Enhanced Edition
"""
Advanced Data Quality Validation Module.

Features:
- Comprehensive data quality checks
- Configurable severity levels
- Performance optimizations for large datasets
- Detailed findings with actionable insights
- Memory-efficient operations
- Extensive type detection
- Backward compatibility
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Literal, Tuple, Set

import numpy as np
import pandas as pd

# ========================================================================================
# LOGGING
# ========================================================================================

logger = logging.getLogger(__name__)

# ========================================================================================
# CONFIGURATION
# ========================================================================================

# Performance limits
MAX_SAMPLE_FOR_CHECKS = 100_000  # Limit for expensive checks
MAX_COLS_FOR_CORRELATION = 200
MAX_PAIRS_CHECK_LIMIT = 10_000  # For identical columns check

# Default thresholds
DEFAULT_HIGH_MISSING = 0.50
DEFAULT_NEAR_CONSTANT = 0.99
DEFAULT_UNIQUE_AS_ID = 0.98
DEFAULT_ZSCORE = 4.0
DEFAULT_IQR = 3.0
DEFAULT_STRONG_CORR = 0.98

# ========================================================================================
# TYPES
# ========================================================================================

Severity = Literal["info", "warn", "error"]


@dataclass(frozen=True)
class QualityOptions:
    """Configuration for data quality validation."""
    
    # Missing data thresholds
    high_missing_threshold: float = DEFAULT_HIGH_MISSING
    
    # Variance thresholds
    near_constant_threshold: float = DEFAULT_NEAR_CONSTANT
    unique_as_id_ratio: float = DEFAULT_UNIQUE_AS_ID
    
    # Outlier detection
    zscore_threshold: float = DEFAULT_ZSCORE
    iqr_threshold: float = DEFAULT_IQR
    min_outlier_sample: int = 10
    
    # Text quality
    sample_text_check: int = 100
    detect_whitespace_only: bool = True
    detect_unusual_chars: bool = True
    
    # Advanced checks
    check_correlations: bool = False
    strong_corr_threshold: float = DEFAULT_STRONG_CORR
    limit_cols_for_corr: int = MAX_COLS_FOR_CORRELATION
    
    detect_mixed_types: bool = True
    detect_identical_columns: bool = True
    
    # Performance
    memory_stats: bool = True
    sample_large_datasets: bool = True
    max_sample_size: int = MAX_SAMPLE_FOR_CHECKS
    
    # Reporting
    verbose_findings: bool = True
    include_recommendations: bool = True
    
    def __post_init__(self):
        """Validate options."""
        # Validate thresholds
        if not 0 < self.high_missing_threshold <= 1:
            object.__setattr__(self, 'high_missing_threshold', DEFAULT_HIGH_MISSING)
            logger.warning("Invalid high_missing_threshold, using default")
        
        if not 0 < self.near_constant_threshold <= 1:
            object.__setattr__(self, 'near_constant_threshold', DEFAULT_NEAR_CONSTANT)
            logger.warning("Invalid near_constant_threshold, using default")
        
        if not 0 < self.unique_as_id_ratio <= 1:
            object.__setattr__(self, 'unique_as_id_ratio', DEFAULT_UNIQUE_AS_ID)
            logger.warning("Invalid unique_as_id_ratio, using default")


@dataclass(frozen=True)
class Finding:
    """Individual quality finding."""
    
    kind: str
    column: Optional[str]
    message: str
    severity: Severity
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kind": self.kind,
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "recommendation": self.recommendation
        }


@dataclass(frozen=True)
class QualityReport:
    """Comprehensive data quality report."""
    
    # Basic stats
    rows: int
    cols: int
    missing_pct: float
    dupes: int
    
    # Type information
    dtypes_summary: Dict[str, int]
    memory_bytes: Optional[int]
    
    # Column-level issues
    constant_columns: List[str]
    near_constant_columns: Dict[str, float]
    unique_columns: List[str]
    high_missing_cols: Dict[str, float]
    
    # Outliers
    outliers_detected_z: Dict[str, int]
    outliers_detected_iqr: Dict[str, int]
    
    # Text quality
    textual_anomalies: List[str]
    
    # Advanced checks
    mixed_type_columns: List[str]
    identical_columns: List[Tuple[str, str]]
    strong_correlations: List[Tuple[str, str, float]]
    
    # Findings
    findings: List[Finding]
    
    # Metadata
    sampled: bool = False
    sample_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "missing_pct": self.missing_pct,
            "dupes": self.dupes,
            "dtypes_summary": self.dtypes_summary,
            "memory_bytes": self.memory_bytes,
            "constant_columns": self.constant_columns,
            "near_constant_columns": self.near_constant_columns,
            "unique_columns": self.unique_columns,
            "high_missing_cols": self.high_missing_cols,
            "outliers_detected_z": self.outliers_detected_z,
            "outliers_detected_iqr": self.outliers_detected_iqr,
            "textual_anomalies": self.textual_anomalies,
            "mixed_type_columns": self.mixed_type_columns,
            "identical_columns": self.identical_columns,
            "strong_correlations": self.strong_correlations,
            "findings": [f.to_dict() for f in self.findings],
            "sampled": self.sampled,
            "sample_size": self.sample_size
        }
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"Quality Report Summary",
            f"=" * 50,
            f"Dataset: {self.rows:,} rows × {self.cols} columns",
            f"Missing: {self.missing_pct:.2%}",
            f"Duplicates: {self.dupes:,}",
            f"",
            f"Issues Found: {len(self.findings)}",
        ]
        
        # Count by severity
        severity_counts = {"error": 0, "warn": 0, "info": 0}
        for f in self.findings:
            severity_counts[f.severity] += 1
        
        lines.append(f"  - Errors: {severity_counts['error']}")
        lines.append(f"  - Warnings: {severity_counts['warn']}")
        lines.append(f"  - Info: {severity_counts['info']}")
        
        return "\n".join(lines)


# ========================================================================================
# HELPER FUNCTIONS
# ========================================================================================

def _safe_sample(df: pd.DataFrame, max_size: int) -> Tuple[pd.DataFrame, bool]:
    """
    Safely sample DataFrame for expensive checks.
    
    Args:
        df: Source DataFrame
        max_size: Maximum sample size
        
    Returns:
        Tuple of (sampled_df, was_sampled)
    """
    if len(df) <= max_size:
        return df, False
    
    logger.info(f"Sampling {max_size:,} rows from {len(df):,} for quality checks")
    return df.sample(n=max_size, random_state=42), True


def _dtype_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary of data types.
    
    Args:
        df: DataFrame
        
    Returns:
        Dict with type counts
    """
    try:
        cats = {
            "numeric": df.select_dtypes(include=[np.number]).shape[1],
            "bool": df.select_dtypes(include=["bool", "boolean"]).shape[1],
            "datetime": df.select_dtypes(include=["datetime64", "datetimetz"]).shape[1],
            "category": df.select_dtypes(include=["category"]).shape[1],
            "string": df.select_dtypes(include=["string"]).shape[1],
            "object": df.select_dtypes(include=["object"]).shape[1],
            "timedelta": df.select_dtypes(include=["timedelta"]).shape[1],
        }
        return {k: v for k, v in cats.items() if v > 0}
    except Exception as e:
        logger.error(f"Error getting dtype summary: {e}")
        return {}


def _detect_constant_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns with only one unique value.
    
    Args:
        df: DataFrame
        
    Returns:
        List of constant column names
    """
    try:
        return [
            c for c in df.columns 
            if df[c].nunique(dropna=True) <= 1
        ]
    except Exception as e:
        logger.error(f"Error detecting constant columns: {e}")
        return []


def _detect_near_constant(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """
    Detect columns where one value dominates.
    
    Args:
        df: DataFrame
        threshold: Dominance threshold (0-1)
        
    Returns:
        Dict of {column: dominance_ratio}
    """
    res: Dict[str, float] = {}
    
    try:
        for c in df.columns:
            s = df[c]
            if len(s) == 0:
                continue
            
            vc = s.value_counts(dropna=True)
            if vc.empty:
                continue
            
            dom_ratio = float(vc.iloc[0]) / float(len(s))
            nunique = s.nunique(dropna=True)
            
            # Must have > 1 unique value (otherwise it's constant)
            if dom_ratio >= threshold and nunique > 1:
                res[c] = round(dom_ratio, 4)
                
    except Exception as e:
        logger.error(f"Error detecting near-constant columns: {e}")
    
    return res


def _detect_unique_columns(df: pd.DataFrame, as_id_ratio: float) -> List[str]:
    """
    Detect columns that look like identifiers (high uniqueness).
    
    Args:
        df: DataFrame
        as_id_ratio: Uniqueness threshold
        
    Returns:
        List of potential ID columns
    """
    n = len(df)
    if n == 0:
        return []
    
    out = []
    
    try:
        for c in df.columns:
            nunique = df[c].nunique(dropna=True)
            ratio = nunique / n
            
            if ratio >= as_id_ratio:
                out.append(c)
                
    except Exception as e:
        logger.error(f"Error detecting unique columns: {e}")
    
    return out


def _detect_high_missing(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """
    Detect columns with high missing percentages.
    
    Args:
        df: DataFrame
        threshold: Missing threshold (0-1)
        
    Returns:
        Dict of {column: missing_percentage}
    """
    try:
        ratios = df.isna().mean()
        mask = ratios >= threshold
        return {
            c: round(float(rat * 100.0), 2) 
            for c, rat in ratios[mask].items()
        }
    except Exception as e:
        logger.error(f"Error detecting high missing: {e}")
        return {}


def _outliers_zscore(
    df: pd.DataFrame, 
    z_thresh: float, 
    min_n: int
) -> Dict[str, int]:
    """
    Detect outliers using Z-score method.
    
    Args:
        df: DataFrame
        z_thresh: Z-score threshold
        min_n: Minimum sample size
        
    Returns:
        Dict of {column: outlier_count}
    """
    out: Dict[str, int] = {}
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for c in numeric_cols:
            vals = df[c].dropna()
            
            if len(vals) < min_n:
                continue
            
            # Calculate Z-scores
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            
            if std == 0 or math.isclose(std, 0.0):
                continue
            
            z_scores = (vals - mean) / std
            n_outliers = int((np.abs(z_scores) > z_thresh).sum())
            
            if n_outliers > 0:
                out[c] = n_outliers
                
    except Exception as e:
        logger.error(f"Error detecting Z-score outliers: {e}")
    
    return out


def _outliers_iqr(
    df: pd.DataFrame, 
    iqr_k: float, 
    min_n: int
) -> Dict[str, int]:
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Args:
        df: DataFrame
        iqr_k: IQR multiplier (typically 1.5 or 3.0)
        min_n: Minimum sample size
        
    Returns:
        Dict of {column: outlier_count}
    """
    out: Dict[str, int] = {}
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for c in numeric_cols:
            vals = df[c].dropna()
            
            if len(vals) < min_n:
                continue
            
            # Calculate quartiles
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            
            if iqr <= 0:
                continue
            
            # Calculate bounds
            lower_bound = q1 - iqr_k * iqr
            upper_bound = q3 + iqr_k * iqr
            
            # Count outliers
            n_outliers = int(((vals < lower_bound) | (vals > upper_bound)).sum())
            
            if n_outliers > 0:
                out[c] = n_outliers
                
    except Exception as e:
        logger.error(f"Error detecting IQR outliers: {e}")
    
    return out


def _textual_anomalies(
    df: pd.DataFrame, 
    sample_n: int,
    check_whitespace: bool = True,
    check_unusual: bool = True
) -> List[str]:
    """
    Detect text columns with quality issues.
    
    Args:
        df: DataFrame
        sample_n: Sample size for check
        check_whitespace: Check for whitespace-only values
        check_unusual: Check for unusual characters
        
    Returns:
        List of problematic column names
    """
    try:
        text_cols = df.select_dtypes(include=["object", "string"]).columns
        bad: List[str] = []
        
        for c in text_cols:
            s = df[c].astype(str).dropna().head(sample_n)
            
            if s.empty:
                continue
            
            # Check for whitespace-only values
            if check_whitespace:
                if (s.str.strip() == "").any():
                    bad.append(c)
                    continue
            
            # Check for unusual characters (optional)
            if check_unusual:
                # Check for non-printable characters
                has_unusual = s.str.contains(r'[\x00-\x1f\x7f-\x9f]', regex=True, na=False).any()
                if has_unusual:
                    if c not in bad:
                        bad.append(c)
        
        return bad
        
    except Exception as e:
        logger.error(f"Error detecting textual anomalies: {e}")
        return []


def _mixed_type_columns(df: pd.DataFrame, sample_n: int = 200) -> List[str]:
    """
    Detect object columns with mixed data types.
    
    Args:
        df: DataFrame
        sample_n: Sample size for check
        
    Returns:
        List of mixed-type column names
    """
    out: List[str] = []
    
    try:
        for c in df.columns:
            if df[c].dtype != "object":
                continue
            
            # Sample for performance
            s = df[c].dropna().head(sample_n)
            
            if s.empty:
                continue
            
            # Get unique types
            types: Set[str] = set()
            for val in s:
                types.add(type(val).__name__)
            
            # If more than one type, it's mixed
            if len(types) > 1:
                out.append(c)
                
    except Exception as e:
        logger.error(f"Error detecting mixed-type columns: {e}")
    
    return out


def _identical_columns(
    df: pd.DataFrame, 
    limit: int = MAX_PAIRS_CHECK_LIMIT
) -> List[Tuple[str, str]]:
    """
    Detect pairs of identical columns.
    
    Args:
        df: DataFrame
        limit: Maximum pairs to check (for performance)
        
    Returns:
        List of (col1, col2) tuples
    """
    ncols = df.shape[1]
    if ncols <= 1:
        return []
    
    try:
        # Create fingerprints for fast comparison
        fingerprints: Dict[str, Tuple] = {}
        
        for c in df.columns:
            s = df[c]
            
            # Create fingerprint: (head_hash, tail_hash, nan_count, dtype)
            head_vals = tuple(s.head(50).tolist())
            tail_vals = tuple(s.tail(50).tolist())
            nan_count = int(s.isna().sum())
            dtype_str = str(s.dtype)
            
            fingerprints[c] = (
                hash(head_vals), 
                hash(tail_vals), 
                nan_count, 
                dtype_str
            )
        
        # Find pairs with matching fingerprints
        pairs: List[Tuple[str, str]] = []
        cols = list(df.columns)
        checked = 0
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if checked >= limit:
                    logger.warning(f"Reached check limit ({limit}) for identical columns")
                    return pairs
                
                checked += 1
                col_a, col_b = cols[i], cols[j]
                
                # Quick fingerprint check
                if fingerprints[col_a] == fingerprints[col_b]:
                    # Confirm with full comparison (expensive)
                    if df[col_a].equals(df[col_b]):
                        pairs.append((col_a, col_b))
        
        return pairs
        
    except Exception as e:
        logger.error(f"Error detecting identical columns: {e}")
        return []


def _strong_correlations(
    df: pd.DataFrame, 
    threshold: float, 
    max_cols: int
) -> List[Tuple[str, str, float]]:
    """
    Detect strongly correlated numeric columns.
    
    Args:
        df: DataFrame
        threshold: Correlation threshold
        max_cols: Maximum columns to process
        
    Returns:
        List of (col1, col2, correlation) tuples
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return []
        
        if numeric_df.shape[1] > max_cols:
            logger.warning(
                f"Too many numeric columns ({numeric_df.shape[1]}) for correlation, "
                f"limiting to {max_cols}"
            )
            numeric_df = numeric_df.iloc[:, :max_cols]
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr(numeric_only=True).abs()
        
        # Find strong correlations
        pairs: List[Tuple[str, str, float]] = []
        cols = corr_matrix.columns.tolist()
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = float(corr_matrix.iat[i, j])
                
                if corr_val >= threshold:
                    pairs.append((cols[i], cols[j], round(corr_val, 4)))
        
        return pairs
        
    except Exception as e:
        logger.error(f"Error computing correlations: {e}")
        return []


def _generate_recommendations(finding: Finding) -> str:
    """
    Generate actionable recommendation for a finding.
    
    Args:
        finding: Quality finding
        
    Returns:
        Recommendation string
    """
    recommendations = {
        "duplicates": "Consider using df.drop_duplicates() to remove duplicate rows.",
        "missing_overall": "Investigate missing data patterns. Consider imputation or removal.",
        "constant": "Consider dropping this column as it provides no information.",
        "near_constant": "This column has low variance. Consider feature engineering or removal.",
        "id_candidate": "This column appears to be an identifier. Consider using as index.",
        "missing_column": "High missing rate. Consider dropping or investigating why data is missing.",
        "text_whitespace": "Clean whitespace-only values using df[col].str.strip().",
        "mixed_types": "Standardize data types in this column for consistency.",
        "identical_columns": "One of these columns is redundant and can be removed.",
        "strong_corr": "Consider removing one of these correlated features to reduce multicollinearity.",
    }
    
    return recommendations.get(finding.kind, "Review this finding and take appropriate action.")


# ========================================================================================
# MAIN API
# ========================================================================================

def validate_pro(
    df: pd.DataFrame, 
    opts: QualityOptions = QualityOptions()
) -> QualityReport:
    """
    Perform comprehensive data quality validation.
    
    Args:
        df: DataFrame to validate
        opts: Quality options
        
    Returns:
        QualityReport with findings
        
    Raises:
        ValueError: If DataFrame is invalid
        
    Examples:
        >>> report = validate_pro(df)
        >>> print(report.summary())
        >>> 
        >>> opts = QualityOptions(check_correlations=True)
        >>> report = validate_pro(df, opts)
    """
    # Validation
    if df is None:
        raise ValueError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    logger.info(f"Starting quality validation: {len(df):,} rows × {df.shape[1]} cols")
    
    # Sample if needed
    df_check, was_sampled = _safe_sample(df, opts.max_sample_size) if opts.sample_large_datasets else (df, False)
    
    # Basic stats
    rows, cols = df.shape
    total_cells = df.size or 1
    missing_count = df.isna().sum().sum()
    missing_pct = float(missing_count / total_cells)
    dupes = int(df.duplicated().sum())
    
    # Type summary
    dtypes_sum = _dtype_summary(df)
    
    # Memory
    mem_bytes = None
    if opts.memory_stats:
        try:
            mem_bytes = int(df.memory_usage(deep=True).sum())
        except Exception as e:
            logger.warning(f"Failed to compute memory: {e}")
    
    # Column-level checks
    const_cols = _detect_constant_columns(df_check)
    near_const = _detect_near_constant(df_check, opts.near_constant_threshold)
    unique_cols = _detect_unique_columns(df_check, opts.unique_as_id_ratio)
    high_missing = _detect_high_missing(df, opts.high_missing_threshold)
    
    # Outlier detection
    out_z = _outliers_zscore(df_check, opts.zscore_threshold, opts.min_outlier_sample)
    out_iqr = _outliers_iqr(df_check, opts.iqr_threshold, opts.min_outlier_sample)
    
    # Text quality
    text_anom = []
    if opts.sample_text_check > 0:
        text_anom = _textual_anomalies(
            df_check, 
            opts.sample_text_check,
            opts.detect_whitespace_only,
            opts.detect_unusual_chars
        )
    
    # Advanced checks
    mixed_cols = _mixed_type_columns(df_check) if opts.detect_mixed_types else []
    identical = _identical_columns(df_check) if opts.detect_identical_columns else []
    
    strong_corr = []
    if opts.check_correlations:
        strong_corr = _strong_correlations(
            df_check, 
            opts.strong_corr_threshold, 
            opts.limit_cols_for_corr
        )
    
    # Generate findings
    findings: List[Finding] = []
    
    # Duplicates
    if dupes > 0:
        severity: Severity = "error" if dupes > rows * 0.1 else "warn"
        rec = _generate_recommendations(Finding("duplicates", None, "", "warn"))
        findings.append(Finding(
            "duplicates", 
            None, 
            f"{dupes:,} zduplikowanych wierszy ({dupes/rows:.1%}).", 
            severity,
            {"count": dupes, "percentage": round(dupes/rows * 100, 2)},
            rec if opts.include_recommendations else None
        ))
    
    # Overall missing
    if missing_pct > 0.2:
        rec = _generate_recommendations(Finding("missing_overall", None, "", "warn"))
        findings.append(Finding(
            "missing_overall", 
            None, 
            f"Wysoki udział braków: {missing_pct:.1%}.", 
            "warn",
            {"percentage": round(missing_pct * 100, 2)},
            rec if opts.include_recommendations else None
        ))
    
    # Constant columns
    for c in const_cols:
        rec = _generate_recommendations(Finding("constant", c, "", "warn"))
        findings.append(Finding(
            "constant", 
            c, 
            "Kolumna stała (brak wariancji).", 
            "warn",
            {},
            rec if opts.include_recommendations else None
        ))
    
    # Near-constant columns
    for c, ratio in near_const.items():
        rec = _generate_recommendations(Finding("near_constant", c, "", "info"))
        findings.append(Finding(
            "near_constant", 
            c, 
            f"Dominująca wartość {ratio*100:.1f}% obserwacji.", 
            "info",
            {"dominance_ratio": ratio},
            rec if opts.include_recommendations else None
        ))
    
    # ID candidates
    for c in unique_cols:
        rec = _generate_recommendations(Finding("id_candidate", c, "", "info"))
        findings.append(Finding(
            "id_candidate", 
            c, 
            "Kolumna wygląda na identyfikator (prawie unikalna).", 
            "info",
            {},
            rec if opts.include_recommendations else None
        ))
    
    # High missing columns
    for c, pct in high_missing.items():
        sev: Severity = "error" if pct >= 90 else ("warn" if pct >= 50 else "info")
        rec = _generate_recommendations(Finding("missing_column", c, "", sev))
        findings.append(Finding(
            "missing_column", 
            c, 
            f"Braki: {pct:.2f}%.", 
            sev,
            {"percentage": pct},
            rec if opts.include_recommendations else None
        ))
    
    # Text anomalies
    for c in text_anom:
        rec = _generate_recommendations(Finding("text_whitespace", c, "", "info"))
        findings.append(Finding(
            "text_whitespace", 
            c, 
            "Puste / whitespace-only wartości w próbie.", 
            "info",
            {},
            rec if opts.include_recommendations else None
        ))
    
    # Mixed types
    for c in mixed_cols:
        rec = _generate_recommendations(Finding("mixed_types", c, "", "warn"))
        findings.append(Finding(
            "mixed_types", 
            c, 
            "Mieszane typy w kolumnie object.", 
            "warn",
            {},
            rec if opts.include_recommendations else None
        ))
    
    # Identical columns
    for col_a, col_b in identical:
        rec = _generate_recommendations(Finding("identical_columns", None, "", "info"))
        findings.append(Finding(
            "identical_columns", 
            None, 
            f"Kolumny '{col_a}' i '{col_b}' są identyczne.", 
            "info",
            {"columns": [col_a, col_b]},
            rec if opts.include_recommendations else None
        ))
    
    # Strong correlations
    for col_a, col_b, corr_val in strong_corr:
        rec = _generate_recommendations(Finding("strong_corr", None, "", "info"))
        findings.append(Finding(
            "strong_corr", 
            None, 
            f"Silna korelacja {col_a} ~ {col_b}: {corr_val:.3f}.", 
            "info",
            {"columns": [col_a, col_b], "correlation": corr_val},
            rec if opts.include_recommendations else None
        ))
    
    logger.info(f"Quality validation complete: {len(findings)} findings")
    
    # Build report
    return QualityReport(
        rows=rows,
        cols=cols,
        missing_pct=missing_pct,
        dupes=dupes,
        dtypes_summary=dtypes_sum,
        memory_bytes=mem_bytes,
        constant_columns=const_cols,
        near_constant_columns=near_const,
        unique_columns=unique_cols,
        high_missing_cols=high_missing,
        outliers_detected_z=out_z,
        outliers_detected_iqr=out_iqr,
        textual_anomalies=text_anom,
        mixed_type_columns=mixed_cols,
        identical_columns=identicalidentical_columns=identical,
        strong_correlations=strong_corr,
        findings=findings,
        sampled=was_sampled,
        sample_size=len(df_check) if was_sampled else None
    )


# ========================================================================================
# BACKWARD COMPATIBILITY
# ========================================================================================

def validate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Backward compatible wrapper for validate_pro.
    
    Returns a flat dictionary similar to the original version,
    but uses validate_pro under the hood for consistency.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict with validation results
        
    Examples:
        >>> result = validate(df)
        >>> print(f"Rows: {result['rows']}, Duplicates: {result['dupes']}")
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("Invalid DataFrame in validate()")
        return {"error": "Brak danych lub nieprawidłowy DataFrame"}
    
    if df.empty:
        logger.warning("Empty DataFrame in validate()")
        return {"error": "DataFrame jest pusty"}
    
    try:
        # Use validate_pro with default options
        report = validate_pro(df, QualityOptions())
        
        # Convert to old format
        out: Dict[str, Any] = {
            "rows": report.rows,
            "cols": report.cols,
            "missing_pct": report.missing_pct,
            "dupes": report.dupes,
            "constant_columns": report.constant_columns,
            "unique_columns": report.unique_columns,
            "high_missing_cols": report.high_missing_cols,
            
            # Merge both outlier detection methods
            "outliers_detected": {
                **report.outliers_detected_z,
                **report.outliers_detected_iqr
            },
            
            "textual_anomalies": report.textual_anomalies,
            
            # Additional fields (backward compatible additions)
            "near_constant_columns": report.near_constant_columns,
            "mixed_type_columns": report.mixed_type_columns,
            "identical_columns": report.identical_columns,
            "dtypes_summary": report.dtypes_summary,
        }
        
        # Add memory if available
        if report.memory_bytes is not None:
            out["memory_mb"] = round(report.memory_bytes / 1e6, 2)
        
        return out
        
    except Exception as e:
        logger.exception(f"Error in validate(): {e}")
        return {"error": f"Błąd walidacji: {str(e)}"}


def basic_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Alias for validate() for compatibility with existing code.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dict with basic quality metrics
    """
    return validate(df)


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def get_quality_score(report: QualityReport) -> float:
    """
    Calculate overall quality score (0-100).
    
    Higher is better. Considers:
    - Missing data
    - Duplicates
    - Constant columns
    - Severity of findings
    
    Args:
        report: QualityReport
        
    Returns:
        Quality score (0-100)
    """
    score = 100.0
    
    # Penalize missing data
    score -= min(report.missing_pct * 100, 30)  # Max 30 point penalty
    
    # Penalize duplicates
    if report.rows > 0:
        dup_ratio = report.dupes / report.rows
        score -= min(dup_ratio * 50, 20)  # Max 20 point penalty
    
    # Penalize constant columns
    if report.cols > 0:
        const_ratio = len(report.constant_columns) / report.cols
        score -= min(const_ratio * 100, 15)  # Max 15 point penalty
    
    # Penalize severe findings
    error_count = sum(1 for f in report.findings if f.severity == "error")
    warn_count = sum(1 for f in report.findings if f.severity == "warn")
    
    score -= error_count * 5  # 5 points per error
    score -= warn_count * 2   # 2 points per warning
    
    return max(0.0, min(100.0, score))


def filter_findings_by_severity(
    report: QualityReport, 
    severity: Severity
) -> List[Finding]:
    """
    Filter findings by severity level.
    
    Args:
        report: QualityReport
        severity: Severity level to filter
        
    Returns:
        List of findings with specified severity
    """
    return [f for f in report.findings if f.severity == severity]


def get_actionable_findings(report: QualityReport) -> List[Finding]:
    """
    Get findings that require action (errors and warnings).
    
    Args:
        report: QualityReport
        
    Returns:
        List of actionable findings
    """
    return [
        f for f in report.findings 
        if f.severity in ("error", "warn")
    ]


def format_findings_as_text(findings: List[Finding], include_recommendations: bool = True) -> str:
    """
    Format findings as readable text.
    
    Args:
        findings: List of findings
        include_recommendations: Whether to include recommendations
        
    Returns:
        Formatted text
    """
    if not findings:
        return "✅ No issues found."
    
    lines = []
    
    # Group by severity
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]
    info = [f for f in findings if f.severity == "info"]
    
    if errors:
        lines.append("🔴 ERRORS:")
        for f in errors:
            col_str = f" [{f.column}]" if f.column else ""
            lines.append(f"  • {f.message}{col_str}")
            if include_recommendations and f.recommendation:
                lines.append(f"    → {f.recommendation}")
        lines.append("")
    
    if warnings:
        lines.append("⚠️  WARNINGS:")
        for f in warnings:
            col_str = f" [{f.column}]" if f.column else ""
            lines.append(f"  • {f.message}{col_str}")
            if include_recommendations and f.recommendation:
                lines.append(f"    → {f.recommendation}")
        lines.append("")
    
    if info:
        lines.append("ℹ️  INFO:")
        for f in info:
            col_str = f" [{f.column}]" if f.column else ""
            lines.append(f"  • {f.message}{col_str}")
    
    return "\n".join(lines)


def export_report_to_dict(report: QualityReport) -> Dict[str, Any]:
    """
    Export full report to dictionary (for JSON serialization).
    
    Args:
        report: QualityReport
        
    Returns:
        Dict representation
    """
    return report.to_dict()


def export_report_to_markdown(report: QualityReport) -> str:
    """
    Export report as Markdown document.
    
    Args:
        report: QualityReport
        
    Returns:
        Markdown string
    """
    lines = [
        "# Data Quality Report",
        "",
        "## Overview",
        "",
        f"- **Dataset Size:** {report.rows:,} rows × {report.cols} columns",
        f"- **Missing Data:** {report.missing_pct:.2%}",
        f"- **Duplicates:** {report.dupes:,}",
        f"- **Quality Score:** {get_quality_score(report):.1f}/100",
        "",
    ]
    
    if report.memory_bytes:
        lines.append(f"- **Memory Usage:** {report.memory_bytes / 1e6:.2f} MB")
        lines.append("")
    
    if report.sampled:
        lines.append(f"*Note: Analysis performed on sample of {report.sample_size:,} rows*")
        lines.append("")
    
    # Data types
    if report.dtypes_summary:
        lines.append("## Data Types")
        lines.append("")
        for dtype, count in sorted(report.dtypes_summary.items()):
            lines.append(f"- **{dtype}:** {count} columns")
        lines.append("")
    
    # Findings
    if report.findings:
        lines.append("## Findings")
        lines.append("")
        
        errors = filter_findings_by_severity(report, "error")
        warnings = filter_findings_by_severity(report, "warn")
        info = filter_findings_by_severity(report, "info")
        
        if errors:
            lines.append("### 🔴 Errors")
            lines.append("")
            for f in errors:
                col_str = f" (`{f.column}`)" if f.column else ""
                lines.append(f"- {f.message}{col_str}")
                if f.recommendation:
                    lines.append(f"  - *Recommendation:* {f.recommendation}")
            lines.append("")
        
        if warnings:
            lines.append("### ⚠️ Warnings")
            lines.append("")
            for f in warnings:
                col_str = f" (`{f.column}`)" if f.column else ""
                lines.append(f"- {f.message}{col_str}")
                if f.recommendation:
                    lines.append(f"  - *Recommendation:* {f.recommendation}")
            lines.append("")
        
        if info:
            lines.append("### ℹ️ Information")
            lines.append("")
            for f in info:
                col_str = f" (`{f.column}`)" if f.column else ""
                lines.append(f"- {f.message}{col_str}")
            lines.append("")
    
    # Column details
    if report.constant_columns:
        lines.append("## Constant Columns")
        lines.append("")
        for col in report.constant_columns:
            lines.append(f"- `{col}`")
        lines.append("")
    
    if report.high_missing_cols:
        lines.append("## High Missing Data")
        lines.append("")
        lines.append("| Column | Missing % |")
        lines.append("|--------|-----------|")
        for col, pct in sorted(report.high_missing_cols.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{col}` | {pct:.2f}% |")
        lines.append("")
    
    if report.outliers_detected_z or report.outliers_detected_iqr:
        lines.append("## Outliers Detected")
        lines.append("")
        
        all_outliers = {}
        for col, count in report.outliers_detected_z.items():
            all_outliers[col] = all_outliers.get(col, 0) + count
        for col, count in report.outliers_detected_iqr.items():
            all_outliers[col] = all_outliers.get(col, 0) + count
        
        lines.append("| Column | Outlier Count |")
        lines.append("|--------|---------------|")
        for col, count in sorted(all_outliers.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{col}` | {count:,} |")
        lines.append("")
    
    if report.strong_correlations:
        lines.append("## Strong Correlations")
        lines.append("")
        lines.append("| Column 1 | Column 2 | Correlation |")
        lines.append("|----------|----------|-------------|")
        for col1, col2, corr in sorted(report.strong_correlations, key=lambda x: x[2], reverse=True):
            lines.append(f"| `{col1}` | `{col2}` | {corr:.3f} |")
        lines.append("")
    
    return "\n".join(lines)


# ========================================================================================
# QUICK CHECK FUNCTIONS
# ========================================================================================

def quick_quality_check(df: pd.DataFrame) -> str:
    """
    Quick quality check with text summary.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Text summary
    """
    try:
        report = validate_pro(df, QualityOptions(
            check_correlations=False,
            detect_identical_columns=False,
            sample_large_datasets=True
        ))
        
        return report.summary()
        
    except Exception as e:
        return f"❌ Error during quality check: {e}"


def is_data_quality_acceptable(
    df: pd.DataFrame, 
    min_score: float = 70.0
) -> Tuple[bool, float, str]:
    """
    Check if data quality meets minimum threshold.
    
    Args:
        df: DataFrame to check
        min_score: Minimum acceptable quality score (0-100)
        
    Returns:
        Tuple of (is_acceptable, actual_score, message)
    """
    try:
        report = validate_pro(df, QualityOptions(
            check_correlations=False,
            sample_large_datasets=True
        ))
        
        score = get_quality_score(report)
        is_acceptable = score >= min_score
        
        if is_acceptable:
            msg = f"✅ Quality score {score:.1f}/100 meets threshold ({min_score})"
        else:
            msg = f"❌ Quality score {score:.1f}/100 below threshold ({min_score})"
            
            # Add top issues
            actionable = get_actionable_findings(report)[:3]
            if actionable:
                msg += "\nTop issues:"
                for f in actionable:
                    msg += f"\n  - {f.message}"
        
        return is_acceptable, score, msg
        
    except Exception as e:
        logger.exception(f"Error in quality check: {e}")
        return False, 0.0, f"Error: {e}"


# ========================================================================================
# VALIDATION HELPERS
# ========================================================================================

def validate_for_ml(df: pd.DataFrame, target_col: Optional[str] = None) -> QualityReport:
    """
    Validate DataFrame for machine learning readiness.
    
    Performs stricter checks suitable for ML workflows.
    
    Args:
        df: DataFrame to validate
        target_col: Optional target column name
        
    Returns:
        QualityReport with ML-focused findings
    """
    opts = QualityOptions(
        high_missing_threshold=0.30,  # Stricter for ML
        near_constant_threshold=0.95,  # Stricter
        check_correlations=True,
        strong_corr_threshold=0.95,
        detect_mixed_types=True,
        detect_identical_columns=True,
        include_recommendations=True
    )
    
    report = validate_pro(df, opts)
    
    # Add ML-specific findings
    additional_findings = []
    
    # Check target column if specified
    if target_col:
        if target_col not in df.columns:
            additional_findings.append(Finding(
                "missing_target",
                target_col,
                f"Target column '{target_col}' not found in DataFrame.",
                "error",
                {},
                "Ensure the target column name is correct."
            ))
        else:
            # Check target for issues
            target_missing = df[target_col].isna().mean()
            if target_missing > 0:
                additional_findings.append(Finding(
                    "target_missing",
                    target_col,
                    f"Target column has {target_missing:.1%} missing values.",
                    "error" if target_missing > 0.05 else "warn",
                    {"percentage": round(target_missing * 100, 2)},
                    "Target column should not have missing values. Consider removing or imputing."
                ))
    
    # Check for features with high cardinality
    for col in df.columns:
        if col == target_col:
            continue
        
        if df[col].dtype == 'object':
            nunique = df[col].nunique()
            if nunique > len(df) * 0.5:  # More than 50% unique
                additional_findings.append(Finding(
                    "high_cardinality",
                    col,
                    f"High cardinality: {nunique:,} unique values.",
                    "warn",
                    {"nunique": nunique},
                    "Consider encoding, binning, or removing this feature."
                ))
    
    # Merge findings
    all_findings = list(report.findings) + additional_findings
    
    # Create new report with updated findings
    return QualityReport(
        rows=report.rows,
        cols=report.cols,
        missing_pct=report.missing_pct,
        dupes=report.dupes,
        dtypes_summary=report.dtypes_summary,
        memory_bytes=report.memory_bytes,
        constant_columns=report.constant_columns,
        near_constant_columns=report.near_constant_columns,
        unique_columns=report.unique_columns,
        high_missing_cols=report.high_missing_cols,
        outliers_detected_z=report.outliers_detected_z,
        outliers_detected_iqr=report.outliers_detected_iqr,
        textual_anomalies=report.textual_anomalies,
        mixed_type_columns=report.mixed_type_columns,
        identical_columns=report.identical_columns,
        strong_correlations=report.strong_correlations,
        findings=all_findings,
        sampled=report.sampled,
        sample_size=report.sample_size
    )


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
        "name": "data_quality",
        "version": "2.0.0-pro++",
        "description": "Advanced data quality validation module",
        "features": [
            "Comprehensive quality checks",
            "Configurable severity levels",
            "Performance optimizations",
            "ML-readiness validation",
            "Multiple export formats",
            "Backward compatibility"
        ]
    }