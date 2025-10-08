# profiling_report.py — PRO++ Enhanced Edition
"""
Enhanced Data Profiling Module with ydata-profiling integration.

Features:
- Intelligent sampling for large datasets
- Fallback HTML reports when ydata-profiling unavailable
- Memory-safe operations with configurable limits
- Comprehensive error handling and logging
- Type-safe operations with validation
- Performance optimizations
- Backward compatibility maintained
"""

from __future__ import annotations

import html
import logging
import traceback
import gc
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd
import numpy as np

# ========================================================================================
# IMPORTS & COMPATIBILITY
# ========================================================================================

# Try ydata-profiling (new) first, then pandas-profiling (legacy)
ProfileReport = None
PROFILER_AVAILABLE = False
PROFILER_VERSION = "none"

try:
    from ydata_profiling import ProfileReport  # type: ignore
    PROFILER_AVAILABLE = True
    PROFILER_VERSION = "ydata-profiling"
except ImportError:
    try:
        from pandas_profiling import ProfileReport  # type: ignore
        PROFILER_AVAILABLE = True
        PROFILER_VERSION = "pandas-profiling"
    except ImportError:
        pass

# ========================================================================================
# LOGGING
# ========================================================================================

logger = logging.getLogger(__name__)

# ========================================================================================
# CONFIGURATION
# ========================================================================================

# Memory limits (MB)
MAX_DATAFRAME_MEMORY_MB = 500
MAX_PROFILE_MEMORY_MB = 1000

# HTML themes
VALID_THEMES = {"flatly", "united", "simplex", "cosmo", "sandstone", "darkly"}

# Sampling defaults
DEFAULT_SAMPLE_MAX = 5_000
DEFAULT_MAX_COLS = 1_000

# Preview limits for fallback HTML
FALLBACK_PREVIEW_ROWS = 20
FALLBACK_PREVIEW_COLS = 20

# ========================================================================================
# TYPES
# ========================================================================================

Mode = Literal["minimal", "explorative", "full"]


@dataclass(frozen=True)
class ProfileOptions:
    """Configuration for profile report generation."""
    
    mode: Mode = "minimal"
    sample_max: int = DEFAULT_SAMPLE_MAX
    use_random_sample: bool = True
    random_state: int = 42
    correlations: bool = False
    max_cols: int = DEFAULT_MAX_COLS
    html_theme: str = "flatly"
    full_width: bool = True
    
    def __post_init__(self):
        """Validate options after initialization."""
        # Validate sample_max
        if self.sample_max <= 0:
            object.__setattr__(self, 'sample_max', DEFAULT_SAMPLE_MAX)
            logger.warning(f"Invalid sample_max, using default: {DEFAULT_SAMPLE_MAX}")
        
        # Validate max_cols
        if self.max_cols <= 0:
            object.__setattr__(self, 'max_cols', DEFAULT_MAX_COLS)
            logger.warning(f"Invalid max_cols, using default: {DEFAULT_MAX_COLS}")
        
        # Validate theme
        if self.html_theme not in VALID_THEMES:
            logger.warning(f"Invalid theme '{self.html_theme}', using 'flatly'")
            object.__setattr__(self, 'html_theme', 'flatly')


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _coerce_mode(m: str) -> Mode:
    """
    Safely coerce mode string to Mode type.
    
    Args:
        m: Mode string
        
    Returns:
        Valid Mode literal
    """
    if not isinstance(m, str):
        logger.warning(f"Invalid mode type: {type(m)}, defaulting to 'minimal'")
        return "minimal"
    
    m = (m or "minimal").strip().lower()
    
    if m == "explorative":
        return "explorative"
    elif m == "full":
        return "full"
    else:
        return "minimal"


def _estimate_memory_mb(df: pd.DataFrame) -> float:
    """
    Estimate DataFrame memory usage in MB.
    
    Args:
        df: DataFrame to estimate
        
    Returns:
        Memory usage in MB
    """
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        return float(memory_bytes / 1e6)
    except Exception as e:
        logger.warning(f"Failed to estimate memory: {e}")
        # Fallback estimation: assume 8 bytes per cell
        return float(df.size * 8 / 1e6)


def _safe_sample_dataframe(
    df: pd.DataFrame,
    sample_max: int,
    use_random: bool,
    random_state: int
) -> tuple[pd.DataFrame, str]:
    """
    Safely sample DataFrame with memory checks.
    
    Args:
        df: Source DataFrame
        sample_max: Maximum rows
        use_random: Whether to use random sampling
        random_state: Random seed
        
    Returns:
        Tuple of (sampled_df, note_message)
    """
    original_rows = len(df)
    
    if original_rows <= sample_max:
        return df.copy(), f"Pełny zbiór ({original_rows:,} wierszy)."
    
    # Sample
    try:
        if use_random:
            df_sampled = df.sample(n=sample_max, random_state=random_state)
            note = f"Próbka losowa {sample_max:,} z {original_rows:,} wierszy."
        else:
            df_sampled = df.head(sample_max)
            note = f"Pierwszych {sample_max:,} z {original_rows:,} wierszy."
    except Exception as e:
        logger.warning(f"Sampling failed: {e}, using head()")
        df_sampled = df.head(sample_max)
        note = f"Pierwszych {sample_max:,} z {original_rows:,} wierszy (fallback)."
    
    # Verify memory after sampling
    memory_mb = _estimate_memory_mb(df_sampled)
    
    if memory_mb > MAX_DATAFRAME_MEMORY_MB:
        logger.warning(
            f"Sampled DataFrame still too large ({memory_mb:.1f} MB), "
            f"reducing to {MAX_DATAFRAME_MEMORY_MB} MB limit"
        )
        
        # Further reduce
        reduction_factor = MAX_DATAFRAME_MEMORY_MB / memory_mb
        new_sample_size = int(sample_max * reduction_factor * 0.9)  # 10% safety margin
        new_sample_size = max(100, new_sample_size)  # Minimum 100 rows
        
        df_sampled = df_sampled.head(new_sample_size)
        note += f" Dodatkowo zredukowano do {new_sample_size:,} wierszy (limit pamięci)."
    
    return df_sampled, note


def _limit_columns(
    df: pd.DataFrame,
    max_cols: int
) -> tuple[pd.DataFrame, str]:
    """
    Limit number of columns with informative message.
    
    Args:
        df: Source DataFrame
        max_cols: Maximum columns
        
    Returns:
        Tuple of (limited_df, note_message)
    """
    original_cols = df.shape[1]
    
    if original_cols <= max_cols:
        return df, ""
    
    logger.info(f"Limiting columns from {original_cols} to {max_cols}")
    
    df_limited = df.iloc[:, :max_cols].copy()
    note = f"Ograniczono do pierwszych {max_cols} kolumn (z {original_cols}). "
    
    return df_limited, note


def _html_note(text: str, color: str = "#888") -> str:
    """
    Create styled HTML note.
    
    Args:
        text: Note text
        color: Text color
        
    Returns:
        HTML string
    """
    return (
        f"<p style='color:{color};font-size:0.9em;margin:10px 0;'>"
        f"<em>{html.escape(text)}</em></p>"
    )


def _sanitize_html(content: str) -> str:
    """
    Sanitize HTML content for safe display.
    
    Args:
        content: HTML content
        
    Returns:
        Sanitized HTML
    """
    # Basic sanitization - prevent script injection
    content = content.replace("<script", "&lt;script")
    content = content.replace("</script>", "&lt;/script&gt;")
    content = content.replace("javascript:", "")
    content = content.replace("onerror=", "")
    content = content.replace("onclick=", "")
    
    return content


def _compute_basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics safely.
    
    Args:
        df: DataFrame
        
    Returns:
        Dict with basic stats
    """
    try:
        rows, cols = df.shape
        total_cells = df.size or 1
        missing_count = df.isna().sum().sum()
        missing_pct = float(missing_count / total_cells * 100.0)
        dupes = int(df.duplicated().sum())
        memory_mb = _estimate_memory_mb(df)
        
        # Type breakdown
        type_counts = df.dtypes.astype(str).value_counts().to_dict()
        
        # Numeric stats
        numeric_cols = df.select_dtypes(include=np.number).columns
        num_numeric = len(numeric_cols)
        
        # Categorical stats
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_categorical = len(cat_cols)
        
        # Datetime stats
        dt_cols = df.select_dtypes(include=['datetime64', 'timedelta64']).columns
        num_datetime = len(dt_cols)
        
        return {
            "rows": rows,
            "cols": cols,
            "missing_pct": missing_pct,
            "dupes": dupes,
            "memory_mb": memory_mb,
            "type_counts": type_counts,
            "num_numeric": num_numeric,
            "num_categorical": num_categorical,
            "num_datetime": num_datetime
        }
    except Exception as e:
        logger.error(f"Error computing basic stats: {e}")
        return {
            "rows": len(df),
            "cols": df.shape[1],
            "missing_pct": 0.0,
            "dupes": 0,
            "memory_mb": 0.0,
            "type_counts": {},
            "num_numeric": 0,
            "num_categorical": 0,
            "num_datetime": 0
        }


# ========================================================================================
# FALLBACK HTML REPORT
# ========================================================================================

def _fallback_kpi_html(
    df: pd.DataFrame,
    title: str,
    note: str,
    err: Optional[str] = None
) -> str:
    """
    Generate lightweight fallback HTML report with enhanced styling.
    
    Args:
        df: DataFrame to profile
        title: Report title
        note: Note message
        err: Optional error message
        
    Returns:
        HTML report string
    """
    # Compute stats
    stats = _compute_basic_stats(df)
    
    # Type breakdown table
    type_rows = ""
    if stats["type_counts"]:
        type_rows = "".join([
            f"<tr><td>{html.escape(str(dtype))}</td><td>{count}</td></tr>"
            for dtype, count in sorted(stats["type_counts"].items())
        ])
        type_table = f"""
        <div class="card">
            <h3>📊 Typy danych</h3>
            <table>
                <thead><tr><th>Typ</th><th>Liczba kolumn</th></tr></thead>
                <tbody>{type_rows}</tbody>
            </table>
        </div>
        """
    else:
        type_table = ""
    
    # Preview table
    try:
        preview_df = df.head(FALLBACK_PREVIEW_ROWS).iloc[:, :FALLBACK_PREVIEW_COLS]
        head_html = preview_df.to_html(
            index=False,
            border=0,
            classes="preview-table",
            max_rows=FALLBACK_PREVIEW_ROWS
        )
        
        col_info = ""
        if df.shape[1] > FALLBACK_PREVIEW_COLS:
            col_info = f"<p><em>Pokazano pierwsze {FALLBACK_PREVIEW_COLS} z {df.shape[1]} kolumn</em></p>"
        
        preview_section = f"""
        <div class="card">
            <h3>📄 Podgląd danych</h3>
            {col_info}
            {head_html}
        </div>
        """
    except Exception as e:
        logger.error(f"Error generating preview table: {e}")
        preview_section = """
        <div class="card">
            <h3>📄 Podgląd danych</h3>
            <p><em>Nie udało się wygenerować podglądu.</em></p>
        </div>
        """
    
    # Error block
    err_block = ""
    if err:
        err_block = f"""
        <div class="card error-card">
            <h3>⚠️ Szczegóły błędu</h3>
            <pre class="error-pre">{html.escape(err)}</pre>
        </div>
        """
    
    # Profiler status
    profiler_status = f"""
    <div class="info-banner">
        <strong>ℹ️ Informacja:</strong> 
        {f'Używam {PROFILER_VERSION}' if PROFILER_AVAILABLE else 'Pakiet ydata-profiling nie jest zainstalowany'}.
        Wyświetlam uproszczony raport.
    </div>
    """
    
    return f"""<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                         "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ 
            color: #2c3e50; 
            margin-bottom: 10px;
            font-size: 2em;
        }}
        h3 {{ color: #34495e; margin-top: 0; }}
        .card {{
            border: 1px solid #e1e8ed;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            background: #fff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .error-card {{ 
            border-color: #ffcdd2; 
            background: #fff5f5; 
        }}
        .info-banner {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .kpi {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            transition: transform 0.2s;
        }}
        .kpi:hover {{ transform: translateY(-2px); }}
        .kpi-label {{ 
            font-size: 0.85em; 
            opacity: 0.95; 
            margin-bottom: 8px;
            font-weight: 500;
        }}
        .kpi-value {{ 
            font-size: 1.8em; 
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{ 
            font-size: 0.8em; 
            color: #666; 
            margin-bottom: 4px;
        }}
        .stat-value {{ 
            font-size: 1.3em; 
            font-weight: bold; 
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9em;
        }}
        th, td {{
            border-bottom: 1px solid #e1e8ed;
            padding: 10px;
            text-align: left;
        }}
        thead th {{
            background: #f8f9fa;
            font-weight: 600;
            position: sticky;
            top: 0;
            border-bottom: 2px solid #dee2e6;
        }}
        tbody tr:hover {{ background: #f8f9fa; }}
        .preview-table {{ 
            max-height: 400px; 
            overflow: auto;
            display: block;
        }}
        .error-pre {{
            background: #f8f8f8;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #e0e0e0;
        }}
        small {{ color: #666; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            background: #e3f2fd;
            color: #1976d2;
            margin-left: 8px;
            font-weight: 500;
        }}
        @media (max-width: 768px) {{
            .grid {{ grid-template-columns: 1fr 1fr; }}
            .kpi-value {{ font-size: 1.4em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)} <span class="badge">Wersja Lite</span></h1>
        {_html_note(note, "#555")}
        
        {profiler_status}
        
        <div class="grid">
            <div class="kpi">
                <div class="kpi-label">Wiersze</div>
                <div class="kpi-value">{stats['rows']:,}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Kolumny</div>
                <div class="kpi-value">{stats['cols']:,}</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Braki</div>
                <div class="kpi-value">{stats['missing_pct']:.2f}%</div>
            </div>
            <div class="kpi">
                <div class="kpi-label">Duplikaty</div>
                <div class="kpi-value">{stats['dupes']:,}</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Dodatkowe statystyki</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Pamięć</div>
                    <div class="stat-value">{stats['memory_mb']:.1f} MB</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Numeryczne</div>
                    <div class="stat-value">{stats['num_numeric']}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Kategoryczne</div>
                    <div class="stat-value">{stats['num_categorical']}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Datetime</div>
                    <div class="stat-value">{stats['num_datetime']}</div>
                </div>
            </div>
        </div>
        
        {type_table}
        {preview_section}
        {err_block}
        
        <p style="text-align: center; margin-top: 30px;">
            <small>
                Wersja lite • 
                {f'{PROFILER_VERSION} niedostępny' if PROFILER_AVAILABLE else 'ydata-profiling nie zainstalowany'} lub wystąpił błąd • 
                Wygenerowano raport podstawowy
            </small>
        </p>
    </div>
</body>
</html>"""


# ========================================================================================
# MAIN PROFILE GENERATION
# ========================================================================================

def _make_profile_html_pro(
    df: pd.DataFrame,
    title: str,
    opts: ProfileOptions
) -> str:
    """
    Internal implementation for profile HTML generation.
    
    Args:
        df: DataFrame to profile
        title: Report title
        opts: Profile options
        
    Returns:
        HTML string
    """
    # Walidacja wejścia
    if df is None:
        logger.error("DataFrame is None")
        return "<p><strong>⚠️ Błąd: DataFrame jest None.</strong></p>"
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Invalid type: expected DataFrame, got {type(df)}")
        return f"<p><strong>⚠️ Błąd: oczekiwano DataFrame, otrzymano {type(df)}.</strong></p>"
    
    if df.empty:
        logger.warning("DataFrame is empty")
        return "<p><strong>ℹ️ DataFrame jest pusty - brak danych do profilowania.</strong></p>"
    
    original_rows = len(df)
    original_cols = df.shape[1]
    
    logger.info(f"Starting profile generation: {original_rows:,} rows × {original_cols} cols")
    
    # Limit columns
    df_work, col_note = _limit_columns(df, opts.max_cols)
    
    # Sample for performance
    df_work, sample_note = _safe_sample_dataframe(
        df_work,
        opts.sample_max,
        opts.use_random_sample,
        opts.random_state
    )
    
    note = sample_note + col_note
    
    # Build correlation config
    corr_cfg = None
    if opts.correlations:
        corr_cfg = {
            "pearson": True,
            "spearman": True,
            "kendall": False,  # Very slow
            "phi_k": False,    # Very slow
            "cramers": False,  # Very slow
        }
        logger.info("Correlations enabled (Pearson + Spearman)")
    
    # Check if profiler available
    if not PROFILER_AVAILABLE or ProfileReport is None:
        logger.warning(f"Profiler not available (version: {PROFILER_VERSION}), using fallback")
        return _fallback_kpi_html(
            df_work,
            title,
            note,
            err="Pakiet ydata-profiling nie jest zainstalowany. Zainstaluj: pip install ydata-profiling"
        )
    
    # Generate full profile with error handling
    try:
        logger.info(f"Generating profile (mode: {opts.mode}, profiler: {PROFILER_VERSION})")
        
        # Determine mode flags
        minimal = opts.mode == "minimal"
        explorative = opts.mode == "explorative"
        
        # Create profile
        profile = ProfileReport(
            df_work,
            title=title,
            minimal=minimal,
            explorative=explorative,
            correlations=corr_cfg,
            progress_bar=opts.progress_bar,
            pool_size=opts.pool_size,
            infer_dtypes=True,
            html={
                "style": {
                    "full_width": opts.full_width,
                    "theme": opts.html_theme
                }
            },
        )
        
        # Generate HTML
        logger.info("Converting profile to HTML")
        html_out = profile.to_html()
        
        # Sanitize
        html_out = _sanitize_html(html_out)
        
        # Add note at the top
        result = _html_note(note, "#555") + html_out
        
        logger.info("Profile generated successfully")
        
        # Cleanup
        del profile
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.exception(f"Profile generation failed: {e}")
        
        # Fallback to lite HTML with traceback
        return _fallback_kpi_html(
            df_work,
            title,
            note,
            err=traceback.format_exc()
        )


# ========================================================================================
# PUBLIC API
# ========================================================================================

def make_profile_html(
    df: pd.DataFrame,
    title: str = "Data Profile",
    mode: str = "minimal",
    sample_max: int = DEFAULT_SAMPLE_MAX,
    correlations: bool = False,
) -> str:
    """
    Tworzy bezpieczny i szybki raport profilujący (HTML) dla danych.
    
    Backward compatible API - zachowuje ten sam podpis co wcześniej.
    
    Args:
        df: DataFrame to profile
        title: Report title
        mode: Profile mode ('minimal', 'explorative', 'full')
        sample_max: Maximum rows to sample
        correlations: Whether to compute correlations
        
    Returns:
        HTML report as string
        
    Examples:
        >>> html = make_profile_html(df, title="My Data")
        >>> html = make_profile_html(df, mode="full", correlations=True)
    """
    opts = ProfileOptions(
        mode=_coerce_mode(mode),
        sample_max=sample_max,
        correlations=bool(correlations),
    )
    return _make_profile_html_pro(df, title=title, opts=opts)


def make_profile_html_pro(
    df: pd.DataFrame,
    title: str = "Data Profile",
    opts: ProfileOptions = ProfileOptions(),
) -> str:
    """
    Nowe, bogatsze API z pełną konfiguracją przez ProfileOptions.
    
    Args:
        df: DataFrame to profile
        title: Report title
        opts: ProfileOptions object with advanced settings
        
    Returns:
        HTML report as string
        
    Examples:
        >>> opts = ProfileOptions(mode="explorative", correlations=True)
        >>> html = make_profile_html_pro(df, opts=opts)
    """
    return _make_profile_html_pro(df, title=title, opts=opts)


def get_profiler_info() -> dict:
    """
    Get information about the available profiler.
    
    Returns:
        Dict with profiler information
    """
    return {
        "available": PROFILER_AVAILABLE,
        "version": PROFILER_VERSION,
        "package": PROFILER_VERSION if PROFILER_AVAILABLE else None
    }