"""
Advanced Visualization Charts - Profesjonalne wykresy z Plotly.

Funkcjonalności:
- Custom dark theme (business-style)
- Interactive charts (histogram, scatter, line, heatmap)
- Automatic binning (Freedman-Diaconis)
- WebGL rendering dla dużych zbiorów
- Trendlines (OLS, LOWESS)
- Uncertainty bands
- Feature importance plots
- Statistical overlays (mean, median)
- Auto datetime handling
- Responsive design
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, List, Dict, Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ========================================================================================
# KONFIGURACJA - THEME
# ========================================================================================

# Color palette
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#22d3ee"
ACCENT_COLOR = "#a78bfa"
SUCCESS_COLOR = "#34d399"
WARNING_COLOR = "#f59e0b"
ERROR_COLOR = "#f87171"

# Background colors
PAPER_BG = "#0E1117"
PLOT_BG = "#111827"
GRID_COLOR = "rgba(255,255,255,0.06)"

# Typography
FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
FONT_SIZE = 14

# Chart styling
DEFAULT_OPACITY = 0.85
LARGE_DATA_OPACITY = 0.5
LINE_WIDTH = 2
MARKER_SIZE = 6

# Performance thresholds
WEBGL_THRESHOLD = 50_000
LOW_OPACITY_THRESHOLD = 15_000
MARKER_THRESHOLD = 500

# Binning
MIN_BINS = 8
MAX_BINS = 150

# Feature importance
DEFAULT_TOP_FEATURES = 25

# Template name
TEMPLATE_NAME = "ip_business_dark"

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "charts", level: int = logging.INFO) -> logging.Logger:
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
# THEME SETUP
# ========================================================================================

def _register_custom_template() -> None:
    """
    Rejestruje custom dark theme dla wykresów.
    """
    if TEMPLATE_NAME in pio.templates:
        pio.templates.default = TEMPLATE_NAME
        LOGGER.debug(f"Template '{TEMPLATE_NAME}' już zarejestrowany")
        return
    
    # Base na plotly_dark
    base_template = pio.templates["plotly_dark"]
    
    # Custom layout
    custom_layout = base_template.layout.update(
        # Typography
        font=dict(
            family=FONT_FAMILY,
            size=FONT_SIZE,
            color="#e5e7eb"
        ),
        
        # Backgrounds
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        
        # Color palette
        colorway=[
            PRIMARY_COLOR,
            SECONDARY_COLOR,
            ACCENT_COLOR,
            SUCCESS_COLOR,
            WARNING_COLOR,
            ERROR_COLOR,
            "#60a5fa",
            "#4ade80",
            "#c084fc",
            "#f472b6"
        ],
        
        # Axes
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zeroline=False,
            showspikes=True,
            spikedash="dot",
            spikethickness=1,
            ticks="outside",
            tickcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            linewidth=1
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zeroline=False,
            showspikes=True,
            spikedash="dot",
            spikethickness=1,
            ticks="outside",
            tickcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            linewidth=1
        ),
        
        # Legend
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        
        # Margins
        margin=dict(l=60, r=20, t=60, b=50),
        
        # Hover
        hoverlabel=dict(
            bgcolor="#0b1220",
            bordercolor="#1f2937",
            font=dict(
                color="#e5e7eb",
                size=12
            )
        ),
        
        # Title
        title=dict(
            font=dict(size=18, color="#f9fafb"),
            x=0.02,
            xanchor="left"
        )
    )
    
    # Register template
    pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=custom_layout)
    pio.templates.default = TEMPLATE_NAME
    
    LOGGER.info(f"Registered custom template: {TEMPLATE_NAME}")


# Auto-register on import
_register_custom_template()


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _coerce_datetime(series: pd.Series) -> pd.Series:
    """
    Próbuje skonwertować serie do datetime.
    
    Args:
        series: Serie do konwersji
        
    Returns:
        Datetime series lub oryginalna seria
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    
    try:
        dt_series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        
        # Akceptuj jeśli >60% się sparsowało
        if dt_series.notna().mean() > 0.6:
            return dt_series
    except Exception:
        pass
    
    return series


def _calculate_optimal_bins(data: np.ndarray) -> int:
    """
    Oblicza optymalną liczbę bins używając Freedman-Diaconis rule.
    
    Args:
        data: Numpy array z danymi
        
    Returns:
        Liczba bins
    """
    # Remove NaN
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 2:
        return MIN_BINS
    
    # IQR method
    q75, q25 = np.percentile(clean_data, [75, 25])
    iqr = q75 - q25
    
    if iqr <= 0:
        return MIN_BINS
    
    # Freedman-Diaconis bin width
    bin_width = 2 * iqr * (len(clean_data) ** (-1/3))
    
    if bin_width <= 0:
        return MIN_BINS
    
    # Calculate number of bins
    data_range = clean_data.max() - clean_data.min()
    n_bins = int(np.ceil(data_range / bin_width))
    
    # Clip to reasonable range
    return int(np.clip(n_bins, MIN_BINS, MAX_BINS))


def _is_large_dataset(n: int, threshold: int = WEBGL_THRESHOLD) -> bool:
    """
    Sprawdza czy dataset jest duży.
    
    Args:
        n: Liczba punktów
        threshold: Próg
        
    Returns:
        True jeśli duży
    """
    return n >= threshold


def _apply_common_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    show_legend: bool = True
) -> go.Figure:
    """
    Aplikuje wspólny layout do figury.
    
    Args:
        fig: Plotly figure
        title: Tytuł wykresu
        x_title: Tytuł osi X
        y_title: Tytuł osi Y
        show_legend: Czy pokazywać legendę
        
    Returns:
        Zaktualizowana figura
    """
    if title:
        fig.update_layout(title=dict(text=title))
    
    if x_title:
        fig.update_xaxes(title=x_title)
    
    if y_title:
        fig.update_yaxes(title=y_title)
    
    fig.update_layout(showlegend=show_legend)
    
    return fig


def _drop_na_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Usuwa wiersze z NaN w specified columns.
    
    Args:
        df: DataFrame
        columns: Lista kolumn
        
    Returns:
        Oczyszczony DataFrame
    """
    return df[columns].dropna()


# ========================================================================================
# HISTOGRAM
# ========================================================================================

def histogram(
    df: pd.DataFrame,
    col: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    histnorm: Optional[Literal["percent", "probability", "density"]] = None,
    nbins: Optional[int] = None,
    show_mean: bool = True,
    show_median: bool = True,
    marginal: Optional[Literal["box", "violin", "rug"]] = None,
    opacity: float = DEFAULT_OPACITY,
    **kwargs
) -> go.Figure:
    """
    Zaawansowany histogram z adaptacyjnymi bins i statystykami.
    
    Args:
        df: DataFrame z danymi
        col: Nazwa kolumny do histogramu
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania (optional)
        histnorm: Normalizacja ("percent", "probability", "density")
        nbins: Liczba bins (auto jeśli None)
        show_mean: Czy pokazać linię średniej
        show_median: Czy pokazać linię mediany
        marginal: Marginal plot type ("box", "violin", "rug")
        opacity: Przezroczystość słupków
        **kwargs: Dodatkowe argumenty dla px.histogram
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = histogram(df, "price", show_mean=True, marginal="box")
        >>> fig.show()
    """
    if col not in df.columns:
        LOGGER.error(f"Column '{col}' not found in DataFrame")
        raise ValueError(f"Kolumna '{col}' nie istnieje")
    
    # Try numeric conversion
    numeric_series = pd.to_numeric(df[col], errors="coerce")
    
    # If mostly non-numeric, fall back to categorical bar chart
    if numeric_series.notna().mean() < 0.6:
        LOGGER.debug(f"Column '{col}' is non-numeric, creating categorical bar chart")
        
        value_counts = df[col].astype(str).fillna("(NaN)").value_counts().reset_index()
        value_counts.columns = [col, "count"]
        
        fig = px.bar(
            value_counts,
            x=col,
            y="count",
            color=None,
            opacity=opacity
        )
        
        return _apply_common_layout(
            fig,
            title=title or f"Rozkład: {col}",
            x_title=col,
            y_title="Liczność"
        )
    
    # Numeric histogram
    data = numeric_series.values
    
    # Auto-calculate bins if not provided
    if nbins is None:
        nbins = _calculate_optimal_bins(data)
        LOGGER.debug(f"Auto-calculated {nbins} bins for '{col}'")
    
    # Create histogram
    fig = px.histogram(
        df.assign(_numeric_col=numeric_series),
        x="_numeric_col",
        color=color if (color and color in df.columns) else None,
        nbins=nbins,
        opacity=opacity,
        histnorm=histnorm,
        marginal=marginal,
        **kwargs
    )
    
    # Update hover
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>count=%{y}<extra></extra>"
    )
    
    # Apply layout
    y_label = "Liczność" if not histnorm else histnorm
    fig = _apply_common_layout(
        fig,
        title=title or f"Histogram: {col}",
        x_title=col,
        y_title=y_label
    )
    
    # Add statistical lines
    shapes = []
    annotations = []
    
    if show_mean:
        mean_val = float(np.nanmean(data))
        shapes.append(dict(
            type="line",
            x0=mean_val,
            x1=mean_val,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=SECONDARY_COLOR, width=LINE_WIDTH, dash="dot")
        ))
        annotations.append(dict(
            x=mean_val,
            y=1.02,
            yref="paper",
            text="średnia",
            showarrow=False,
            font=dict(size=11, color=SECONDARY_COLOR)
        ))
    
    if show_median:
        median_val = float(np.nanmedian(data))
        shapes.append(dict(
            type="line",
            x0=median_val,
            x1=median_val,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=WARNING_COLOR, width=LINE_WIDTH, dash="dot")
        ))
        annotations.append(dict(
            x=median_val,
            y=1.02,
            yref="paper",
            text="mediana",
            showarrow=False,
            font=dict(size=11, color=WARNING_COLOR)
        ))
    
    if shapes:
        fig.update_layout(shapes=shapes, annotations=annotations)
    
    return fig


# ========================================================================================
# SCATTER PLOT
# ========================================================================================

def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[Sequence[str]] = None,
    trendline: Optional[Literal["ols", "lowess"]] = "ols",
    add_identity: bool = False,
    opacity: Optional[float] = None,
    **kwargs
) -> go.Figure:
    """
    Rozszerzony scatter plot z trendline i auto-optimization.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X
        y: Kolumna dla osi Y
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania (optional)
        size: Kolumna dla rozmiaru markerów (optional)
        hover_data: Dodatkowe kolumny w hover (optional)
        trendline: Typ trendline ("ols", "lowess", None)
        add_identity: Czy dodać linię y=x
        opacity: Przezroczystość (auto jeśli None)
        **kwargs: Dodatkowe argumenty dla px.scatter
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = scatter(df, "x", "y", trendline="ols", add_identity=True)
        >>> fig.show()
    """
    if x not in df.columns or y not in df.columns:
        LOGGER.error(f"Columns '{x}' or '{y}' not found")
        raise ValueError(f"Kolumny '{x}' lub '{y}' nie istnieją")
    
    # Clean data
    clean_cols = [x, y]
    if color and color in df.columns:
        clean_cols.append(color)
    if size and size in df.columns:
        clean_cols.append(size)
    
    df_clean = _drop_na_columns(df, clean_cols).copy()
    
    if df_clean.empty:
        LOGGER.warning("No data after removing NaN")
        return go.Figure()
    
    # Datetime handling
    df_clean[x] = _coerce_datetime(df_clean[x])
    
    # Numeric conversion for y
    if not pd.api.types.is_datetime64_any_dtype(df_clean[y]):
        df_clean[y] = pd.to_numeric(df_clean[y], errors="coerce")
    
    df_clean = df_clean.dropna()
    
    # Performance optimization
    render_mode = "webgl" if _is_large_dataset(len(df_clean)) else "auto"
    
    if opacity is None:
        opacity = LARGE_DATA_OPACITY if _is_large_dataset(len(df_clean), LOW_OPACITY_THRESHOLD) else DEFAULT_OPACITY
    
    # Create scatter
    fig = px.scatter(
        df_clean,
        x=x,
        y=y,
        color=color if (color and color in df_clean.columns) else None,
        size=size if (size and size in df_clean.columns) else None,
        hover_data=list(hover_data) if hover_data else None,
        trendline=trendline,
        render_mode=render_mode,
        **kwargs
    )
    
    # Style markers
    fig.update_traces(
        marker=dict(
            opacity=opacity,
            line=dict(width=0)
        ),
        selector=dict(mode="markers")
    )
    
    # Hover mode
    fig.update_layout(hovermode="closest")
    
    # Apply layout
    fig = _apply_common_layout(
        fig,
        title=title or f"Scatter: {x} vs {y}",
        x_title=x,
        y_title=y
    )
    
    # Add identity line (y=x)
    if add_identity:
        is_numeric_x = not pd.api.types.is_datetime64_any_dtype(df_clean[x])
        is_numeric_y = not pd.api.types.is_datetime64_any_dtype(df_clean[y])
        
        if is_numeric_x and is_numeric_y:
            x_min = float(df_clean[x].min())
            x_max = float(df_clean[x].max())
            y_min = float(df_clean[y].min())
            y_max = float(df_clean[y].max())
            
            min_val = min(x_min, y_min)
            max_val = max(x_max, y_max)
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#64748b", dash="dash", width=2),
                name="y=x",
                showlegend=True,
                hoverinfo="skip"
            ))
    
    return fig


# ========================================================================================
# LINE PLOT
# ========================================================================================

def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    markers_auto: bool = True,
    show_uncertainty_band: bool = True,
    **kwargs
) -> go.Figure:
    """
    Line plot z auto datetime handling i uncertainty band.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X (często datetime)
        y: Kolumna dla osi Y
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania linii (optional)
        markers_auto: Czy automatycznie dodać markery (dla małych zbiorów)
        show_uncertainty_band: Czy pokazać pasmo niepewności (yhat_lower/upper)
        **kwargs: Dodatkowe argumenty dla px.line
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = line(df, "date", "sales", show_uncertainty_band=True)
        >>> fig.show()
    """
    if x not in df.columns or y not in df.columns:
        LOGGER.error(f"Columns '{x}' or '{y}' not found")
        raise ValueError(f"Kolumny '{x}' lub '{y}' nie istnieją")
    
    # Select columns
    cols_to_use = [x, y]
    if color and color in df.columns:
        cols_to_use.append(color)
    
    df_clean = df[cols_to_use].copy()
    
    # Datetime conversion
    df_clean[x] = _coerce_datetime(df_clean[x])
    
    # Remove NaN and sort
    df_clean = df_clean.dropna(subset=[x, y]).sort_values(by=x)
    
    # Decide on markers
    show_markers = markers_auto and len(df_clean) <= MARKER_THRESHOLD
    
    # Create line plot
    fig = px.line(
        df_clean,
        x=x,
        y=y,
        color=color if (color and color in df_clean.columns) else None,
        markers=show_markers,
        **kwargs
    )
    
    # Apply layout
    fig = _apply_common_layout(
        fig,
        title=title or f"Linia: {y} w czasie",
        x_title=x,
        y_title=y
    )
    
    # Add uncertainty band if available
    if show_uncertainty_band:
        lower_cols = [c for c in df.columns if c.lower().endswith(("yhat_lower", "lower"))]
        upper_cols = [c for c in df.columns if c.lower().endswith(("yhat_upper", "upper"))]
        
        if lower_cols and upper_cols:
            lower_col = lower_cols[0]
            upper_col = upper_cols[0]
            
            try:
                band_df = df[[x, lower_col, upper_col]].copy()
                band_df[x] = _coerce_datetime(band_df[x])
                band_df = band_df.dropna().sort_values(by=x)
                
                if not band_df.empty:
                    # Add uncertainty band
                    fig.add_traces([
                        go.Scatter(
                            x=band_df[x],
                            y=band_df[upper_col],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        go.Scatter(
                            x=band_df[x],
                            y=band_df[lower_col],
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor=f"rgba(74, 144, 226, 0.15)",  # PRIMARY with alpha
                            name="przedział ufności",
                            hoverinfo="skip"
                        )
                    ])
                    
                    LOGGER.debug(f"Added uncertainty band: {lower_col} - {upper_col}")
            except Exception as e:
                LOGGER.warning(f"Failed to add uncertainty band: {e}")
    
    return fig


# ========================================================================================
# CORRELATION HEATMAP
# ========================================================================================

def correlation_heatmap(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: Optional[str] = None
) -> go.Figure:
    """
    Correlation heatmap dla kolumn numerycznych.
    
    Args:
        df: DataFrame z danymi
        method: Metoda korelacji
        title: Tytuł wykresu
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = correlation_heatmap(df, method="spearman")
        >>> fig.show()
    """
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        LOGGER.warning("No numeric columns for correlation")
        return go.Figure()
    
    # Calculate correlation
    corr_matrix = numeric_df.corr(method=method).round(3)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    fig.update_coloraxes(colorbar_title="ρ")
    
    title_text = title or f"Korelacje ({method.capitalize()})"
    fig = _apply_common_layout(fig, title=title_text, show_legend=False)
    
    return fig


# ========================================================================================
# FEATURE IMPORTANCE
# ========================================================================================

def feature_importance(
    importances: Sequence[Tuple[str, float]],
    top: int = DEFAULT_TOP_FEATURES,
    title: Optional[str] = None,
    orientation: Literal["h", "v"] = "h"
) -> go.Figure:
    """
    Feature importance bar chart.
    
    Args:
        importances: Lista tuple (feature_name, importance)
        top: Liczba top features do pokazania
        title: Tytuł wykresu
        orientation: Orientacja ("h" = horizontal, "v" = vertical)
        
    Returns:
        Plotly Figure
        
    Example:
        >>> importances = [("feature1", 0.5), ("feature2", 0.3)]
        >>> fig = feature_importance(importances, top=10)
        >>> fig.show()
    """
    if not importances:
        LOGGER.warning("No importances provided")
        return go.Figure()
    
    # Sort and select top
    sorted_data = sorted(
        list(importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top]
    
    features, values = zip(*sorted_data)
    
    # Create DataFrame
    df = pd.DataFrame({
        "feature": features,
        "importance": values
    })
    
    # Create bar chart
    if orientation == "h":
        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues"
        )
        x_label, y_label = "Ważność", "Cecha"
    else:
        fig = px.bar(
            df,
            x="feature",
            y="importance",
            orientation="v",
            color="importance",
            color_continuous_scale="Blues"
        )
        x_label, y_label = "Cecha", "Ważność"
    
    fig = _apply_common_layout(
        fig,
        title=title or "Ważność cech",
        x_title=x_label,
        y_title=y_label,
        show_legend=False
    )
    
    return fig


# ========================================================================================
# UTILITIES
# ========================================================================================

def set_theme(theme: Literal["dark", "light", "default"] = "dark") -> None:
    """
    Zmienia globalny theme dla wykresów.
    
    Args:
        theme: Nazwa theme
    """
    if theme == "dark":
        pio.templates.default = TEMPLATE_NAME
    elif theme == "light":
        pio.templates.default = "plotly_white"
    else:
        pio.templates.default = "plotly"
    
    LOGGER.info(f"Theme changed to: {theme}")


def get_available_themes() -> List[str]:
    """
    Zwraca listę dostępnych themes.
    
    Returns:
        Lista nazw themes
    """
    return list(pio.templates.keys())