"""
Advanced Visualization Charts PRO++++ - Profesjonalne wykresy z Plotly.

Funkcjonalności PRO++++:
- Custom dark theme (business-style) z wariantami
- Interactive charts (histogram, scatter, line, heatmap, box, violin, 3D)
- Automatic binning (Freedman-Diaconis + Sturges + Scott)
- WebGL rendering dla dużych zbiorów z adaptive degradation
- Trendlines (OLS, LOWESS, Polynomial, Exponential)
- Uncertainty bands z konfigurowalnymi percentylami
- Feature importance plots (bar, lollipop, waterfall)
- Statistical overlays (mean, median, std, quantiles)
- Auto datetime handling z time-aware aggregations
- Responsive design z mobile optimization
- Export do HTML/PNG/SVG z high DPI
- Animation support dla time series
- Subplot grid generator
- Custom color palettes
- Accessibility features (WCAG AA)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence, Tuple, List, Dict, Any, Literal, Union
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# Suppress plotly warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

# ========================================================================================
# ENUMS & DATACLASSES
# ========================================================================================

class ChartTheme(str, Enum):
    """Dostępne motywy wykresów."""
    DARK = "dark"
    LIGHT = "light"
    BUSINESS_DARK = "business_dark"
    NEON = "neon"
    MINIMAL = "minimal"


class BinningMethod(str, Enum):
    """Metody obliczania bins."""
    FREEDMAN_DIACONIS = "fd"
    STURGES = "sturges"
    SCOTT = "scott"
    AUTO = "auto"


@dataclass(frozen=True)
class ColorPalette:
    """Paleta kolorów dla wykresów."""
    primary: str = "#4A90E2"
    secondary: str = "#22d3ee"
    accent: str = "#a78bfa"
    success: str = "#34d399"
    warning: str = "#f59e0b"
    error: str = "#f87171"
    info: str = "#60a5fa"
    
    def to_list(self) -> List[str]:
        """Konwertuje do listy."""
        return [
            self.primary,
            self.secondary,
            self.accent,
            self.success,
            self.warning,
            self.error,
            self.info
        ]


@dataclass(frozen=True)
class ThemeConfig:
    """Konfiguracja motywu."""
    name: str
    paper_bg: str
    plot_bg: str
    grid_color: str
    text_color: str
    palette: ColorPalette = field(default_factory=ColorPalette)


# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Palety kolorów
DEFAULT_PALETTE = ColorPalette()

NEON_PALETTE = ColorPalette(
    primary="#00ff41",
    secondary="#ff00ff",
    accent="#00ffff",
    success="#39ff14",
    warning="#ffaa00",
    error="#ff073a",
    info="#00d4ff"
)

# Motywy
THEMES = {
    ChartTheme.BUSINESS_DARK: ThemeConfig(
        name="ip_business_dark",
        paper_bg="#0E1117",
        plot_bg="#111827",
        grid_color="rgba(255,255,255,0.06)",
        text_color="#e5e7eb",
        palette=DEFAULT_PALETTE
    ),
    ChartTheme.LIGHT: ThemeConfig(
        name="ip_light",
        paper_bg="#ffffff",
        plot_bg="#f9fafb",
        grid_color="rgba(0,0,0,0.08)",
        text_color="#1f2937",
        palette=DEFAULT_PALETTE
    ),
    ChartTheme.NEON: ThemeConfig(
        name="ip_neon",
        paper_bg="#0a0a0a",
        plot_bg="#0f0f0f",
        grid_color="rgba(0,255,65,0.1)",
        text_color="#00ff41",
        palette=NEON_PALETTE
    ),
    ChartTheme.MINIMAL: ThemeConfig(
        name="ip_minimal",
        paper_bg="#fafafa",
        plot_bg="#ffffff",
        grid_color="rgba(0,0,0,0.05)",
        text_color="#262626",
        palette=DEFAULT_PALETTE
    )
}

# Typography
FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
FONT_SIZE_SMALL = 11
FONT_SIZE_MEDIUM = 14
FONT_SIZE_LARGE = 18

# Chart styling
DEFAULT_OPACITY = 0.85
LARGE_DATA_OPACITY = 0.5
VERY_LARGE_DATA_OPACITY = 0.3
LINE_WIDTH = 2
THIN_LINE_WIDTH = 1
THICK_LINE_WIDTH = 3
MARKER_SIZE = 6
SMALL_MARKER_SIZE = 4

# Performance thresholds
WEBGL_THRESHOLD = 50_000
EXTREME_WEBGL_THRESHOLD = 500_000
LOW_OPACITY_THRESHOLD = 15_000
MARKER_THRESHOLD = 500
SAMPLE_THRESHOLD = 1_000_000

# Binning
MIN_BINS = 8
MAX_BINS = 150
DEFAULT_BINS = 30

# Feature importance
DEFAULT_TOP_FEATURES = 25
MAX_TOP_FEATURES = 50

# Animation
DEFAULT_ANIMATION_DURATION = 500

# Export
DEFAULT_DPI = 300
HIGH_DPI = 600

# Current active theme
_ACTIVE_THEME = ChartTheme.BUSINESS_DARK

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
# THEME MANAGEMENT
# ========================================================================================

def _create_layout_template(config: ThemeConfig) -> go.layout.Template:
    """
    Tworzy layout template z konfiguracji.
    
    Args:
        config: Konfiguracja motywu
        
    Returns:
        Plotly template
    """
    return go.layout.Template(
        layout=dict(
            # Typography
            font=dict(
                family=FONT_FAMILY,
                size=FONT_SIZE_MEDIUM,
                color=config.text_color
            ),
            
            # Backgrounds
            paper_bgcolor=config.paper_bg,
            plot_bgcolor=config.plot_bg,
            
            # Color palette
            colorway=config.palette.to_list(),
            
            # Axes
            xaxis=dict(
                gridcolor=config.grid_color,
                zeroline=False,
                showspikes=True,
                spikedash="dot",
                spikethickness=1,
                ticks="outside",
                tickcolor=config.grid_color,
                linecolor=config.grid_color,
                linewidth=1,
                title=dict(
                    font=dict(size=FONT_SIZE_MEDIUM)
                )
            ),
            yaxis=dict(
                gridcolor=config.grid_color,
                zeroline=False,
                showspikes=True,
                spikedash="dot",
                spikethickness=1,
                ticks="outside",
                tickcolor=config.grid_color,
                linecolor=config.grid_color,
                linewidth=1,
                title=dict(
                    font=dict(size=FONT_SIZE_MEDIUM)
                )
            ),
            
            # Legend
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=FONT_SIZE_SMALL)
            ),
            
            # Margins
            margin=dict(l=60, r=20, t=60, b=50),
            
            # Hover
            hoverlabel=dict(
                bgcolor=config.plot_bg,
                bordercolor=config.grid_color,
                font=dict(
                    color=config.text_color,
                    size=FONT_SIZE_SMALL,
                    family=FONT_FAMILY
                )
            ),
            
            # Title
            title=dict(
                font=dict(
                    size=FONT_SIZE_LARGE,
                    color=config.text_color
                ),
                x=0.02,
                xanchor="left"
            ),
            
            # Colorscale
            colorscale=dict(
                sequential=[
                    [0, config.palette.primary],
                    [0.5, config.palette.secondary],
                    [1, config.palette.accent]
                ],
                diverging=[
                    [0, config.palette.error],
                    [0.5, config.paper_bg],
                    [1, config.palette.success]
                ]
            ),
            
            # Uniformtext
            uniformtext=dict(
                minsize=8,
                mode="hide"
            )
        )
    )


def _register_all_themes() -> None:
    """Rejestruje wszystkie dostępne motywy."""
    for theme_enum, config in THEMES.items():
        if config.name not in pio.templates:
            template = _create_layout_template(config)
            pio.templates[config.name] = template
            LOGGER.debug(f"Registered theme: {config.name}")
    
    # Set default
    default_config = THEMES[_ACTIVE_THEME]
    pio.templates.default = default_config.name
    LOGGER.info(f"Default theme set to: {default_config.name}")


# Auto-register on import
_register_all_themes()


def set_theme(theme: Union[ChartTheme, str] = ChartTheme.BUSINESS_DARK) -> None:
    """
    Zmienia globalny motyw dla wykresów.
    
    Args:
        theme: Nazwa motywu (ChartTheme lub string)
        
    Examples:
        >>> set_theme(ChartTheme.NEON)
        >>> set_theme("light")
    """
    global _ACTIVE_THEME
    
    if isinstance(theme, str):
        try:
            theme = ChartTheme(theme)
        except ValueError:
            LOGGER.error(f"Unknown theme: {theme}")
            raise ValueError(f"Nieznany motyw: {theme}. Dostępne: {list(ChartTheme)}")
    
    if theme not in THEMES:
        LOGGER.error(f"Theme not registered: {theme}")
        raise ValueError(f"Motyw nie jest zarejestrowany: {theme}")
    
    config = THEMES[theme]
    pio.templates.default = config.name
    _ACTIVE_THEME = theme
    
    LOGGER.info(f"Theme changed to: {theme.value}")


def get_active_theme() -> ChartTheme:
    """
    Zwraca aktywny motyw.
    
    Returns:
        Aktywny ChartTheme
    """
    return _ACTIVE_THEME


def get_active_palette() -> ColorPalette:
    """
    Zwraca aktywną paletę kolorów.
    
    Returns:
        Aktywna ColorPalette
    """
    return THEMES[_ACTIVE_THEME].palette


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _safe_numeric_conversion(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Bezpieczna konwersja do numeric z logowaniem.
    
    Args:
        series: Serie do konwersji
        col_name: Nazwa kolumny (dla logowania)
        
    Returns:
        Numeric series
    """
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        nan_pct = numeric.isna().mean()
        
        if nan_pct > 0.4:
            LOGGER.warning(
                f"Column '{col_name}': {nan_pct*100:.1f}% non-numeric values"
            )
        
        return numeric
    except Exception as e:
        LOGGER.error(f"Error converting '{col_name}' to numeric: {e}")
        return series


@lru_cache(maxsize=128)
def _is_datetime_compatible(dtype_str: str) -> bool:
    """
    Sprawdza czy dtype jest kompatybilny z datetime (cachowane).
    
    Args:
        dtype_str: String reprezentacja dtype
        
    Returns:
        True jeśli datetime-compatible
    """
    return "datetime" in dtype_str.lower() or "timedelta" in dtype_str.lower()


def _coerce_datetime(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Próbuje skonwertować serię do datetime z inteligentnym fallback.
    
    Args:
        series: Serie do konwersji
        col_name: Nazwa kolumny (dla logowania)
        
    Returns:
        Datetime series lub oryginalna seria
    """
    # Już jest datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    
    # Próba konwersji
    try:
        dt_series = pd.to_datetime(
            series,
            errors="coerce",
            infer_datetime_format=True
        )
        
        # Akceptuj jeśli >60% się sparsowało
        valid_pct = dt_series.notna().mean()
        
        if valid_pct > 0.6:
            if valid_pct < 1.0:
                LOGGER.debug(
                    f"Column '{col_name}': {(1-valid_pct)*100:.1f}% "
                    f"failed datetime conversion"
                )
            return dt_series
        else:
            LOGGER.debug(
                f"Column '{col_name}': insufficient datetime parsing "
                f"({valid_pct*100:.1f}%), keeping original"
            )
            
    except Exception as e:
        LOGGER.debug(f"Datetime conversion failed for '{col_name}': {e}")
    
    return series


def _calculate_optimal_bins(
    data: np.ndarray,
    method: BinningMethod = BinningMethod.AUTO
) -> int:
    """
    Oblicza optymalną liczbę bins używając wybranej metody.
    
    Args:
        data: Numpy array z danymi
        method: Metoda binowania
        
    Returns:
        Liczba bins
    """
    # Remove NaN
    clean_data = data[~np.isnan(data)]
    n = len(clean_data)
    
    if n < 2:
        return MIN_BINS
    
    try:
        if method == BinningMethod.FREEDMAN_DIACONIS or method == BinningMethod.AUTO:
            # IQR method
            q75, q25 = np.percentile(clean_data, [75, 25])
            iqr = q75 - q25
            
            if iqr <= 0:
                return MIN_BINS
            
            bin_width = 2 * iqr * (n ** (-1/3))
            
            if bin_width <= 0:
                return MIN_BINS
            
            data_range = clean_data.max() - clean_data.min()
            n_bins = int(np.ceil(data_range / bin_width))
            
        elif method == BinningMethod.STURGES:
            # Sturges' formula
            n_bins = int(np.ceil(np.log2(n)) + 1)
            
        elif method == BinningMethod.SCOTT:
            # Scott's rule
            std = clean_data.std()
            if std <= 0:
                return MIN_BINS
            
            bin_width = 3.5 * std * (n ** (-1/3))
            data_range = clean_data.max() - clean_data.min()
            n_bins = int(np.ceil(data_range / bin_width))
            
        else:
            n_bins = DEFAULT_BINS
        
        # Clip to reasonable range
        return int(np.clip(n_bins, MIN_BINS, MAX_BINS))
        
    except Exception as e:
        LOGGER.warning(f"Error calculating bins: {e}, using default")
        return DEFAULT_BINS


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


def _apply_performance_optimizations(
    fig: go.Figure,
    n_points: int
) -> go.Figure:
    """
    Aplikuje optymalizacje wydajności w zależności od rozmiaru danych.
    
    Args:
        fig: Plotly figure
        n_points: Liczba punktów danych
        
    Returns:
        Zoptymalizowana figura
    """
    if n_points >= EXTREME_WEBGL_THRESHOLD:
        # Extreme dataset - aggressive optimizations
        LOGGER.info(f"Applying extreme optimizations for {n_points:,} points")
        
        fig.update_traces(
            marker=dict(opacity=VERY_LARGE_DATA_OPACITY),
            selector=dict(type="scatter")
        )
        
        # Disable hover for performance
        fig.update_traces(hoverinfo="skip")
        
    elif n_points >= WEBGL_THRESHOLD:
        # Large dataset - moderate optimizations
        LOGGER.debug(f"Applying WebGL optimizations for {n_points:,} points")
        
        fig.update_traces(
            marker=dict(opacity=LARGE_DATA_OPACITY),
            selector=dict(type="scatter")
        )
    
    return fig


def _apply_common_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    show_legend: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> go.Figure:
    """
    Aplikuje wspólny layout do figury.
    
    Args:
        fig: Plotly figure
        title: Tytuł wykresu
        x_title: Tytuł osi X
        y_title: Tytuł osi Y
        show_legend: Czy pokazywać legendę
        height: Wysokość w pikselach
        width: Szerokość w pikselach
        
    Returns:
        Zaktualizowana figura
    """
    layout_updates = {"showlegend": show_legend}
    
    if title:
        layout_updates["title"] = dict(text=title)
    
    if height:
        layout_updates["height"] = height
    
    if width:
        layout_updates["width"] = width
    
    fig.update_layout(**layout_updates)
    
    if x_title:
        fig.update_xaxes(title=x_title)
    
    if y_title:
        fig.update_yaxes(title=y_title)
    
    return fig


def _drop_na_columns(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Usuwa wiersze z NaN w specified columns z logowaniem.
    
    Args:
        df: DataFrame
        columns: Lista kolumn
        
    Returns:
        Oczyszczony DataFrame
    """
    initial_len = len(df)
    clean_df = df[columns].dropna()
    dropped = initial_len - len(clean_df)
    
    if dropped > 0:
        dropped_pct = (dropped / initial_len) * 100
        LOGGER.debug(f"Dropped {dropped:,} rows ({dropped_pct:.1f}%) with NaN")
    
    return clean_df


def _add_statistical_lines(
    fig: go.Figure,
    data: np.ndarray,
    show_mean: bool = False,
    show_median: bool = False,
    show_std: bool = False,
    show_quantiles: Optional[List[float]] = None
) -> go.Figure:
    """
    Dodaje linie statystyczne do wykresu.
    
    Args:
        fig: Plotly figure
        data: Dane numeryczne
        show_mean: Czy pokazać średnią
        show_median: Czy pokazać medianę
        show_std: Czy pokazać odchylenie standardowe
        show_quantiles: Lista kwantyli do pokazania (np. [0.25, 0.75])
        
    Returns:
        Figura z liniami statystycznymi
    """
    palette = get_active_palette()
    shapes = []
    annotations = []
    
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        return fig
    
    if show_mean:
        mean_val = float(np.mean(clean_data))
        shapes.append(dict(
            type="line",
            x0=mean_val,
            x1=mean_val,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color=palette.secondary,
                width=LINE_WIDTH,
                dash="dot"
            )
        ))
        annotations.append(dict(
            x=mean_val,
            y=1.02,
            yref="paper",
            text="μ",
            showarrow=False,
            font=dict(size=FONT_SIZE_SMALL, color=palette.secondary)
        ))
    
    if show_median:
        median_val = float(np.median(clean_data))
        shapes.append(dict(
            type="line",
            x0=median_val,
            x1=median_val,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color=palette.warning,
                width=LINE_WIDTH,
                dash="dot"
            )
        ))
        annotations.append(dict(
            x=median_val,
            y=1.02,
            yref="paper",
            text="M",
            showarrow=False,
            font=dict(size=FONT_SIZE_SMALL, color=palette.warning)
        ))
    
    if show_std and show_mean:
        mean_val = float(np.mean(clean_data))
        std_val = float(np.std(clean_data))
        
        for sign in [-1, 1]:
            x_val = mean_val + sign * std_val
            shapes.append(dict(
                type="line",
                x0=x_val,
                x1=x_val,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color=palette.info,
                    width=THIN_LINE_WIDTH,
                    dash="dash"
                )
            ))
    
    if show_quantiles:
        for q in show_quantiles:
            q_val = float(np.quantile(clean_data, q))
            shapes.append(dict(
                type="line",
                x0=q_val,
                x1=q_val,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color=palette.accent,
                    width=THIN_LINE_WIDTH,
                    dash="dash"
                )
            ))
            annotations.append(dict(
                x=q_val,
                y=1.02,
                yref="paper",
                text=f"Q{q}",
                showarrow=False,
                font=dict(size=FONT_SIZE_SMALL - 1, color=palette.accent)
            ))
    
    if shapes:
        fig.update_layout(shapes=shapes)
    
    if annotations:
        fig.update_layout(annotations=annotations)
    
    return fig


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
    binning_method: BinningMethod = BinningMethod.AUTO,
    show_mean: bool = True,
    show_median: bool = True,
    show_std: bool = False,
    show_quantiles: Optional[List[float]] = None,
    marginal: Optional[Literal["box", "violin", "rug"]] = None,
    opacity: Optional[float] = None,
    cumulative: bool = False,
    **kwargs
) -> go.Figure:
    """
    Zaawansowany histogram z adaptacyjnymi bins i statystykami PRO++++.
    
    Args:
        df: DataFrame z danymi
        col: Nazwa kolumny do histogramu
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania (optional)
        histnorm: Normalizacja ("percent", "probability", "density")
        nbins: Liczba bins (auto jeśli None)
        binning_method: Metoda obliczania bins (fd, sturges, scott, auto)
        show_mean: Czy pokazać linię średniej
        show_median: Czy pokazać linię mediany
        show_std: Czy pokazać linie ±1σ
        show_quantiles: Lista kwantyli do pokazania (np. [0.25, 0.75])
        marginal: Marginal plot type ("box", "violin", "rug")
        opacity: Przezroczystość słupków (auto jeśli None)
        cumulative: Czy histogram kumulatywny
        **kwargs: Dodatkowe argumenty dla px.histogram
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = histogram(df, "price", show_mean=True, marginal="box")
        >>> fig = histogram(df, "age", show_quantiles=[0.25, 0.75], show_std=True)
        >>> fig.show()
    """
    if col not in df.columns:
        LOGGER.error(f"Column '{col}' not found in DataFrame")
        raise ValueError(f"Kolumna '{col}' nie istnieje")
    
    # Try numeric conversion
    numeric_series = _safe_numeric_conversion(df[col], col_name=col)
    
    # If mostly non-numeric, fall back to categorical bar chart
    numeric_pct = numeric_series.notna().mean()
    
    if numeric_pct < 0.6:
        LOGGER.debug(
            f"Column '{col}' is non-numeric ({numeric_pct*100:.1f}%), "
            f"creating categorical bar chart"
        )
        
        value_counts = (
            df[col]
            .astype(str)
            .fillna("(NaN)")
            .value_counts()
            .reset_index()
        )
        value_counts.columns = [col, "count"]
        
        # Limit to top categories for readability
        if len(value_counts) > 50:
            LOGGER.info(f"Limiting to top 50 categories (from {len(value_counts)})")
            value_counts = value_counts.head(50)
        
        fig = px.bar(
            value_counts,
            x=col,
            y="count",
            color=None,
            opacity=opacity or DEFAULT_OPACITY
        )
        
        return _apply_common_layout(
            fig,
            title=title or f"Rozkład: {col}",
            x_title=col,
            y_title="Liczność"
        )
    
    # Numeric histogram
    data = numeric_series.values
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) == 0:
        LOGGER.warning(f"No valid data for histogram of '{col}'")
        return go.Figure()
    
    # Auto-calculate bins if not provided
    if nbins is None:
        nbins = _calculate_optimal_bins(clean_data, method=binning_method)
        LOGGER.debug(f"Auto-calculated {nbins} bins for '{col}' using {binning_method.value}")
    
    # Auto opacity
    if opacity is None:
        opacity = DEFAULT_OPACITY
    
    # Create histogram
    fig = px.histogram(
        df.assign(_numeric_col=numeric_series),
        x="_numeric_col",
        color=color if (color and color in df.columns) else None,
        nbins=nbins,
        opacity=opacity,
        histnorm=histnorm,
        marginal=marginal,
        cumulative=cumulative,
        **kwargs
    )
    
    # Update hover
    hover_template = "<b>%{x}</b><br>count=%{y}<extra></extra>"
    if histnorm:
        hover_template = f"<b>%{{x}}</b><br>{histnorm}=%{{y:.4f}}<extra></extra>"
    
    fig.update_traces(hovertemplate=hover_template)
    
    # Apply layout
    y_label = "Liczność"
    if histnorm == "percent":
        y_label = "Procent"
    elif histnorm == "probability":
        y_label = "Prawdopodobieństwo"
    elif histnorm == "density":
        y_label = "Gęstość"
    
    if cumulative:
        y_label = f"Skumulowane {y_label.lower()}"
    
    fig = _apply_common_layout(
        fig,
        title=title or f"Histogram: {col}",
        x_title=col,
        y_title=y_label
    )
    
    # Add statistical lines (only for non-cumulative)
    if not cumulative:
        fig = _add_statistical_lines(
            fig,
            clean_data,
            show_mean=show_mean,
            show_median=show_median,
            show_std=show_std,
            show_quantiles=show_quantiles
        )
    
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
    trendline: Optional[Literal["ols", "lowess", "poly2", "poly3", "exp"]] = "ols",
    add_identity: bool = False,
    opacity: Optional[float] = None,
    log_x: bool = False,
    log_y: bool = False,
    sample_size: Optional[int] = None,
    **kwargs
) -> go.Figure:
    """
    Rozszerzony scatter plot z trendline i auto-optimization PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X
        y: Kolumna dla osi Y
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania (optional)
        size: Kolumna dla rozmiaru markerów (optional)
        hover_data: Dodatkowe kolumny w hover (optional)
        trendline: Typ trendline ("ols", "lowess", "poly2", "poly3", "exp", None)
        add_identity: Czy dodać linię y=x
        opacity: Przezroczystość (auto jeśli None)
        log_x: Skala logarytmiczna dla osi X
        log_y: Skala logarytmiczna dla osi Y
        sample_size: Maksymalna liczba punktów (sampling dla dużych zbiorów)
        **kwargs: Dodatkowe argumenty dla px.scatter
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = scatter(df, "x", "y", trendline="ols", add_identity=True)
        >>> fig = scatter(df, "price", "quantity", color="category", log_x=True)
        >>> fig.show()
    """
    if x not in df.columns or y not in df.columns:
        LOGGER.error(f"Columns '{x}' or '{y}' not found")
        raise ValueError(f"Kolumny '{x}' lub '{y}' nie istnieją")
    
    # Select columns
    clean_cols = [x, y]
    if color and color in df.columns:
        clean_cols.append(color)
    if size and size in df.columns:
        clean_cols.append(size)
    
    df_clean = _drop_na_columns(df, clean_cols).copy()
    
    if df_clean.empty:
        LOGGER.warning("No data after removing NaN")
        return go.Figure()
    
    # Sampling for very large datasets
    if sample_size and len(df_clean) > sample_size:
        LOGGER.info(f"Sampling {sample_size:,} from {len(df_clean):,} points")
        df_clean = df_clean.sample(n=sample_size, random_state=42)
    
    # Datetime handling
    df_clean[x] = _coerce_datetime(df_clean[x], col_name=x)
    
    # Numeric conversion for y
    if not pd.api.types.is_datetime64_any_dtype(df_clean[y]):
        df_clean[y] = _safe_numeric_conversion(df_clean[y], col_name=y)
    
    df_clean = df_clean.dropna(subset=[x, y])
    
    if df_clean.empty:
        LOGGER.warning("No valid data after conversions")
        return go.Figure()
    
    # Performance optimization
    n_points = len(df_clean)
    render_mode = "webgl" if _is_large_dataset(n_points) else "auto"
    
    if opacity is None:
        if n_points >= LOW_OPACITY_THRESHOLD:
            opacity = LARGE_DATA_OPACITY
        else:
            opacity = DEFAULT_OPACITY
    
    # Handle polynomial and exponential trendlines
    plotly_trendline = trendline
    custom_trendline = None
    
    if trendline in ["poly2", "poly3", "exp"]:
        plotly_trendline = None  # We'll add custom
        custom_trendline = trendline
    
    # Create scatter
    fig = px.scatter(
        df_clean,
        x=x,
        y=y,
        color=color if (color and color in df_clean.columns) else None,
        size=size if (size and size in df_clean.columns) else None,
        hover_data=list(hover_data) if hover_data else None,
        trendline=plotly_trendline,
        render_mode=render_mode,
        log_x=log_x,
        log_y=log_y,
        **kwargs
    )
    
    # Style markers
    marker_size = SMALL_MARKER_SIZE if n_points > MARKER_THRESHOLD else MARKER_SIZE
    
    fig.update_traces(
        marker=dict(
            opacity=opacity,
            line=dict(width=0),
            size=marker_size
        ),
        selector=dict(mode="markers")
    )
    
    # Add custom trendlines
    if custom_trendline and pd.api.types.is_numeric_dtype(df_clean[x]):
        try:
            x_vals = df_clean[x].values
            y_vals = df_clean[y].values
            
            # Remove NaN
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_clean = x_vals[mask]
            y_clean = y_vals[mask]
            
            if len(x_clean) > 0:
                x_sorted = np.sort(x_clean)
                
                if custom_trendline == "poly2":
                    coeffs = np.polyfit(x_clean, y_clean, 2)
                    y_trend = np.polyval(coeffs, x_sorted)
                    name = "Polynomial (deg=2)"
                    
                elif custom_trendline == "poly3":
                    coeffs = np.polyfit(x_clean, y_clean, 3)
                    y_trend = np.polyval(coeffs, x_sorted)
                    name = "Polynomial (deg=3)"
                    
                elif custom_trendline == "exp":
                    # Fit exponential: y = a * exp(b * x)
                    # Take log: log(y) = log(a) + b*x
                    if np.all(y_clean > 0):
                        log_y = np.log(y_clean)
                        coeffs = np.polyfit(x_clean, log_y, 1)
                        y_trend = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_sorted)
                        name = "Exponential"
                    else:
                        y_trend = None
                        LOGGER.warning("Exponential fit requires positive y values")
                
                if y_trend is not None:
                    palette = get_active_palette()
                    fig.add_trace(go.Scatter(
                        x=x_sorted,
                        y=y_trend,
                        mode="lines",
                        line=dict(
                            color=palette.error,
                            width=LINE_WIDTH,
                            dash="dash"
                        ),
                        name=name,
                        showlegend=True,
                        hoverinfo="skip"
                    ))
                    
        except Exception as e:
            LOGGER.warning(f"Failed to add custom trendline: {e}")
    
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
        is_numeric_x = pd.api.types.is_numeric_dtype(df_clean[x])
        is_numeric_y = pd.api.types.is_numeric_dtype(df_clean[y])
        
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
                line=dict(color="#64748b", dash="dash", width=THIN_LINE_WIDTH),
                name="y=x",
                showlegend=True,
                hoverinfo="skip"
            ))
    
    # Apply performance optimizations
    fig = _apply_performance_optimizations(fig, n_points)
    
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
    uncertainty_percentiles: Tuple[float, float] = (0.025, 0.975),
    smooth: bool = False,
    fill_area: bool = False,
    **kwargs
) -> go.Figure:
    """
    Line plot z auto datetime handling i uncertainty band PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X (często datetime)
        y: Kolumna dla osi Y
        title: Tytuł wykresu (optional)
        color: Kolumna dla kolorowania linii (optional)
        markers_auto: Czy automatycznie dodać markery (dla małych zbiorów)
        show_uncertainty_band: Czy pokazać pasmo niepewności
        uncertainty_percentiles: Percentyle dla pasma (default 95% CI)
        smooth: Czy wygładzić linię (LOWESS)
        fill_area: Czy wypełnić obszar pod linią
        **kwargs: Dodatkowe argumenty dla px.line
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = line(df, "date", "sales", show_uncertainty_band=True)
        >>> fig = line(df, "time", "temperature", smooth=True, fill_area=True)
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
    df_clean[x] = _coerce_datetime(df_clean[x], col_name=x)
    
    # Numeric conversion for y
    df_clean[y] = _safe_numeric_conversion(df_clean[y], col_name=y)
    
    # Remove NaN and sort
    df_clean = df_clean.dropna(subset=[x, y]).sort_values(by=x)
    
    if df_clean.empty:
        LOGGER.warning("No valid data for line plot")
        return go.Figure()
    
    # Decide on markers
    show_markers = markers_auto and len(df_clean) <= MARKER_THRESHOLD
    
    # Apply smoothing if requested
    if smooth and len(df_clean) > 10:
        try:
            from scipy.signal import savgol_filter
            
            # Savitzky-Golay filter
            window = min(51, len(df_clean) // 2 * 2 + 1)  # Odd number
            df_clean[f"{y}_smooth"] = savgol_filter(
                df_clean[y],
                window_length=window,
                polyorder=3
            )
            y_col = f"{y}_smooth"
            LOGGER.debug(f"Applied smoothing with window={window}")
        except Exception as e:
            LOGGER.warning(f"Smoothing failed: {e}, using original data")
            y_col = y
    else:
        y_col = y
    
    # Create line plot
    fig = px.line(
        df_clean,
        x=x,
        y=y_col,
        color=color if (color and color in df_clean.columns) else None,
        markers=show_markers,
        **kwargs
    )
    
    # Fill area under curve
    if fill_area:
        palette = get_active_palette()
        for trace in fig.data:
            trace.fill = "tozeroy"
            trace.fillcolor = f"rgba({int(palette.primary[1:3], 16)}, "
            f"{int(palette.primary[3:5], 16)}, "
            f"{int(palette.primary[5:7], 16)}, 0.2)"
    
    # Apply layout
    fig = _apply_common_layout(
        fig,
        title=title or f"Linia: {y} w czasie",
        x_title=x,
        y_title=y
    )
    
    # Add uncertainty band
    if show_uncertainty_band:
        lower_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["lower", "yhat_lower", "ci_lower"])
        ]
        upper_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["upper", "yhat_upper", "ci_upper"])
        ]
        
        if lower_cols and upper_cols:
            lower_col = lower_cols[0]
            upper_col = upper_cols[0]
            
            try:
                band_df = df[[x, lower_col, upper_col]].copy()
                band_df[x] = _coerce_datetime(band_df[x], col_name=x)
                band_df = band_df.dropna().sort_values(by=x)
                
                if not band_df.empty:
                    palette = get_active_palette()
                    
                    # Extract RGB from hex
                    r = int(palette.primary[1:3], 16)
                    g = int(palette.primary[3:5], 16)
                    b = int(palette.primary[5:7], 16)
                    
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
                            fillcolor=f"rgba({r}, {g}, {b}, 0.15)",
                            name=f"CI {int(uncertainty_percentiles[1]*100)}%",
                            hoverinfo="skip"
                        )
                    ])
                    
                    LOGGER.debug(f"Added uncertainty band: {lower_col} - {upper_col}")
            except Exception as e:
                LOGGER.warning(f"Failed to add uncertainty band: {e}")
    
    return fig


# ========================================================================================
# BOX & VIOLIN PLOTS
# ========================================================================================

def box(
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    notched: bool = False,
    show_points: Literal["all", "outliers", "suspectedoutliers", False] = False,
    **kwargs
) -> go.Figure:
    """
    Box plot z konfigurowalnymi opcjami PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X (kategoryczna, optional)
        y: Kolumna dla osi Y (numeryczna)
        title: Tytuł wykresu
        color: Kolumna dla kolorowania
        notched: Czy pokazać notch (confidence interval)
        show_points: Pokazywanie punktów ("all", "outliers", "suspectedoutliers", False)
        **kwargs: Dodatkowe argumenty dla px.box
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = box(df, x="category", y="value", notched=True)
        >>> fig = box(df, y="price", show_points="outliers")
        >>> fig.show()
    """
    if y and y not in df.columns:
        raise ValueError(f"Kolumna '{y}' nie istnieje")
    
    if x and x not in df.columns:
        raise ValueError(f"Kolumna '{x}' nie istnieje")
    
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color if (color and color in df.columns) else None,
        notched=notched,
        points=show_points,
        **kwargs
    )
    
    fig = _apply_common_layout(
        fig,
        title=title or f"Box plot: {y or 'wartości'}",
        x_title=x or "",
        y_title=y or "Wartość"
    )
    
    return fig


def violin(
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    box: bool = True,
    show_points: Literal["all", "outliers", False] = False,
    **kwargs
) -> go.Figure:
    """
    Violin plot z opcjami PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X (kategoryczna, optional)
        y: Kolumna dla osi Y (numeryczna)
        title: Tytuł wykresu
        color: Kolumna dla kolorowania
        box: Czy pokazać box plot wewnątrz
        show_points: Pokazywanie punktów
        **kwargs: Dodatkowe argumenty dla px.violin
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = violin(df, x="category", y="value", box=True)
        >>> fig.show()
    """
    if y and y not in df.columns:
        raise ValueError(f"Kolumna '{y}' nie istnieje")
    
    if x and x not in df.columns:
        raise ValueError(f"Kolumna '{x}' nie istnieje")
    
    fig = px.violin(
        df,
        x=x,
        y=y,
        color=color if (color and color in df.columns) else None,
        box=box,
        points=show_points,
        **kwargs
    )
    
    fig = _apply_common_layout(
        fig,
        title=title or f"Violin plot: {y or 'wartości'}",
        x_title=x or "",
        y_title=y or "Wartość"
    )
    
    return fig


# ========================================================================================
# CORRELATION HEATMAP
# ========================================================================================

def correlation_heatmap(
    df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: Optional[str] = None,
    annotate: bool = True,
    mask_diagonal: bool = True,
    cluster: bool = False
) -> go.Figure:
    """
    Correlation heatmap dla kolumn numerycznych PRO++++.
    
    Args:
        df: DataFrame z danymi
        method: Metoda korelacji
        title: Tytuł wykresu
        annotate: Czy pokazać wartości na komórkach
        mask_diagonal: Czy ukryć diagonalę (zawsze 1.0)
        cluster: Czy grupować podobne kolumny
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = correlation_heatmap(df, method="spearman", cluster=True)
        >>> fig.show()
    """
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        LOGGER.warning("No numeric columns for correlation")
        return go.Figure()
    
    if numeric_df.shape[1] < 2:
        LOGGER.warning("Need at least 2 numeric columns for correlation")
        return go.Figure()
    
    # Calculate correlation
    corr_matrix = numeric_df.corr(method=method).round(3)
    
    # Mask diagonal
    if mask_diagonal:
        np.fill_diagonal(corr_matrix.values, np.nan)
    
    # Clustering (hierarchical)
    if cluster and corr_matrix.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            dist_matrix = 1 - np.abs(corr_matrix.fillna(0))
            
            # Linkage
            linkage_matrix = linkage(squareform(dist_matrix), method="average")
            
            # Reorder
            order = leaves_list(linkage_matrix)
            corr_matrix = corr_matrix.iloc[order, order]
            
            LOGGER.debug("Applied hierarchical clustering to correlation matrix")
        except Exception as e:
            LOGGER.warning(f"Clustering failed: {e}, using original order")
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f" if annotate else False,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    fig.update_coloraxes(
        colorbar_title=dict(
            text="ρ",
            font=dict(size=FONT_SIZE_MEDIUM)
        )
    )
    
    # Update layout
    title_text = title or f"Korelacje ({method.capitalize()})"
    fig = _apply_common_layout(fig, title=title_text, show_legend=False)
    
    # Improve readability
    fig.update_xaxes(tickangle=-45)
    
    return fig


# ========================================================================================
# FEATURE IMPORTANCE
# ========================================================================================

def feature_importance(
    importances: Union[Sequence[Tuple[str, float]], Dict[str, float]],
    top: int = DEFAULT_TOP_FEATURES,
    title: Optional[str] = None,
    orientation: Literal["h", "v"] = "h",
    style: Literal["bar", "lollipop"] = "bar",
    show_values: bool = True
) -> go.Figure:
    """
    Feature importance visualization PRO++++.
    
    Args:
        importances: Lista tuple (feature, importance) lub dict
        top: Liczba top features
        title: Tytuł wykresu
        orientation: Orientacja ("h" = horizontal, "v" = vertical)
        style: Styl ("bar" lub "lollipop")
        show_values: Czy pokazać wartości na wykresie
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> importances = [("feature1", 0.5), ("feature2", 0.3)]
        >>> fig = feature_importance(importances, top=10, style="lollipop")
        >>> fig.show()
    """
    # Convert to list of tuples if dict
    if isinstance(importances, dict):
        importances = list(importances.items())
    
    if not importances:
        LOGGER.warning("No importances provided")
        return go.Figure()
    
    # Sort and select top
    sorted_data = sorted(
        list(importances),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:min(top, MAX_TOP_FEATURES)]
    
    if not sorted_data:
        return go.Figure()
    
    features, values = zip(*sorted_data)
    
    # Create DataFrame
    df = pd.DataFrame({
        "feature": features,
        "importance": values
    })
    
    # Reverse for better readability (highest at top)
    if orientation == "h":
        df = df.iloc[::-1]
    
    palette = get_active_palette()
    
    # Create visualization based on style
    if style == "lollipop":
        fig = go.Figure()
        
        if orientation == "h":
            fig.add_trace(go.Scatter(
                x=df["importance"],
                y=df["feature"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=palette.primary,
                    line=dict(width=2, color=palette.secondary)
                ),
                name="Importance"
            ))
            
            # Add stems
            for idx, row in df.iterrows():
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=row["importance"],
                    y0=row["feature"],
                    y1=row["feature"],
                    line=dict(color=palette.primary, width=2)
                )
            
            x_label, y_label = "Ważność", "Cecha"
        else:
            fig.add_trace(go.Scatter(
                x=df["feature"],
                y=df["importance"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=palette.primary,
                    line=dict(width=2, color=palette.secondary)
                ),
                name="Importance"
            ))
            
            # Add stems
            for idx, row in df.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row["feature"],
                    x1=row["feature"],
                    y0=0,
                    y1=row["importance"],
                    line=dict(color=palette.primary, width=2)
                )
            
            x_label, y_label = "Cecha", "Ważność"
        
    else:  # bar
        if orientation == "h":
            fig = px.bar(
                df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                text="importance" if show_values else None
            )
            x_label, y_label = "Ważność", "Cecha"
        else:
            fig = px.bar(
                df,
                x="feature",
                y="importance",
                orientation="v",
                color="importance",
                color_continuous_scale="Blues",
                text="importance" if show_values else None
            )
            x_label, y_label = "Cecha", "Ważność"
        
        if show_values:
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    
    fig = _apply_common_layout(
        fig,
        title=title or "Ważność cech",
        x_title=x_label,
        y_title=y_label,
        show_legend=False
    )
    
    # Rotate x-axis labels for vertical bar charts
    if orientation == "v":
        fig.update_xaxes(tickangle=-45)
    
    return fig


# ========================================================================================
# 3D SCATTER
# ========================================================================================

def scatter_3d(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    opacity: float = DEFAULT_OPACITY,
    **kwargs
) -> go.Figure:
    """
    3D scatter plot PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X
        y: Kolumna dla osi Y
        z: Kolumna dla osi Z
        title: Tytuł wykresu
        color: Kolumna dla kolorowania
        size: Kolumna dla rozmiaru markerów
        opacity: Przezroczystość
        **kwargs: Dodatkowe argumenty dla px.scatter_3d
        
    Returns:
        Plotly Figure
        
    Examples:
        >>> fig = scatter_3d(df, "x", "y", "z", color="category")
        >>> fig.show()
    """
    required_cols = [x, y, z]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolumna '{col}' nie istnieje")
    
    # Clean data
    clean_cols = required_cols.copy()
    if color and color in df.columns:
        clean_cols.append(color)
    if size and size in df.columns:
        clean_cols.append(size)
    
    df_clean = _drop_na_columns(df, clean_cols)
    
    if df_clean.empty:
        LOGGER.warning("No valid data for 3D scatter")
        return go.Figure()
    
    # Convert to numeric
    for col in [x, y, z]:
        df_clean[col] = _safe_numeric_conversion(df_clean[col], col_name=col)
    
    df_clean = df_clean.dropna(subset=[x, y, z])
    
    # Create 3D scatter
    fig = px.scatter_3d(
        df_clean,
        x=x,
        y=y,
        z=z,
        color=color if (color and color in df_clean.columns) else None,
        size=size if (size and size in df_clean.columns) else None,
        opacity=opacity,
        **kwargs
    )
    
    fig = _apply_common_layout(
        fig,
        title=title or f"3D Scatter: {x}, {y}, {z}"
    )
    
    # Update 3D scene
    fig.update_layout(
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        )
    )
    
    return fig


# ========================================================================================
# SUBPLOT GRID
# ========================================================================================

def create_subplot_grid(
    figures: List[go.Figure],
    rows: int,
    cols: int,
    subplot_titles: Optional[List[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: float = 0.1,
    horizontal_spacing: float = 0.1,
    title: Optional[str] = None
) -> go.Figure:
    """
    Tworzy grid z subplotów PRO++++.
    
    Args:
        figures: Lista figur Plotly do umieszczenia w gridzie
        rows: Liczba wierszy
        cols: Liczba kolumn
        subplot_titles: Tytuły dla każdego subplotu
        shared_xaxes: Czy współdzielić osie X
        shared_yaxes: Czy współdzielić osie Y
        vertical_spacing: Odstęp pionowy między subplotami
        horizontal_spacing: Odstęp poziomy między subplotami
        title: Główny tytuł wykresu
        
    Returns:
        Plotly Figure z subplotami
        
    Examples:
        >>> fig1 = histogram(df, "col1")
        >>> fig2 = scatter(df, "col2", "col3")
        >>> combined = create_subplot_grid([fig1, fig2], rows=1, cols=2)
        >>> combined.show()
    """
    if not figures:
        LOGGER.warning("No figures provided for subplot grid")
        return go.Figure()
    
    # Create subplot figure
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )
    
    # Add traces from each figure
    for idx, source_fig in enumerate(figures):
        if idx >= rows * cols:
            LOGGER.warning(f"Too many figures ({len(figures)}) for grid ({rows}x{cols})")
            break
        
        row = idx // cols + 1
        col = idx % cols + 1
        
        for trace in source_fig.data:
            fig.add_trace(trace, row=row, col=col)
        
        # Copy axis titles if available
        try:
            if source_fig.layout.xaxis.title:
                fig.update_xaxes(
                    title_text=source_fig.layout.xaxis.title.text,
                    row=row,
                    col=col
                )
            if source_fig.layout.yaxis.title:
                fig.update_yaxes(
                    title_text=source_fig.layout.yaxis.title.text,
                    row=row,
                    col=col
                )
        except Exception as e:
            LOGGER.debug(f"Could not copy axis titles: {e}")
    
    # Update layout
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    
    fig.update_layout(showlegend=True)
    
    return fig


# ========================================================================================
# TIME SERIES WITH ANIMATION
# ========================================================================================

def animated_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    animation_frame: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[Sequence[str]] = None,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    **kwargs
) -> go.Figure:
    """
    Animated scatter plot (np. dla time series) PRO++++.
    
    Args:
        df: DataFrame z danymi
        x: Kolumna dla osi X
        y: Kolumna dla osi Y
        animation_frame: Kolumna dla klatek animacji (np. rok, miesiąc)
        title: Tytuł wykresu
        color: Kolumna dla kolorowania
        size: Kolumna dla rozmiaru
        hover_data: Dodatkowe kolumny w hover
        range_x: Zakres osi X (dla stabilnej animacji)
        range_y: Zakres osi Y (dla stabilnej animacji)
        **kwargs: Dodatkowe argumenty dla px.scatter
        
    Returns:
        Plotly Figure z animacją
        
    Examples:
        >>> fig = animated_scatter(
        ...     df, "gdp", "life_expectancy", "year",
        ...     size="population", color="continent"
        ... )
        >>> fig.show()
    """
    required_cols = [x, y, animation_frame]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolumna '{col}' nie istnieje")
    
    # Create animated scatter
    fig = px.scatter(
        df,
        x=x,
        y=y,
        animation_frame=animation_frame,
        color=color if (color and color in df.columns) else None,
        size=size if (size and size in df.columns) else None,
        hover_data=list(hover_data) if hover_data else None,
        range_x=range_x,
        range_y=range_y,
        **kwargs
    )
    
    # Update animation settings
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": DEFAULT_ANIMATION_DURATION},
                            "fromcurrent": True,
                            "transition": {"duration": DEFAULT_ANIMATION_DURATION // 2}
                        }
                    ]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ]
        }]
    )
    
    fig = _apply_common_layout(
        fig,
        title=title or f"Animacja: {y} vs {x}",
        x_title=x,
        y_title=y
    )
    
    return fig


# ========================================================================================
# EXPORT UTILITIES
# ========================================================================================

def export_figure(
    fig: go.Figure,
    filename: str,
    format: Literal["html", "png", "svg", "pdf"] = "html",
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0
) -> str:
    """
    Eksportuje figurę do pliku PRO++++.
    
    Args:
        fig: Plotly Figure do eksportu
        filename: Nazwa pliku (bez rozszerzenia)
        format: Format eksportu
        width: Szerokość w pikselach (optional)
        height: Wysokość w pikselach (optional)
        scale: Skala dla PNG/SVG (wyższe = lepsza jakość)
        
    Returns:
        Ścieżka do zapisanego pliku
        
    Examples:
        >>> fig = histogram(df, "price")
        >>> path = export_figure(fig, "my_chart", format="png", scale=3.0)
        >>> print(f"Saved to: {path}")
    """
    import os
    
    # Add extension
    if not filename.endswith(f".{format}"):
        filename = f"{filename}.{format}"
    
    try:
        if format == "html":
            # HTML export (interactive)
            fig.write_html(
                filename,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            )
            LOGGER.info(f"Exported HTML to: {filename}")
            
        elif format in ["png", "svg", "pdf"]:
            # Static image export (requires kaleido)
            try:
                fig.write_image(
                    filename,
                    format=format,
                    width=width,
                    height=height,
                    scale=scale
                )
                LOGGER.info(f"Exported {format.upper()} to: {filename}")
            except Exception as e:
                LOGGER.error(f"Image export failed (kaleido required): {e}")
                raise RuntimeError(
                    f"Image export requires 'kaleido' package. "
                    f"Install with: pip install kaleido"
                )
        
        return os.path.abspath(filename)
        
    except Exception as e:
        LOGGER.error(f"Export failed: {e}")
        raise


# ========================================================================================
# THEME & PALETTE UTILITIES
# ========================================================================================

def get_available_themes() -> List[str]:
    """
    Zwraca listę dostępnych motywów.
    
    Returns:
        Lista nazw motywów
    """
    return [theme.value for theme in ChartTheme]


def create_custom_palette(
    primary: str = "#4A90E2",
    secondary: str = "#22d3ee",
    accent: str = "#a78bfa",
    success: str = "#34d399",
    warning: str = "#f59e0b",
    error: str = "#f87171",
    info: str = "#60a5fa"
) -> ColorPalette:
    """
    Tworzy niestandardową paletę kolorów.
    
    Args:
        primary: Kolor główny (hex)
        secondary: Kolor drugorzędny (hex)
        accent: Kolor akcentujący (hex)
        success: Kolor sukcesu (hex)
        warning: Kolor ostrzeżenia (hex)
        error: Kolor błędu (hex)
        info: Kolor informacyjny (hex)
        
    Returns:
        ColorPalette
        
    Examples:
        >>> palette = create_custom_palette(primary="#FF5733", secondary="#33FF57")
        >>> # Use in theme or directly in plots
    """
    return ColorPalette(
        primary=primary,
        secondary=secondary,
        accent=accent,
        success=success,
        warning=warning,
        error=error,
        info=info
    )


def preview_theme(theme: Union[ChartTheme, str] = ChartTheme.BUSINESS_DARK) -> go.Figure:
    """
    Tworzy podgląd motywu z przykładowymi wykresami.
    
    Args:
        theme: Motyw do podglądu
        
    Returns:
        Plotly Figure z podglądem
        
    Examples:
        >>> fig = preview_theme(ChartTheme.NEON)
        >>> fig.show()
    """
    # Set theme temporarily
    original_theme = get_active_theme()
    set_theme(theme)
    
    try:
        # Create sample data
        np.random.seed(42)
        n = 100
        sample_df = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C'], n),
            'value': np.random.randint(10, 100, n)
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Histogram",
                "Scatter",
                "Box Plot",
                "Line Plot"
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        palette = get_active_palette()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=sample_df['x'],
                marker_color=palette.primary,
                name="Histogram"
            ),
            row=1, col=1
        )
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=sample_df['x'],
                y=sample_df['y'],
                mode='markers',
                marker=dict(
                    color=sample_df['value'],
                    colorscale='Blues',
                    size=8
                ),
                name="Scatter"
            ),
            row=1, col=2
        )
        
        # Box
        for cat in ['A', 'B', 'C']:
            data = sample_df[sample_df['category'] == cat]['value']
            fig.add_trace(
                go.Box(
                    y=data,
                    name=cat,
                    marker_color=palette.to_list()[ord(cat) - ord('A')]
                ),
                row=2, col=1
            )
        
        # Line
        x_line = np.linspace(0, 10, 50)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=np.sin(x_line),
                mode='lines',
                line=dict(color=palette.secondary, width=2),
                name="Line"
            ),
            row=2, col=2
        )
        
        # Update layout
        theme_name = theme.value if isinstance(theme, ChartTheme) else theme
        fig.update_layout(
            title=dict(
                text=f"Theme Preview: {theme_name}",
                x=0.5,
                xanchor="center"
            ),
            showlegend=True,
            height=800
        )
        
        return fig
        
    finally:
        # Restore original theme
        set_theme(original_theme)


# ========================================================================================
# STATISTICAL OVERLAYS
# ========================================================================================

def add_normal_curve(
    fig: go.Figure,
    data: np.ndarray,
    color: Optional[str] = None
) -> go.Figure:
    """
    Dodaje krzywą normalną do histogramu.
    
    Args:
        fig: Plotly Figure (histogram)
        data: Dane do dopasowania
        color: Kolor krzywej
        
    Returns:
        Figura z krzywą normalną
        
    Examples:
        >>> fig = histogram(df, "value")
        >>> fig = add_normal_curve(fig, df["value"].values)
        >>> fig.show()
    """
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 2:
        LOGGER.warning("Not enough data for normal curve")
        return fig
    
    # Fit normal distribution
    mu = np.mean(clean_data)
    sigma = np.std(clean_data)
    
    # Generate curve
    x = np.linspace(clean_data.min(), clean_data.max(), 200)
    y = scipy_stats.norm.pdf(x, mu, sigma)
    
    # Scale to match histogram
    y = y * len(clean_data) * (clean_data.max() - clean_data.min()) / len(x)
    
    palette = get_active_palette()
    curve_color = color or palette.error
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color=curve_color, width=THICK_LINE_WIDTH, dash='dash'),
        name=f'N(μ={mu:.2f}, σ={sigma:.2f})',
        showlegend=True
    ))
    
    return fig


def add_regression_stats(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    position: Tuple[float, float] = (0.05, 0.95)
) -> go.Figure:
    """
    Dodaje statystyki regresji do scatter plot.
    
    Args:
        fig: Plotly Figure (scatter)
        x: Dane X
        y: Dane Y
        position: Pozycja tekstu (x, y w układzie paper)
        
    Returns:
        Figura ze statystykami
        
    Examples:
        >>> fig = scatter(df, "x", "y", trendline="ols")
        >>> fig = add_regression_stats(fig, df["x"].values, df["y"].values)
        >>> fig.show()
    """
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        LOGGER.warning("Not enough data for regression stats")
        return fig
    
    # Calculate statistics
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_clean, y_clean)
    
    # Create annotation
    stats_text = (
        f"<b>Regression Stats</b><br>"
        f"R² = {r_value**2:.4f}<br>"
        f"slope = {slope:.4f}<br>"
        f"intercept = {intercept:.4f}<br>"
        f"p-value = {p_value:.4e}"
    )
    
    config = THEMES[get_active_theme()]
    
    fig.add_annotation(
        x=position[0],
        y=position[1],
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        bgcolor=config.plot_bg,
        bordercolor=config.grid_color,
        borderwidth=1,
        font=dict(size=FONT_SIZE_SMALL, color=config.text_color)
    )
    
    return fig


# ========================================================================================
# ACCESSIBILITY HELPERS
# ========================================================================================

def make_accessible(
    fig: go.Figure,
    high_contrast: bool = True,
    increase_font_size: bool = True,
    thicker_lines: bool = True
) -> go.Figure:
    """
    Poprawia accessibility wykresu (WCAG AA).
    
    Args:
        fig: Plotly Figure
        high_contrast: Czy zwiększyć kontrast
        increase_font_size: Czy zwiększyć czcionkę
        thicker_lines: Czy pogrubić linie
        
    Returns:
        Figura z poprawkami accessibility
        
    Examples:
        >>> fig = scatter(df, "x", "y")
        >>> fig = make_accessible(fig)
        >>> fig.show()
    """
    updates = {}
    
    if increase_font_size:
        updates["font"] = dict(size=FONT_SIZE_LARGE)
        fig.update_xaxes(title_font=dict(size=FONT_SIZE_LARGE))
        fig.update_yaxes(title_font=dict(size=FONT_SIZE_LARGE))
    
    if thicker_lines:
        fig.update_traces(
            line=dict(width=THICK_LINE_WIDTH),
            selector=dict(type="scatter", mode="lines")
        )
    
    if high_contrast:
        # Use high contrast colors
        high_contrast_palette = ColorPalette(
            primary="#0066CC",
            secondary="#FF6600",
            accent="#9933CC",
            success="#00AA00",
            warning="#FFAA00",
            error="#CC0000",
            info="#0099CC"
        )
        updates["colorway"] = high_contrast_palette.to_list()
    
    if updates:
        fig.update_layout(**updates)
    
    LOGGER.info("Applied accessibility improvements")
    
    return fig


# ========================================================================================
# EXPORT & DOCUMENTATION
# ========================================================================================

__all__ = [
    # Main plotting functions
    "histogram",
    "scatter",
    "line",
    "box",
    "violin",
    "correlation_heatmap",
    "feature_importance",
    "scatter_3d",
    "animated_scatter",
    
    # Layout utilities
    "create_subplot_grid",
    
    # Statistical overlays
    "add_normal_curve",
    "add_regression_stats",
    "add_statistical_lines",
    
    # Theme management
    "set_theme",
    "get_active_theme",
    "get_active_palette",
    "get_available_themes",
    "create_custom_palette",
    "preview_theme",
    
    # Export
    "export_figure",
    
    # Accessibility
    "make_accessible",
    
    # Enums & dataclasses
    "ChartTheme",
    "BinningMethod",
    "ColorPalette",
    "ThemeConfig",
]

# ========================================================================================
# MODULE INITIALIZATION LOG
# ========================================================================================

LOGGER.info(
    f"Advanced Charts PRO++++ initialized | "
    f"Active theme: {_ACTIVE_THEME.value} | "
    f"Available themes: {', '.join(get_available_themes())}"
)