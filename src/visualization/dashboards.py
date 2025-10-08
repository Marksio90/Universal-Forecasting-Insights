"""
Dashboard Components - Komponenty dashboardów dla complex analytics.

Funkcjonalności:
- KPI boards (key metrics overview)
- EDA overview (distributions + correlations)
- Forecast boards (time series predictions)
- Model performance boards (classification/regression metrics)
- Anomaly detection boards (PCA visualization)
- Grid composer (multiple charts in grid)
- Professional styling
- Interactive components
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Dict, List, Tuple, Literal, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import chart functions
from .charts import (
    histogram,
    scatter,
    line,
    correlation_heatmap,
)

# Optional validator import
try:
    from ..utils.validators import basic_quality_checks
    HAS_VALIDATORS = True
except ImportError:
    basic_quality_checks = None
    HAS_VALIDATORS = False

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Template name (from charts.py)
TEMPLATE_NAME = "ip_business_dark"

# KPI board dimensions
KPI_HEIGHT = 160
KPI_MARGIN = dict(l=20, r=20, t=60, b=10)

# Standard board dimensions
BOARD_HEIGHT = 420
BOARD_MARGIN = dict(l=20, r=20, t=60, b=20)

# EDA overview
EDA_HEIGHT_SINGLE = 360
EDA_HEIGHT_DOUBLE = 600
EDA_SPACING = 0.12
EDA_TOP_FEATURES = 4

# Grid composer
GRID_ROW_HEIGHT = 360
GRID_H_SPACING = 0.08
GRID_V_SPACING = 0.12

# Metric orders
CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc", "precision", "recall"]
REGRESSION_METRICS = ["rmse", "mae", "r2", "mape", "mse"]

# Metric formatting
METRIC_FORMATS = {
    "accuracy": ".3f",
    "balanced_accuracy": ".3f",
    "f1_weighted": ".3f",
    "roc_auc": ".3f",
    "precision": ".3f",
    "recall": ".3f",
    "rmse": ".3f",
    "mae": ".3f",
    "r2": ".3f",
    "mape": ".2f",
    "mse": ".3f"
}

METRIC_SUFFIXES = {
    "mape": "%"
}

# Colors
COLOR_NORMAL = "#4A90E2"
COLOR_ANOMALY = "#f87171"

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "dashboards", level: int = logging.INFO) -> logging.Logger:
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
# UTILITY FUNCTIONS
# ========================================================================================

def _create_empty_figure(message: str, height: int = BOARD_HEIGHT) -> go.Figure:
    """
    Tworzy pustą figurę z komunikatem.
    
    Args:
        message: Komunikat do wyświetlenia
        height: Wysokość figury
        
    Returns:
        Empty Plotly Figure
    """
    fig = go.Figure()
    
    fig.update_layout(
        template=TEMPLATE_NAME,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="#9ca3af")
        )],
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


def _calculate_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Oblicza podstawowe statystyki dla DataFrame.
    
    Args:
        df: DataFrame
        
    Returns:
        Dict ze statystykami
    """
    rows = len(df)
    cols = df.shape[1]
    
    # Missing
    total_cells = df.size
    missing_all = df.isna().sum().sum()
    missing_pct = float(missing_all / max(1, total_cells))
    
    # Duplicates
    dupes = int(df.duplicated().sum())
    dupes_pct = float(dupes / rows) if rows > 0 else 0.0
    
    return {
        "rows": rows,
        "cols": cols,
        "missing_pct": missing_pct,
        "dupes": dupes,
        "dupes_pct": dupes_pct
    }


# ========================================================================================
# KPI BOARD
# ========================================================================================

def kpi_board(
    df: Optional[pd.DataFrame] = None,
    *,
    stats: Optional[Dict[str, Any]] = None,
    title: str = "KPI danych"
) -> go.Figure:
    """
    Tworzy board z kluczowymi metrykami danych.
    
    Args:
        df: DataFrame (optional jeśli podano stats)
        stats: Pre-calculated statistics (optional)
        title: Tytuł boardu
        
    Returns:
        Plotly Figure z KPI indicators
        
    Example:
        >>> fig = kpi_board(df, title="Dataset Overview")
        >>> fig.show()
    """
    # Get or calculate stats
    if stats is None:
        if df is None:
            LOGGER.error("Must provide either df or stats")
            raise ValueError("Podaj `df` lub `stats`")
        
        if HAS_VALIDATORS and basic_quality_checks:
            LOGGER.debug("Using basic_quality_checks for stats")
            stats = basic_quality_checks(df)
        else:
            LOGGER.debug("Calculating basic stats manually")
            stats = _calculate_basic_stats(df)
    
    # Extract values
    rows = int(stats.get("rows", 0))
    cols = int(stats.get("cols", 0))
    
    # Missing percentage (handle both 0-1 and 0-100 scales)
    missing_pct = float(stats.get("missing_pct", 0.0))
    if missing_pct <= 1.0:
        missing_pct *= 100.0
    
    # Duplicates
    dupes = int(stats.get("dupes", 0))
    dupes_pct = float(stats.get("dupes_pct", 0.0))
    if dupes_pct <= 1.0:
        dupes_pct *= 100.0
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=("Wiersze", "Kolumny", "Braki [%]", "Duplikaty [%]")
    )
    
    # Add indicators
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=rows,
            number={"valueformat": ",.0f"}
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=cols,
            number={"valueformat": ",.0f"}
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=missing_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 20
                }
            }
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=dupes_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 1
                }
            }
        ),
        row=1, col=4
    )
    
    # Update layout
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=KPI_MARGIN,
        height=KPI_HEIGHT,
        showlegend=False
    )
    
    LOGGER.debug(f"Created KPI board: rows={rows}, cols={cols}")
    
    return fig


# ========================================================================================
# EDA OVERVIEW
# ========================================================================================

def eda_overview(
    df: pd.DataFrame,
    *,
    top_numeric: int = EDA_TOP_FEATURES,
    title: str = "EDA – rozkłady i korelacje"
) -> go.Figure:
    """
    Tworzy overview EDA z histogramami i heatmapą korelacji.
    
    Args:
        df: DataFrame do analizy
        top_numeric: Liczba top features (by variance)
        title: Tytuł boardu
        
    Returns:
        Plotly Figure z grid
        
    Example:
        >>> fig = eda_overview(df, top_numeric=6)
        >>> fig.show()
    """
    if df is None or df.empty:
        LOGGER.warning("Empty DataFrame for EDA")
        return _create_empty_figure("Brak danych do EDA")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    
    # Select top N by variance
    if not numeric_df.empty:
        try:
            variances = numeric_df.var(numeric_only=True).sort_values(ascending=False).dropna()
            top_cols = list(variances.head(max(1, top_numeric)).index)
            LOGGER.debug(f"Selected {len(top_cols)} top features by variance")
        except Exception as e:
            LOGGER.warning(f"Failed to calculate variances: {e}")
            top_cols = list(numeric_df.columns[:top_numeric])
    else:
        LOGGER.warning("No numeric columns found")
        top_cols = []
    
    # Determine grid size
    has_correlation = len(numeric_df.columns) >= 2
    n_rows = 2 if (top_cols and has_correlation) else 1
    n_cols = max(1, len(top_cols)) if top_cols else 1
    
    # Create specs
    if n_rows == 2:
        specs_histograms = [[{"type": "xy"} for _ in range(n_cols)]]
        specs_correlation = [[{"type": "heatmap", "colspan": n_cols}] + [None] * (n_cols - 1)]
        specs = specs_histograms + specs_correlation
        subplot_titles = [f"Histogram: {c}" for c in top_cols] + ["Korelacje (Pearson)"]
    else:
        specs = [[{"type": "xy"} for _ in range(n_cols)]]
        subplot_titles = [f"Histogram: {c}" for c in top_cols]
    
    # Create figure
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=EDA_SPACING,
        specs=specs,
        subplot_titles=tuple(subplot_titles)
    )
    
    # Add histograms
    for i, col in enumerate(top_cols, start=1):
        try:
            hist_fig = histogram(df, col)
            for trace in hist_fig.data:
                fig.add_trace(trace, row=1, col=i)
        except Exception as e:
            LOGGER.warning(f"Failed to create histogram for {col}: {e}")
    
    # Add correlation heatmap
    if n_rows == 2 and has_correlation:
        try:
            corr_fig = correlation_heatmap(df)
            for trace in corr_fig.data:
                fig.add_trace(trace, row=2, col=1)
        except Exception as e:
            LOGGER.warning(f"Failed to create correlation heatmap: {e}")
    
    # Update layout
    height = EDA_HEIGHT_DOUBLE if n_rows == 2 else EDA_HEIGHT_SINGLE
    
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=dict(text=title, x=0.01, xanchor="left"),
        height=height,
        margin=BOARD_MARGIN
    )
    
    return fig


# ========================================================================================
# FORECAST BOARD
# ========================================================================================

def forecast_board(
    forecast_df: pd.DataFrame,
    *,
    history_df: Optional[pd.DataFrame] = None,
    title: str = "Prognoza"
) -> go.Figure:
    """
    Tworzy board z prognozą time series.
    
    Args:
        forecast_df: DataFrame z prognozą (ds, yhat, yhat_lower, yhat_upper)
        history_df: Optional historical data (ds, y)
        title: Tytuł boardu
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = forecast_board(forecast_df, history_df=train_df)
        >>> fig.show()
    """
    # Validate forecast DataFrame
    required_cols = {"ds", "yhat"}
    if not required_cols.issubset(forecast_df.columns):
        LOGGER.error(f"Forecast DataFrame missing required columns: {required_cols}")
        raise ValueError("forecast_df musi mieć kolumny 'ds' i 'yhat'")
    
    # Create base line plot with uncertainty band
    fig = line(
        forecast_df,
        x="ds",
        y="yhat",
        show_uncertainty_band=True,
        title=None  # Set later
    )
    
    # Add historical data if provided
    if history_df is not None and {"ds", "y"}.issubset(history_df.columns):
        hist_clean = history_df.dropna(subset=["ds", "y"]).copy()
        
        if not hist_clean.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_clean["ds"],
                    y=hist_clean["y"],
                    mode="lines",
                    name="historia",
                    line=dict(width=2, color=COLOR_NORMAL),
                    showlegend=True
                )
            )
            LOGGER.debug(f"Added {len(hist_clean)} historical points")
    
    # Update layout
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=dict(text=title, x=0.01, xanchor="left"),
        height=BOARD_HEIGHT,
        margin=BOARD_MARGIN,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    return fig


# ========================================================================================
# MODEL PERFORMANCE BOARD
# ========================================================================================

def model_performance_board(
    metrics: Dict[str, float],
    problem_type: Literal["classification", "regression"],
    *,
    title: Optional[str] = None
) -> go.Figure:
    """
    Tworzy board z metrykami modelu.
    
    Args:
        metrics: Dict z metrykami {metric_name: value}
        problem_type: "classification" lub "regression"
        title: Tytuł boardu (optional)
        
    Returns:
        Plotly Figure
        
    Example:
        >>> metrics = {"accuracy": 0.95, "f1_weighted": 0.93}
        >>> fig = model_performance_board(metrics, "classification")
        >>> fig.show()
    """
    if not metrics:
        LOGGER.warning("No metrics provided")
        return _create_empty_figure("Brak metryk modelu")
    
    # Determine metric order based on problem type
    if problem_type.startswith("class"):
        metric_order = [m for m in CLASSIFICATION_METRICS if m in metrics]
        default_title = "Wyniki modelu (Klasyfikacja)"
    else:
        metric_order = [m for m in REGRESSION_METRICS if m in metrics]
        default_title = "Wyniki modelu (Regresja)"
    
    # If no known metrics, use all
    if not metric_order:
        metric_order = list(metrics.keys())
    
    # Prepare data
    metric_values = [float(metrics[m]) for m in metric_order]
    
    # Format text
    metric_texts = []
    for m in metric_order:
        fmt = METRIC_FORMATS.get(m, ".3f")
        suffix = METRIC_SUFFIXES.get(m, "")
        text = f"{m}: {metrics[m]:{fmt}}{suffix}"
        metric_texts.append(text)
    
    # Create subplots (bar chart + table)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "bar"}, {"type": "table"}]],
        subplot_titles=("Metryki", "Szczegóły")
    )
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=metric_order,
            y=metric_values,
            text=metric_texts,
            textposition="auto",
            name="metryki",
            marker=dict(color=COLOR_NORMAL)
        ),
        row=1, col=1
    )
    
    # Add table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metryka", "Wartość"],
                align="left",
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    metric_order,
                    [f"{metrics[m]:{METRIC_FORMATS.get(m, '.3f')}}{METRIC_SUFFIXES.get(m, '')}" 
                     for m in metric_order]
                ],
                align="left",
                font=dict(size=11)
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=dict(text=title or default_title, x=0.01, xanchor="left"),
        height=BOARD_HEIGHT,
        margin=BOARD_MARGIN,
        showlegend=False
    )
    
    LOGGER.debug(f"Created performance board with {len(metric_order)} metrics")
    
    return fig


# ========================================================================================
# ANOMALY BOARD
# ========================================================================================

def anomaly_board(
    df_scored: pd.DataFrame,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str = "Wykrywanie anomalii"
) -> go.Figure:
    """
    Tworzy board do wizualizacji anomalii.
    
    Jeśli DataFrame zawiera '_is_anomaly' i '_anomaly_score', tworzy:
    - PCA 2D scatter (normal vs anomalies)
    - Histogram anomaly scores
    
    W przeciwnym razie fallback na scatter/histogram.
    
    Args:
        df_scored: DataFrame z wynikami (opcjonalnie z _is_anomaly, _anomaly_score)
        x: Kolumna X dla fallback (optional)
        y: Kolumna Y dla fallback (optional)
        title: Tytuł boardu
        
    Returns:
        Plotly Figure
        
    Example:
        >>> fig = anomaly_board(df_with_anomalies)
        >>> fig.show()
    """
    # Check for anomaly columns
    has_anomaly_flags = {"_is_anomaly", "_anomaly_score"}.issubset(df_scored.columns)
    
    if has_anomaly_flags:
        LOGGER.debug("Found anomaly flags, attempting PCA visualization")
        
        # Select numeric columns for PCA
        numeric_cols = df_scored.select_dtypes(include=np.number)
        numeric_cols = numeric_cols.drop(
            columns=["_is_anomaly", "_anomaly_score"],
            errors="ignore"
        )
        
        # Need at least 2 features and 3 samples for PCA
        if numeric_cols.shape[1] >= 2 and len(numeric_cols) >= 3:
            try:
                from sklearn.decomposition import PCA
                
                # Prepare data
                X = numeric_cols.replace([np.inf, -np.inf], np.nan).fillna(numeric_cols.median())
                
                # Fit PCA
                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X.values)
                
                # Create visualization DataFrame
                viz_df = pd.DataFrame({
                    "PC1": Z[:, 0],
                    "PC2": Z[:, 1],
                    "_is_anomaly": df_scored["_is_anomaly"].astype(int).values,
                    "_anomaly_score": df_scored["_anomaly_score"].values
                })
                
                # Create subplots
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    column_widths=[0.6, 0.4],
                    specs=[[{"type": "xy"}, {"type": "xy"}]],
                    subplot_titles=("PCA: normal vs anomalie", "Rozkład anomaly score")
                )
                
                # Scatter plot (PCA)
                normal_points = viz_df[viz_df["_is_anomaly"] == 0]
                anomaly_points = viz_df[viz_df["_is_anomaly"] == 1]
                
                fig.add_trace(
                    go.Scatter(
                        x=normal_points["PC1"],
                        y=normal_points["PC2"],
                        mode="markers",
                        name="normal",
                        marker=dict(size=6, opacity=0.5, color=COLOR_NORMAL)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points["PC1"],
                        y=anomaly_points["PC2"],
                        mode="markers",
                        name="anomalia",
                        marker=dict(
                            size=8,
                            opacity=0.9,
                            color=COLOR_ANOMALY,
                            line=dict(width=1, color="white")
                        )
                    ),
                    row=1, col=1
                )
                
                # Histogram (anomaly scores)
                hist_fig = histogram(viz_df, col="_anomaly_score")
                for trace in hist_fig.data:
                    fig.add_trace(trace, row=1, col=2)
                
                # Update layout
                fig.update_layout(
                    template=TEMPLATE_NAME,
                    title=dict(text=title, x=0.01, xanchor="left"),
                    height=BOARD_HEIGHT,
                    margin=BOARD_MARGIN
                )
                
                n_anomalies = len(anomaly_points)
                LOGGER.debug(f"Created anomaly board: {n_anomalies} anomalies detected")
                
                return fig
                
            except ImportError:
                LOGGER.warning("sklearn not available for PCA, falling back")
            except Exception as e:
                LOGGER.warning(f"Failed to create PCA visualization: {e}")
    
    # Fallback: scatter or histogram
    if x and y and {x, y}.issubset(df_scored.columns):
        LOGGER.debug(f"Fallback: creating scatter plot ({x} vs {y})")
        fig = scatter(df_scored, x=x, y=y, trendline="ols")
        fig.update_layout(title=title)
        return fig
    
    if x and x in df_scored.columns:
        LOGGER.debug(f"Fallback: creating histogram ({x})")
        fig = histogram(df_scored, col=x)
        fig.update_layout(title=title)
        return fig
    
    if y and y in df_scored.columns:
        LOGGER.debug(f"Fallback: creating histogram ({y})")
        fig = histogram(df_scored, col=y)
        fig.update_layout(title=title)
        return fig
    
    LOGGER.warning("No suitable columns for anomaly visualization")
    return _create_empty_figure("Brak kolumn do wizualizacji anomalii")


# ========================================================================================
# GRID COMPOSER
# ========================================================================================

def compose_grid(
    figures: Sequence[go.Figure],
    rows: int,
    cols: int,
    *,
    title: str = "Dashboard Grid"
) -> go.Figure:
    """
    Komponuje multiple figury w grid layout.
    
    Args:
        figures: Lista figur do złożenia
        rows: Liczba wierszy
        cols: Liczba kolumn
        title: Tytuł całego grid
        
    Returns:
        Plotly Figure z grid
        
    Example:
        >>> fig1 = histogram(df, "col1")
        >>> fig2 = scatter(df, "x", "y")
        >>> grid = compose_grid([fig1, fig2], rows=1, cols=2)
        >>> grid.show()
    """
    if rows * cols < len(figures):
        LOGGER.warning(f"Grid too small: {rows}×{cols} < {len(figures)} figures")
        raise ValueError(f"Za mała siatka ({rows}×{cols}) dla {len(figures)} figur")
    
    # Create specs
    specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    
    # Create subplots
    grid = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        horizontal_spacing=GRID_H_SPACING,
        vertical_spacing=GRID_V_SPACING
    )
    
    # Add figures
    row_idx = 1
    col_idx = 1
    
    for fig in figures:
        for trace in fig.data:
            grid.add_trace(trace, row=row_idx, col=col_idx)
        
        col_idx += 1
        if col_idx > cols:
            col_idx = 1
            row_idx += 1
    
    # Update layout
    grid.update_layout(
        template=TEMPLATE_NAME,
        title=dict(text=title, x=0.01, xanchor="left"),
        height=GRID_ROW_HEIGHT * rows,
        margin=BOARD_MARGIN,
        showlegend=True
    )
    
    LOGGER.debug(f"Created grid: {rows}×{cols} with {len(figures)} figures")
    
    return grid


# ========================================================================================
# UTILITIES
# ========================================================================================

def create_full_dashboard(
    df: pd.DataFrame,
    *,
    include_kpi: bool = True,
    include_eda: bool = True,
    include_correlation: bool = True,
    title: str = "Full Analytics Dashboard"
) -> go.Figure:
    """
    Tworzy kompletny dashboard z multiple komponentami.
    
    Args:
        df: DataFrame do analizy
        include_kpi: Czy dodać KPI board
        include_eda: Czy dodać EDA overview
        include_correlation: Czy dodać correlation heatmap
        title: Tytuł dashboard
        
    Returns:
        Plotly Figure z complete dashboard
    """
    figures = []
    
    if include_kpi:
        figures.append(kpi_board(df, title="Dataset KPIs"))
    
    if include_eda:
        figures.append(eda_overview(df, title="Distribution Analysis"))
    
    if include_correlation and len(df.select_dtypes(include=np.number).columns) >= 2:
        corr_fig = correlation_heatmap(df, title="Feature Correlations")
        figures.append(corr_fig)
    
    if len(figures) == 0:
        return _create_empty_figure("No components to display")
    
    # Determine grid size
    n_figs = len(figures)
    if n_figs == 1:
        return figures[0]
    elif n_figs == 2:
        return compose_grid(figures, rows=1, cols=2, title=title)
    elif n_figs == 3:
        return compose_grid(figures, rows=2, cols=2, title=title)
    else:
        rows = (n_figs + 1) // 2
        return compose_grid(figures, rows=rows, cols=2, title=title)