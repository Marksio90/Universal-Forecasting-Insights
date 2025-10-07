# src/visualization/dashboards.py
from __future__ import annotations
from typing import Optional, Sequence, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# nasze wykresy i motyw
from .charts import (
    histogram,
    scatter,
    line,
    correlation_heatmap,
)

# opcjonalnie: szybkie metryki jakości (jeśli dostępne)
try:
    from src.utils.validators import basic_quality_checks  # type: ignore
except Exception:
    basic_quality_checks = None  # fallback


# =========================================================
# KPI BOARD
# =========================================================
def kpi_board(
    df: Optional[pd.DataFrame] = None,
    *,
    stats: Optional[Dict] = None,
    title: str = "KPI danych",
) -> go.Figure:
    """
    Tworzy rząd wskaźników KPI dla danych wejściowych.
    Jeśli podasz `stats`, użyje ich bez liczenia (kompatybilne z basic_quality_checks).
    """
    if stats is None:
        if df is None:
            raise ValueError("Podaj `df` lub `stats`.")
        if basic_quality_checks:
            stats = basic_quality_checks(df)  # rozszerzony raport
        else:
            stats = {
                "rows": len(df),
                "cols": df.shape[1],
                "missing_pct": float((df.isna().sum().sum()) / max(1, df.size)),
                "dupes": int(df.duplicated().sum()),
            }

    rows = int(stats.get("rows", 0))
    cols = int(stats.get("cols", 0))
    missing_pct = float(stats.get("missing_pct", 0.0)) * (100.0 if stats.get("missing_pct", 0.0) <= 1 else 1)
    dupes = int(stats.get("dupes", 0))
    dupes_pct = float(stats.get("dupes_pct", dupes / rows if rows else 0.0)) * 100.0

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=("Wiersze", "Kolumny", "Braki [%]", "Duplikaty [%]"),
    )

    fig.add_trace(go.Indicator(mode="number", value=rows, number={"valueformat": ",.0f"}), 1, 1)
    fig.add_trace(go.Indicator(mode="number", value=cols, number={"valueformat": ",.0f"}), 1, 2)
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=missing_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={"shape": "bullet", "axis": {"range": [0, 100]}},
        ),
        1,
        3,
    )
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=dupes_pct,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={"shape": "bullet", "axis": {"range": [0, 100]}},
        ),
        1,
        4,
    )
    fig.update_layout(
        template="ip_business_dark",
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=dict(l=20, r=20, t=60, b=10),
        height=160,
        showlegend=False,
    )
    return fig


# =========================================================
# EDA OVERVIEW
# =========================================================
def eda_overview(
    df: pd.DataFrame,
    *,
    top_numeric: int = 4,
    title: str = "EDA – rozkłady i korelacje",
) -> go.Figure:
    """
    Grid: top N histogramów (po wariancji) + heatmapa korelacji.
    """
    if df is None or df.empty:
        return _empty("Brak danych do EDA")

    num = df.select_dtypes(include=np.number)
    # wybierz top N po wariancji (z ochroną przed NaN)
    if not num.empty:
        vars_ = (num.var(numeric_only=True)).sort_values(ascending=False).dropna()
        top_cols = list(vars_.head(max(1, top_numeric)).index)
    else:
        top_cols = []

    rows = 2 if top_cols else 1
    cols = max(1, len(top_cols))
    specs_top = [[{"type": "xy"} for _ in range(cols)]]
    specs_corr = [[{"type": "heatmap", "colspan": cols}] + [None] * (cols - 1)]
    specs = specs_top + specs_corr

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.12 if rows == 2 else 0.08,
        specs=specs,
        subplot_titles=tuple([f"Histogram: {c}" for c in top_cols] + (["Korelacje (Pearson)"] if rows == 2 else [])),
    )

    # Histogramy
    for i, c in enumerate(top_cols, start=1):
        h = histogram(df, c)
        for tr in h.data:
            fig.add_trace(tr, row=1, col=i)

    # Korelacje
    if rows == 2:
        corr_fig = correlation_heatmap(df)
        for tr in corr_fig.data:
            fig.add_trace(tr, row=2, col=1)

    fig.update_layout(
        template="ip_business_dark",
        title=dict(text=title, x=0.01, xanchor="left"),
        height=600 if rows == 2 else 360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# =========================================================
# FORECAST BOARD
# =========================================================
def forecast_board(
    fcst: pd.DataFrame,
    *,
    history: Optional[pd.DataFrame] = None,   # np. df z kolumnami "ds" i "y"
    title: str = "Prognoza",
) -> go.Figure:
    """
    Rysuje prognozę (yhat + pasmo) + (opcjonalnie) serię historyczną.
    """
    assert "ds" in fcst.columns and "yhat" in fcst.columns, "fcst musi mieć kolumny 'ds' i 'yhat'."

    # główna linia prognozy
    lf = line(fcst, x="ds", y="yhat", show_uncertainty_band=True)
    fig = go.Figure(lf.data)

    # historia
    if history is not None and {"ds", "y"}.issubset(set(history.columns)):
        hist = history.dropna(subset=["ds", "y"]).copy()
        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=hist["y"],
                mode="lines",
                name="historia",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        template="ip_business_dark",
        title=dict(text=title, x=0.01, xanchor="left"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# =========================================================
# MODEL PERFORMANCE BOARD
# =========================================================
_CLF_ORDER = ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]
_REG_ORDER = ["rmse", "mae", "r2", "mape"]

def model_performance_board(
    metrics: Dict[str, float],
    problem_type: str,
    *,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Prosty, czytelny panel metryk dla klasyfikacji/regresji.
    """
    if not metrics:
        return _empty("Brak metryk modelu")

    if problem_type.startswith("class"):
        order = [m for m in _CLF_ORDER if m in metrics]
        fmt = {"accuracy": ".3f", "balanced_accuracy": ".3f", "f1_weighted": ".3f", "roc_auc": ".3f"}
        suffix = {"accuracy": "", "balanced_accuracy": "", "f1_weighted": "", "roc_auc": ""}
    else:
        order = [m for m in _REG_ORDER if m in metrics]
        fmt = {"rmse": ".3f", "mae": ".3f", "r2": ".3f", "mape": ".2f"}
        suffix = {"mape": "%"}  # reszta bez sufiksu

    vals = [float(metrics[m]) for m in order]
    texts = [
        f"{m}: {metrics[m]:{fmt.get(m, '.3f')}}{suffix.get(m,'')}"
        for m in order
    ]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], specs=[[{"type": "bar"}, {"type": "table"}]])

    # bar
    fig.add_trace(go.Bar(x=order, y=vals, text=texts, textposition="auto", name="metryki"), 1, 1)

    # tabela
    fig.add_trace(
        go.Table(
            header=dict(values=["Metryka", "Wartość"], align="left"),
            cells=dict(values=[order, [metrics[m] for m in order]], align="left"),
        ),
        1,
        2,
    )

    fig.update_layout(
        template="ip_business_dark",
        title=dict(text=title or "Wyniki modelu", x=0.01, xanchor="left"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


# =========================================================
# ANOMALY BOARD (PCA 2D)
# =========================================================
def anomaly_board(
    df_scored: pd.DataFrame,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: str = "Wykrywanie anomalii",
) -> go.Figure:
    """
    Jeśli są kolumny `_is_anomaly` i `_anomaly_score` → wizualizuje PCA 2D + histogram score.
    W przeciwnym razie fallback na scatter/histogram wskazanych kolumn.
    """
    has_flags = {"_is_anomaly", "_anomaly_score"}.issubset(df_scored.columns)

    if has_flags:
        # wybierz numeryczne do PCA
        num = df_scored.select_dtypes(include=np.number).drop(columns=["_is_anomaly", "_anomaly_score"], errors="ignore").copy()
        if num.shape[1] >= 2 and len(num) >= 3:
            try:
                from sklearn.decomposition import PCA  # local import
                X = num.replace([np.inf, -np.inf], np.nan).fillna(num.median())
                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X.values)
                dd = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1]})
                dd["_is_anomaly"] = df_scored["_is_anomaly"].astype(int).values
                dd["_anomaly_score"] = df_scored["_anomaly_score"].values

                fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], specs=[[{"type": "xy"}, {"type": "xy"}]],
                                    subplot_titles=("PCA: normal vs anomalie", "Rozkład score"))
                # scatter
                normal = dd[dd["_is_anomaly"] == 0]
                anoms = dd[dd["_is_anomaly"] == 1]
                fig.add_trace(
                    go.Scatter(
                        x=normal["PC1"], y=normal["PC2"], mode="markers", name="normal",
                        marker=dict(size=6, opacity=0.5)
                    ),
                    1, 1
                )
                fig.add_trace(
                    go.Scatter(
                        x=anoms["PC1"], y=anoms["PC2"], mode="markers", name="anomalia",
                        marker=dict(size=7, opacity=0.9, color="#f87171", line=dict(width=0))
                    ),
                    1, 1
                )
                # histogram score
                h = histogram(dd, col="_anomaly_score")
                for tr in h.data:
                    fig.add_trace(tr, 1, 2)

                fig.update_layout(
                    template="ip_business_dark",
                    title=dict(text=title, x=0.01, xanchor="left"),
                    height=420,
                    margin=dict(l=20, r=20, t=60, b=20),
                )
                return fig
            except Exception:
                pass

    # fallback: jeżeli nie ma flag – sensowny scatter/histogram
    if x and y:
        f = scatter(df_scored, x=x, y=y, trendline="ols")
        f.update_layout(title=title)
        return f
    elif x or y:
        col = x or y
        f = histogram(df_scored, col=col)
        f.update_layout(title=title)
        return f
    return _empty("Brak kolumn do wizualizacji anomalii")


# =========================================================
# COMPOSER / UTILS
# =========================================================
def compose_grid(figs: Sequence[go.Figure], rows: int, cols: int, *, title: str = "") -> go.Figure:
    """
    Skleja kilka figur w siatkę subplotów (bez legend duplikowanych).
    """
    assert rows * cols >= len(figs), "Za mała siatka względem liczby figur."
    specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    grid = make_subplots(rows=rows, cols=cols, specs=specs, horizontal_spacing=0.08, vertical_spacing=0.12)

    r = c = 1
    for f in figs:
        for tr in f.data:
            grid.add_trace(tr, r, c)
        c += 1
        if c > cols:
            c = 1
            r += 1

    grid.update_layout(
        template="ip_business_dark",
        title=dict(text=title, x=0.01, xanchor="left"),
        height=360 * rows,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
    )
    return grid


def _empty(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="ip_business_dark",
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
