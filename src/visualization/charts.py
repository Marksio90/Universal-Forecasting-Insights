# src/visualization/charts.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

PRIMARY = "#4A90E2"
BG = "#0E1117"          # paper bg
PLOT_BG = "#111827"     # axes bg
GRID = "rgba(255,255,255,0.06)"
FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

# =========================
# Motyw / szablon
# =========================
def _register_business_dark_template() -> None:
    template_name = "ip_business_dark"
    if template_name in pio.templates:
        pio.templates.default = template_name
        return

    base = pio.templates["plotly_dark"]
    layout = base.layout.update(
        font=dict(family=FONT, size=14),
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        colorway=[
            PRIMARY, "#22d3ee", "#a78bfa", "#34d399", "#f59e0b",
            "#f472b6", "#60a5fa", "#f87171", "#4ade80", "#c084fc"
        ],
        xaxis=dict(
            gridcolor=GRID, zeroline=False, showspikes=True, spikedash="dot",
            ticks="outside", tickcolor=GRID, linecolor=GRID
        ),
        yaxis=dict(
            gridcolor=GRID, zeroline=False, showspikes=True, spikedash="dot",
            ticks="outside", tickcolor=GRID, linecolor=GRID
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02
        ),
        margin=dict(l=60, r=20, t=60, b=50),
        hoverlabel=dict(bgcolor="#0b1220", bordercolor="#1f2937", font=dict(color="#e5e7eb")),
    )
    pio.templates[template_name] = go.layout.Template(layout=layout)
    pio.templates.default = template_name

_register_business_dark_template()

# =========================
# Helpery
# =========================
def _coerce_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() > 0.6:
            return dt
    except Exception:
        pass
    return s

def _fd_bins(x: np.ndarray, min_bins: int = 8, max_bins: int = 150) -> int:
    """Freedman–Diaconis binning (z osłonami)."""
    x = x[~np.isnan(x)]
    if x.size < 2:
        return min_bins
    q75, q25 = np.percentile(x, [75, 25])
    iqr = max(q75 - q25, 1e-9)
    bw = 2 * iqr * (x.size ** (-1 / 3))
    if bw <= 0:
        return min_bins
    nb = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(nb, min_bins, max_bins))

def _is_large(n: int, threshold: int = 50_000) -> bool:
    return n >= threshold

def _apply_common_layout(fig: go.Figure, title: Optional[str] = None, x_title: Optional[str] = None, y_title: Optional[str] = None) -> go.Figure:
    fig.update_layout(title=dict(text=title or "", x=0.02, xanchor="left"))
    if x_title:
        fig.update_xaxes(title=x_title)
    if y_title:
        fig.update_yaxes(title=y_title)
    # lekkie zaokrąglenie narożników tooltipa (hack przez padding/margins – reszta w template)
    return fig

def _drop_na_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df[[x, y]].dropna()

# =========================
# Wykresy główne
# =========================
def histogram(
    df: pd.DataFrame,
    col: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    histnorm: Optional[str] = None,  # None | "percent" | "probability" | "density"
    nbins: Optional[int] = None,
    show_mean: bool = True,
    show_median: bool = True,
    marginal: Optional[str] = None,   # "box" | "violin" | "rug" | None
    opacity: float = 0.85,
    **kwargs,
) -> go.Figure:
    """
    Zaawansowany histogram z adaptacyjnymi binami i liniami referencyjnymi.
    """
    assert col in df.columns, f"Kolumna '{col}' nie istnieje."
    s = pd.to_numeric(df[col], errors="coerce")
    # Jeśli kolumna nie jest numeryczna po cast → spróbuj potraktować ją jako kategorie (bar)
    if s.notna().mean() < 0.6:
        # upadek na wykres słupkowy częstości kategorii
        vc = df[col].astype(str).fillna("(NaN)").value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", color=None)
        fig.update_traces(opacity=opacity)
        return _apply_common_layout(fig, title or f"Rozkład: {col}", x_title=col, y_title="Liczność")

    x = s.values
    if nbins is None:
        nbins = _fd_bins(x)

    fig = px.histogram(
        df.assign(_num_=s),
        x="_num_",
        color=color if (color in df.columns) else None,
        nbins=nbins,
        opacity=opacity,
        histnorm=histnorm,
        marginal=marginal,
        **kwargs,
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>count=%{y}<extra></extra>")
    fig = _apply_common_layout(fig, title or f"Histogram: {col}", x_title=col, y_title="Liczność" if not histnorm else histnorm)

    mean_v = float(np.nanmean(x))
    median_v = float(np.nanmedian(x))

    shapes = []
    annotations = []
    if show_mean:
        shapes.append(dict(type="line", x0=mean_v, x1=mean_v, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#22d3ee", width=2, dash="dot")))
        annotations.append(dict(x=mean_v, y=1.02, yref="paper", text="mean", showarrow=False, font=dict(size=12)))
    if show_median:
        shapes.append(dict(type="line", x0=median_v, x1=median_v, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#f59e0b", width=2, dash="dot")))
        annotations.append(dict(x=median_v, y=1.02, yref="paper", text="median", showarrow=False, font=dict(size=12)))

    if shapes:
        fig.update_layout(shapes=shapes, annotations=annotations)

    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[Sequence[str]] = None,
    trendline: Optional[str] = "ols",  # "ols" | "lowess" | None
    add_identity: bool = False,
    opacity: Optional[float] = None,
    **kwargs,
) -> go.Figure:
    """
    Rozszerzony scatter:
      - auto webgl dla dużych zbiorów,
      - trendline (ols/lowess) jeśli dostępny,
      - opcjonalna linia y=x,
      - ogarnięte hovery i brakujące wartości.
    """
    assert x in df.columns and y in df.columns, f"Kolumny '{x}' lub '{y}' nie istnieją."
    dff = _drop_na_xy(df, x, y).copy()
    if dff.empty:
        return go.Figure()

    # konwersja osi czasu jeśli trzeba
    dff[x] = _coerce_datetime(dff[x])
    dff[y] = pd.to_numeric(dff[y], errors="coerce") if not pd.api.types.is_datetime64_any_dtype(dff[y]) else dff[y]
    dff = dff.dropna()

    render_mode = "webgl" if _is_large(len(dff)) else "auto"
    if opacity is None:
        opacity = 0.5 if _is_large(len(dff), 15_000) else 0.85

    fig = px.scatter(
        dff,
        x=x,
        y=y,
        color=color if (color in dff.columns) else None,
        size=size if (size in dff.columns) else None,
        hover_data=list(hover_data) if hover_data else None,
        trendline=trendline,
        render_mode=render_mode,
        **kwargs,
    )
    fig.update_traces(marker=dict(opacity=opacity, line=dict(width=0)), selector=dict(mode="markers"))
    fig.update_layout(hovermode="x unified")
    fig = _apply_common_layout(fig, title or f"Scatter: {x} vs {y}", x_title=x, y_title=y)

    if add_identity and not pd.api.types.is_datetime64_any_dtype(dff[y]) and not pd.api.types.is_datetime64_any_dtype(dff[x]):
        # y=x
        mn = float(np.nanmin([dff[x].min(), dff[y].min()]))
        mx = float(np.nanmax([dff[x].max(), dff[y].max()]))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(color="#64748b", dash="dash"), name="y=x", showlegend=True))

    return fig


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: Optional[str] = None,
    color: Optional[str] = None,
    markers_auto: bool = True,
    show_uncertainty_band: bool = True,  # rozpoznaje kolumny yhat_lower/yhat_upper
    **kwargs,
) -> go.Figure:
    """
    Wykres liniowy z auto-czasem i pasmem niepewności (jeśli dostępne).
    """
    assert x in df.columns and y in df.columns, f"Kolumny '{x}' lub '{y}' nie istnieją."
    dff = df[[x, y] + ([color] if color and color in df.columns else [])].copy()
    dff[x] = _coerce_datetime(dff[x])
    dff = dff.dropna(subset=[x, y]).sort_values(by=x)

    fig = px.line(
        dff,
        x=x,
        y=y,
        color=color if (color in dff.columns) else None,
        markers=(markers_auto and len(dff) <= 500),
        **kwargs,
    )
    fig = _apply_common_layout(fig, title or f"Linia: {y} w czasie", x_title=x, y_title=y)

    # Pasmo niepewności: szukaj kolumn z sufiksami
    lower_candidates = [c for c in df.columns if c.lower().endswith("yhat_lower") or c.lower().endswith("lower")]
    upper_candidates = [c for c in df.columns if c.lower().endswith("yhat_upper") or c.lower().endswith("upper")]
    if show_uncertainty_band and lower_candidates and upper_candidates:
        lo = lower_candidates[0]
        hi = upper_candidates[0]
        try:
            band = df[[x, lo, hi]].copy()
            band[x] = _coerce_datetime(band[x])
            band = band.dropna().sort_values(by=x)
            if not band.empty:
                fig.add_traces([
                    go.Scatter(
                        x=band[x], y=band[hi],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    go.Scatter(
                        x=band[x], y=band[lo],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(74,144,226,0.15)",  # PRIMARY z alfa
                        name="przedział",
                        hoverinfo="skip"
                    )
                ])
        except Exception:
            pass

    return fig

# =========================
# Dodatkowe (przydatne) wykresy – opcjonalnie
# =========================
def correlation_heatmap(df: pd.DataFrame, title: str = "Korelacje (Pearson)") -> go.Figure:
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return go.Figure()
    corr = num.corr().round(3)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues", origin="lower")
    fig.update_coloraxes(colorbar_title="ρ")
    return _apply_common_layout(fig, title=title)

def feature_importance(fig_importances: Sequence[Tuple[str, float]], top: int = 25, title: str = "Ważność cech") -> go.Figure:
    data = sorted(list(fig_importances), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    if not data:
        return go.Figure()
    feat, val = zip(*data)
    dd = pd.DataFrame({"feature": feat, "importance": val})
    fig = px.bar(dd, x="importance", y="feature", orientation="h")
    return _apply_common_layout(fig, title=title, x_title="Ważność", y_title="Cecha")
