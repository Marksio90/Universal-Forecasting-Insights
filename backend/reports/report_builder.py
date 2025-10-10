# backend/reports/report_builder.py
# === KONTEKST BIZNESOWY ===
# Generator raportów EDA dla Senior DS:
# - Tworzy samowystarczalny HTML (obrazy Plotly -> base64 przez Kaleido),
# - Renderuje PDF (WeasyPrint),
# - Sekcje: metryki, shape, dtypes, braki, statystyki, histogramy top N kolumn, heatmapa korelacji, próbka wierszy,
# - Dark theme spójny z aplikacją, defensywne błędy.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

import io
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from weasyprint import HTML

# (opcjonalnie) loguru
try:
    from loguru import logger
except Exception:  # pragma: no cover
    class _L:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def error(self, *a, **k): ...
    logger = _L()  # type: ignore


# === USTAWIENIA DOMYŚLNE ===
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 700
DEFAULT_SCALE = 2.0

CSS_DARK = """
:root{
  --bg:#0b0f19; --card:#121826; --muted:#8b95a7; --accent:#6C5CE7;
  --radius:14px; --text:#ffffff; --border:rgba(255,255,255,.08);
}
body{background:var(--bg); color:var(--text); font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:0; padding:24px;}
.container{max-width:1100px; margin:0 auto;}
.card{background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:16px; margin-bottom:16px;}
h1{margin:0 0 8px 0; font-size:28px;}
h2{margin:0 0 8px 0; font-size:20px;}
.small{color:var(--muted); font-size:13px;}
.grid{display:grid; grid-template-columns: repeat(12, 1fr); gap:12px;}
.col-6{grid-column: span 6;}
.col-12{grid-column: span 12;}
table{width:100%; border-collapse:collapse; font-size:13px;}
th,td{border-bottom:1px solid var(--border); padding:8px; text-align:left;}
.kpi{display:flex; gap:16px; flex-wrap:wrap}
.kpi .item{background:linear-gradient(180deg, rgba(108,92,231,.12), rgba(108,92,231,.04)); border:1px solid rgba(108,92,231,.25); padding:12px 16px; border-radius:12px;}
img.plot{width:100%; border-radius:12px; border:1px solid var(--border);}
hr{border:0; border-top:1px solid var(--border); margin:12px 0;}
code{background:#0f1422; padding:2px 6px; border-radius:6px; border:1px solid var(--border);}
"""

@dataclass
class ReportPaths:
    html_path: str
    pdf_path: Optional[str] = None


# === POMOCNICZE ===
def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fig_to_base64(fig: go.Figure, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, scale: float = DEFAULT_SCALE) -> str:
    """Render Plotly figure do PNG (Kaleido) i zwróć data URI base64."""
    png: bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    b64 = base64.b64encode(png).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _safe_sample(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    n = min(n, len(df))
    return df.head(n)

def _format_metrics(metrics: Dict) -> str:
    items = [f'<div class="item"><div class="small">{k}</div><div style="font-weight:800">{v}</div></div>' for k, v in metrics.items()]
    return '<div class="kpi">' + "".join(items) + '</div>'

def _num_cols(df: pd.DataFrame, max_cols: int | None = None) -> List[str]:
    cols = list(df.select_dtypes(include="number").columns)
    if max_cols is not None:
        cols = cols[:max_cols]
    return cols

def _describe_table(df: pd.DataFrame, max_cols: int = 12) -> str:
    if df.select_dtypes(include="number").empty:
        return "<p class='small'>Brak kolumn numerycznych.</p>"
    desc = df.select_dtypes(include="number").describe().T
    if len(desc) > max_cols:
        desc = desc.iloc[:max_cols]
    return desc.to_html(classes="table", border=0)

def _missing_table(df: pd.DataFrame, top_k: int = 20) -> str:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return "<p class='small'>Brak braków danych 🎉</p>"
    out = pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)})
    out = out.head(top_k)
    return out.to_html(classes="table", border=0)

def _corr_heatmap_datauri(df: pd.DataFrame) -> Optional[str]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, aspect="auto", title="Macierz korelacji")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return _fig_to_base64(fig, width=1100, height=700, scale=2)

def _hists_grid_datauris(df: pd.DataFrame, max_cols: int = 6) -> List[Tuple[str, str]]:
    cols = _num_cols(df, max_cols=max_cols)
    out: List[Tuple[str, str]] = []
    for c in cols:
        try:
            fig = px.histogram(df, x=c, nbins=30, title=f"Histogram — {c}")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            out.append((c, _fig_to_base64(fig, width=1100, height=600, scale=2)))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Histogram error for {c}: {e}")
    return out


# === API PUBLICZNE ===
def build_html_summary(
    df: pd.DataFrame,
    metrics: Dict,
    *,
    title: str = "DataGenius Raport",
    output_dir: str | Path = "reports",
    filename: str = "summary.html",
    max_hist_cols: int = 6,
    include_corr: bool = True,
) -> str:
    """
    Tworzy samowystarczalny raport HTML (obrazy osadzone base64).
    Zwraca ścieżkę do zapisanego pliku HTML.
    """
    out_dir = _ensure_dir(output_dir)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Sekcje danych
    shape = f"{df.shape}"
    dtypes = pd.Series({c: str(t) for c, t in df.dtypes.items()})
    dtypes_tbl = dtypes.to_frame("dtype").to_html(classes="table", border=0)

    miss_tbl = _missing_table(df)
    desc_tbl = _describe_table(df)

    # Wykresy
    corr_uri = _corr_heatmap_datauri(df) if include_corr else None
    hists = _hists_grid_datauris(df, max_cols=max_hist_cols)

    sample_tbl = _safe_sample(df, n=12).to_html(classes="table", border=0)

    # HTML templating
    html = f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>{CSS_DARK}</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>{title}</h1>
    <div class="small">Wygenerowano: {now}</div>
  </div>

  <div class="card">
    <h2>Metryki</h2>
    {_format_metrics(metrics)}
  </div>

  <div class="card">
    <h2>Informacje o zbiorze</h2>
    <div class="kpi">
      <div class="item"><div class="small">Shape</div><div style="font-weight:800">{shape}</div></div>
      <div class="item"><div class="small">Kolumn</div><div style="font-weight:800">{len(df.columns)}</div></div>
      <div class="item"><div class="small">Wierszy</div><div style="font-weight:800">{len(df)}</div></div>
    </div>
    <hr/>
    <h3>Dtypes</h3>
    {dtypes_tbl}
  </div>

  <div class="card">
    <h2>Braki danych</h2>
    {miss_tbl}
  </div>

  <div class="card">
    <h2>Statystyki opisowe</h2>
    {desc_tbl}
  </div>

  {"<div class='card'><h2>Macierz korelacji</h2><img class='plot' src='"+corr_uri+"' /></div>" if corr_uri else ""}

  <div class="card">
    <h2>Histogramy</h2>
    {"".join([f"<h3 style='margin:8px 0'>{c}</h3><img class='plot' src='{uri}'/>" for c,uri in hists]) or "<p class='small'>Brak kolumn numerycznych.</p>"}
  </div>

  <div class="card">
    <h2>Próbka wierszy</h2>
    {sample_tbl}
  </div>
</div>
</body>
</html>
"""

    out_html = out_dir / filename
    out_html.write_text(html, encoding="utf-8")
    logger.info(f"Report HTML saved: {out_html}")  # type: ignore[attr-defined]
    return str(out_html)


def build_pdf_from_html(html_path_or_html: str, *, output_dir: str | Path = "reports", filename: str = "summary.pdf") -> str:
    """
    Renderuje PDF z HTML. Przyjmuje ścieżkę do pliku HTML **lub** tekst HTML.
    Zwraca ścieżkę do zapisanego PDF.
    """
    out_dir = _ensure_dir(output_dir)
    out_pdf = out_dir / filename

    # Jeżeli podano ścieżkę – czytamy z pliku; jeśli to surowy HTML – renderujemy bezpośrednio.
    path = Path(html_path_or_html)
    if path.exists():
        HTML(filename=str(path)).write_pdf(str(out_pdf))
    else:
        HTML(string=html_path_or_html).write_pdf(str(out_pdf))

    logger.info(f"Report PDF saved: {out_pdf}")  # type: ignore[attr-defined]
    return str(out_pdf)


# === WERSJA „WSZYSTKO W JEDNYM” ===
def build_full_report(
    df: pd.DataFrame,
    metrics: Dict,
    *,
    title: str = "DataGenius Raport",
    output_dir: str | Path = "reports",
    html_name: str = "summary.html",
    pdf_name: str = "summary.pdf",
    max_hist_cols: int = 6,
    include_corr: bool = True,
) -> ReportPaths:
    """
    Buduje komplet: HTML + PDF, zwraca ścieżki.
    """
    html_path = build_html_summary(
        df, metrics, title=title, output_dir=output_dir, filename=html_name,
        max_hist_cols=max_hist_cols, include_corr=include_corr
    )
    pdf_path = build_pdf_from_html(html_path, output_dir=output_dir, filename=pdf_name)
    return ReportPaths(html_path=html_path, pdf_path=pdf_path)
