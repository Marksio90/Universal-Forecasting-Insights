# src/visualization/reports.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Dict, Any
import html as _html
import io
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# Ustawienia / motyw
# =========================
_PRIMARY = "#4A90E2"
_BG = "#ffffff"
_TEXT = "#111"
_MUTED = "#6b7280"
_CARD = "#f8f9fb"
_BORDER = "#e5e7eb"

# =========================
# Utility
# =========================
def _esc(s: Any) -> str:
    return _html.escape("" if s is None else str(s), quote=True)

def _fmt_num(v: Any, ndigits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, (int,)):
            return f"{v:,}".replace(",", " ")
        fv = float(v)
        if abs(fv) >= 1000:
            return f"{fv:,.0f}".replace(",", " ")
        return f"{fv:,.{ndigits}f}".rstrip("0").rstrip(".")
    except Exception:
        return _esc(v)

def _wrap_card(inner_html: str, title: Optional[str] = None) -> str:
    head = f'<div class="card-title">{_esc(title)}</div>' if title else ""
    return f'<div class="card">{head}{inner_html}</div>'

def _inline_css() -> str:
    return f"""
<style>
:root {{
  --primary: {_PRIMARY};
  --bg: {_BG};
  --text: {_TEXT};
  --muted: {_MUTED};
  --card: {_CARD};
  --border: {_BORDER};
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:24px; font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; color: var(--text); background: var(--bg); }}
h1,h2,h3 {{ letter-spacing:.2px; color:#1d3557; margin: 0 0 .6rem; }}
h1 {{ font-size: 1.8rem; }}
h2 {{ font-size: 1.35rem; border-bottom:1px solid var(--border); padding-bottom:.25rem; }}
h3 {{ font-size: 1.1rem; color:#1f2937; }}
p,li {{ line-height: 1.55; }}
.small {{ color: var(--muted); font-size:.9rem; }}
.row {{ display:flex; gap:16px; flex-wrap: wrap; align-items: stretch; }}
.card {{
  background: var(--card);
  border:1px solid var(--border);
  border-radius:12px; padding:16px; flex:1 1 260px;
  box-shadow: 0 1px 0 rgba(16,24,40,.04);
}}
.card-title {{ font-weight:600; margin-bottom:8px; color:#111827; }}
.kpi {{ display:flex; flex-direction:column; gap:6px; }}
.kpi .value {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.kpi .label {{ color: var(--muted); font-size:.9rem; }}
.kpi .delta.up {{ color:#16a34a; }}
.kpi .delta.down {{ color:#dc2626; }}
.table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:12px; }}
table.tbl {{ width:100%; border-collapse: collapse; font-size:.95rem; }}
table.tbl th, table.tbl td {{ padding:8px 10px; border-bottom:1px solid var(--border); }}
table.tbl th {{ text-align:left; font-weight:600; background:#fafafa; }}
.fig {{ border:1px solid var(--border); border-radius:12px; padding:8px; background:#fff; }}
.caption {{ color:var(--muted); font-size:.9rem; margin:.25rem 2px 0; }}
hr {{ border: none; border-top:1px solid var(--border); margin: 16px 0; }}
.badge {{ display:inline-block; padding:.2rem .5rem; border-radius:999px; background: rgba(74,144,226,.1); color: var(--primary); font-weight:600; font-size:.8rem; }}
footer {{ color:var(--muted); margin-top:24px; }}
</style>
""".strip()

# =========================
# Public helpers
# =========================
def section_title(text: str, level: int = 2, anchor: Optional[str] = None) -> str:
    tag = min(max(level, 1), 3)
    id_attr = f' id="{_esc(anchor)}"' if anchor else ""
    return f"<h{tag}{id_attr}>{_esc(text)}</h{tag}>"

def paragraph(text: str) -> str:
    return f"<p>{_esc(text)}</p>"

def badge(text: str) -> str:
    return f'<span class="badge">{_esc(text)}</span>'

def kpi_card(label: str, value: Any, delta: Optional[float | str] = None, *, good_is_up: bool = True) -> str:
    if isinstance(delta, (int, float)):
        cls = "up" if (delta >= 0 and good_is_up) or (delta < 0 and not good_is_up) else "down"
        delta_html = f'<div class="delta {cls}">{_fmt_num(delta)}%</div>'
    elif delta is None:
        delta_html = ""
    else:
        delta_html = f'<div class="delta">{_esc(delta)}</div>'
    html = f"""
<div class="kpi">
  <div class="label">{_esc(label)}</div>
  <div class="value">{_fmt_num(value)}</div>
  {delta_html}
</div>
""".strip()
    return _wrap_card(html)

def kpi_row(items: Sequence[Dict[str, Any]]) -> str:
    # items: [{"label":..,"value":..,"delta":..,"good_is_up":True}, ...]
    cards = [kpi_card(i.get("label","—"), i.get("value"), i.get("delta"), good_is_up=bool(i.get("good_is_up", True))) for i in items]
    return f'<div class="row">{"".join(cards)}</div>'

def dataframe_table(df: pd.DataFrame, *, index: bool = False, max_rows: int = 1000, caption: Optional[str] = None) -> str:
    if df is None or df.empty:
        return _wrap_card('<div class="small">Brak danych</div>')
    dd = df.head(max_rows)
    html_tbl = dd.to_html(classes="tbl", index=index, border=0, escape=False)
    cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
    return f'<div class="table-wrap">{html_tbl}</div>{cap}'

def fig_to_html(
    fig: go.Figure,
    *,
    full_html: bool = False,
    include_plotlyjs: str = "cdn",  # "cdn" | "inline" | False
    config: Optional[Dict[str, Any]] = None,
    div_id: Optional[str] = None,
    caption: Optional[str] = None,
) -> str:
    """
    Zwraca <div> z interaktywnym wykresem (Plotly).
    Jeśli full_html=True, zwraca kompletny dokument HTML (niezalecane w raporcie wielosekcyjnym).
    """
    config = config or {"displayModeBar": True, "responsive": True, "scrollZoom": False}
    html_div = pio.to_html(fig, full_html=full_html, include_plotlyjs=include_plotlyjs, config=config, div_id=div_id)
    if full_html:
        return html_div
    cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
    return f'<div class="fig">{html_div}</div>{cap}'

def fig_to_img(
    fig: go.Figure,
    *,
    fmt: str = "png",
    scale: float = 2.0,
    caption: Optional[str] = None,
) -> str:
    """
    Przekształca wykres do <img> (wymaga zainstalowanego 'kaleido').
    Jeśli kaleido jest niedostępne, zwróci interaktywny div via fig_to_html().
    """
    try:
        img_bytes = pio.to_image(fig, format=fmt, scale=scale)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        mime = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
        return f'<div class="fig"><img src="data:{mime};base64,{b64}" style="width:100%;border-radius:8px;" />{cap}</div>'
    except Exception:
        # Fallback na HTML
        return fig_to_html(fig, full_html=False, include_plotlyjs="cdn", caption=caption)

def metrics_table(metrics: Dict[str, Any], *, title: Optional[str] = None, order: Optional[Iterable[str]] = None) -> str:
    if not metrics:
        return _wrap_card('<div class="small">Brak metryk</div>', title=title)
    keys = list(order) if order else list(metrics.keys())
    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(metrics[k])}</td></tr>"
        for k in keys if k in metrics
    )
    inner = f'<table class="tbl"><thead><tr><th>Metryka</th><th>Wartość</th></tr></thead><tbody>{rows}</tbody></table>'
    return _wrap_card(inner, title=title)

def card(title: str, body_html: str) -> str:
    return _wrap_card(body_html, title=title)

def hr() -> str:
    return "<hr/>"

# =========================
# Builder raportu
# =========================
class ReportBuilder:
    """
    Prosty składacz HTML. Użycie:
        rb = ReportBuilder("Raport")
        rb.add(section_title("KPI"))
        rb.add(kpi_row([...]))
        rb.add(section_title("Wykres"))
        rb.add(fig_to_html(fig))
        html = rb.build()
    """
    def __init__(self, title: str, *, include_css: bool = True, notes: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        self.title = title
        self.include_css = include_css
        self.notes = notes
        self.meta = meta or {}
        self.parts: List[str] = []

    def add(self, html_fragment: str) -> None:
        self.parts.append(html_fragment)

    def add_markdown(self, text: str) -> None:
        # proste md -> p + <br> (bez zewn. zależności)
        for para in (text or "").split("\n\n"):
            self.parts.append(paragraph(para.replace("\n", "<br/>")))

    def build(self) -> str:
        head_css = _inline_css() if self.include_css else ""
        meta_html = ""
        if self.meta:
            import json
            meta_html = f'<div class="small">Kontekst: <code>{_esc(json.dumps(self.meta, ensure_ascii=False))}</code></div>'

        notes_html = f"<p>{_esc(self.notes)}</p>" if self.notes else ""

        body = "\n".join(self.parts)
        html = f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<title>{_esc(self.title)}</title>
{head_css}
</head>
<body>
<h1>{_esc(self.title)}</h1>
{meta_html}
{notes_html}
{body}
<footer><em>Wygenerowano przez Intelligent Predictor.</em></footer>
</body>
</html>"""
        return html
