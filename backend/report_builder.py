# backend/reports/pdf_builder.py
# === KONTEKST BIZNESOWY ===
# Profesjonalny generator raportów: HTML -> PDF (WeasyPrint).
# - Samowystarczalny HTML (obrazy base64), A4, header/footer z numeracją
# - Light/Dark theme, siatka KPI, tabele z DataFrame, notatki (Markdown/HTML)
# - Drop-in: build_pdf(path_out, context) kompatybilne z Twoim pierwotnym API.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Iterable, List
from pathlib import Path
import base64
import io
import html as _html

import pandas as pd  # do tabel sekcji
from weasyprint import HTML

# Opcjonalne biblioteki — jeśli są, używamy (bez twardej zależności)
try:
    import markdown as _md  # type: ignore
except Exception:
    _md = None  # pragma: no cover

try:
    import bleach  # type: ignore
except Exception:
    bleach = None  # pragma: no cover

# === NAZWA_SEKCJI === DATAMODEL
@dataclass
class ReportContext:
    title: str = "Raport"
    author: str = ""
    company: str = ""
    kpi: Dict[str, Any] = None  # {"KPI 1": 123, ...}
    notes: str = ""             # Markdown lub HTML
    theme: str = "light"        # "light" | "dark"
    logo_data_uri: Optional[str] = None  # np. output img_data_uri(open(...).read())
    tables: Dict[str, pd.DataFrame] = None  # {"Sekcja": df}
    images: Dict[str, str] = None  # {"Tytuł": data_uri_png}

    def __post_init__(self):
        self.kpi = self.kpi or {}
        self.tables = self.tables or {}
        self.images = self.images or {}

# === NAZWA_SEKCJI === THEME_CSS
CSS_LIGHT = """
:root{
  --bg:#ffffff; --text:#111; --muted:#667085; --card:#f9fafb; --border:#e5e7eb; --accent:#6C5CE7;
}
body { background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18mm 16mm; }
h1 { margin: 0 0 6px 0; font-size: 22px; }
h2 { margin: 18px 0 8px 0; font-size: 16px; }
.small { color: var(--muted); font-size: 12px; }
hr { border: 0; border-top: 1px solid var(--border); margin: 10px 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.kpi { border:1px solid var(--border); border-radius: 10px; padding: 10px; background: var(--card); }
.kpi .name { color: var(--muted); font-size: 12px; }
.kpi .val { font-size: 18px; font-weight: 800; }
.section { margin: 12px 0 8px 0; page-break-inside: avoid; }
.card { border:1px solid var(--border); border-radius: 10px; padding: 12px; background: var(--card); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 12px; text-align: left; }
thead th { background: #f3f4f6; }
img.plot { width: 100%; border:1px solid var(--border); border-radius: 10px; }
.header { display:flex; align-items:center; gap:10px; }
.logo { height: 28px; }
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-center { content: "Strona " counter(page) " / " counter(pages); color: #666; font-size: 11px; }
  @top-left { content: string(doc_title); color: #666; font-size: 11px; }
}
h1 { string-set: doc_title content(); }
"""

CSS_DARK = """
:root{
  --bg:#0b0f19; --text:#ffffff; --muted:#8b95a7; --card:#121826; --border:rgba(255,255,255,.12); --accent:#6C5CE7;
}
body { background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 18mm 16mm; }
h1 { margin: 0 0 6px 0; font-size: 22px; }
h2 { margin: 18px 0 8px 0; font-size: 16px; }
.small { color: var(--muted); font-size: 12px; }
hr { border: 0; border-top: 1px solid var(--border); margin: 10px 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.kpi { border:1px solid var(--border); border-radius: 10px; padding: 10px; background: var(--card); }
.kpi .name { color: var(--muted); font-size: 12px; }
.kpi .val { font-size: 18px; font-weight: 800; }
.section { margin: 12px 0 8px 0; page-break-inside: avoid; }
.card { border:1px solid var(--border); border-radius: 10px; padding: 12px; background: var(--card); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 12px; text-align: left; }
thead th { background: #1a2234; }
img.plot { width: 100%; border:1px solid var(--border); border-radius: 10px; }
.header { display:flex; align-items:center; gap:10px; }
.logo { height: 28px; }
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
  @bottom-center { content: "Strona " counter(page) " / " counter(pages); color: #9aa3b2; font-size: 11px; }
  @top-left { content: string(doc_title); color: #9aa3b2; font-size: 11px; }
}
h1 { string-set: doc_title content(); }
"""

# === NAZWA_SEKCJI === HELPERS
def _escape(s: str) -> str:
    return _html.escape(s, quote=True)

def _notes_to_html(notes: str) -> str:
    """Preferuj Markdown→HTML; gdy brak, przepuść HTML (sanityzuj bleach jeśli jest), fallback: escapuj."""
    if not notes:
        return ""
    if _md is not None:
        try:
            return _md.markdown(notes, extensions=["tables", "fenced_code", "sane_lists"])
        except Exception:
            pass
    # jeśli to już HTML i mamy bleach – sanityzuj
    if "<" in notes and bleach is not None:
        return bleach.clean(notes, tags=[
            "p","b","i","strong","em","ul","ol","li","br","hr","table","thead","tbody","tr","th","td","code","pre","h1","h2","h3","h4","h5","h6","span","div","img","a"
        ], attributes={"*":["class","style"], "a":["href","title","target","rel"], "img":["src","alt","title"]}, strip=True)
    # fallback: plain text
    return _escape(notes).replace("\n", "<br/>")

def kpi_html(kpis: Dict[str, Any]) -> str:
    items = []
    for k, v in kpis.items():
        items.append(f"""
          <div class="kpi">
            <div class="name">{_escape(str(k))}</div>
            <div class="val">{_escape(str(v))}</div>
          </div>
        """)
    return '<div class="kpi-grid">' + "".join(items) + '</div>'

def table_html(df: pd.DataFrame, max_rows: int = 200, max_cols: int = 50) -> str:
    if df is None or df.empty:
        return '<div class="small">Brak danych.</div>'
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    # schludny HTML z klasami
    return df2.to_html(classes="table", border=0, index=False)

def img_data_uri(png_bytes: bytes) -> str:
    """Zwraca data URI dla obrazka PNG."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def fig_to_data_uri(fig, *, width: int = 1100, height: int = 600, scale: float = 2.0) -> Optional[str]:
    """Plotly Figure -> PNG data URI (wymaga kaleido)."""
    try:
        png: bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return img_data_uri(png)
    except Exception:
        return None  # brak kaleido lub inny błąd

# === NAZWA_SEKCJI === TEMPLATE
BASE_TMPL = """
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
  <div class="header">
    {logo_html}
    <div>
      <h1>{title}</h1>
      <div class="small"><b>Autor:</b> {author} &nbsp;&nbsp; <b>Firma:</b> {company}</div>
    </div>
  </div>

  <div class="section">
    <h2>KPI</h2>
    {kpi_cards}
  </div>

  {images_block}

  {tables_block}

  <div class="section">
    <h2>Notatki</h2>
    <div class="card">{notes_html}</div>
  </div>

</body>
</html>
"""

# === NAZWA_SEKCJI === RENDERERS
def _render_images(images: Dict[str, str]) -> str:
    if not images:
        return ""
    blocks = []
    for title, data_uri in images.items():
        blocks.append(f"""
        <div class="section">
          <h2>{_escape(str(title))}</h2>
          <img class="plot" src="{data_uri}" />
        </div>
        """)
    return "\n".join(blocks)

def _render_tables(tables: Dict[str, pd.DataFrame]) -> str:
    if not tables:
        return ""
    blocks = []
    for title, df in tables.items():
        blocks.append(f"""
        <div class="section">
          <h2>{_escape(str(title))}</h2>
          <div class="card">
            {table_html(df)}
          </div>
        </div>
        """)
    return "\n".join(blocks)

def build_html(context: ReportContext) -> str:
    """Składa kompletny HTML raportu (samowystarczalny)."""
    css = CSS_DARK if str(context.theme).lower().startswith("d") else CSS_LIGHT
    kpi_cards = kpi_html(context.kpi or {})
    notes_html = _notes_to_html(context.notes or "")
    images_block = _render_images(context.images or {})
    tables_block = _render_tables(context.tables or {})

    logo_html = f'<img class="logo" src="{context.logo_data_uri}"/>' if context.logo_data_uri else ""

    html = BASE_TMPL.format(
        title=_escape(context.title or "Raport"),
        author=_escape(context.author or ""),
        company=_escape(context.company or ""),
        kpi_cards=kpi_cards,
        notes_html=notes_html,
        images_block=images_block,
        tables_block=tables_block,
        css=css,
        logo_html=logo_html,
    )
    return html

# === NAZWA_SEKCJI === PUBLIC API
def build_pdf(path_out: str, context: Dict[str, Any]) -> str:
    """
    Drop-in kompatybilne z Twoją wersją:
      - przyjmuje dict `context` (title, author, company, kpi, notes, theme, logo_data_uri, tables, images)
      - renderuje HTML i zapisuje PDF do `path_out`.
      - zwraca `path_out`.
    """
    ctx = ReportContext(**context)
    html = build_html(ctx)
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(path_out)
    return path_out

def build_pdf_pro(
    path_out: str,
    *,
    title: str,
    author: str = "",
    company: str = "",
    kpi: Dict[str, Any] | None = None,
    notes: str = "",
    theme: str = "dark",
    logo_png_bytes: Optional[bytes] = None,
    tables: Dict[str, pd.DataFrame] | None = None,
    images: Dict[str, str] | None = None,
) -> str:
    """Wygodny wrapper, gdy wołasz parametrami zamiast dict."""
    ctx = ReportContext(
        title=title, author=author, company=company,
        kpi=kpi or {}, notes=notes, theme=theme,
        logo_data_uri=(img_data_uri(logo_png_bytes) if logo_png_bytes else None),
        tables=tables or {}, images=images or {},
    )
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    HTML(string=build_html(ctx)).write_pdf(path_out)
    return path_out
