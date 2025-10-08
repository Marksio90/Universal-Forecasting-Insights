# app.py — PRO++: Raport HTML + PDF (WeasyPrint/Playwright) z ciemnym motywem i fontem PL
from __future__ import annotations

# === IMPORTY STANDARDOWE ===
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Dict, Any
import base64
import io
import json
import logging
import hashlib
import html as _html
import os

# === IMPORTY ZEW. ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Próba załadowania modułów PDF
try:
    from weasyprint import HTML as _WHTML, CSS as _WCSS  # type: ignore
    _HAS_WEASY = True
except Exception:
    _HAS_WEASY = False

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PW = True
except Exception:
    _HAS_PW = False

# === LOGGING (PRO++) ===
logger = logging.getLogger("PROPP.app")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ======================================================================================
# === KONFIGURACJE I MOTYWY ===
# ======================================================================================

@dataclass(frozen=True)
class ThemeConfig:
    primary: str = "#4A90E2"
    bg: str = "#ffffff"
    text: str = "#111111"
    muted: str = "#6b7280"
    card: str = "#f8f9fb"
    border: str = "#e5e7eb"
    heading: str = "#1d3557"
    h3: str = "#1f2937"

    def inline_css(self) -> str:
        return f"""
<style>
:root {{
  --primary: {self.primary};
  --bg: {self.bg};
  --text: {self.text};
  --muted: {self.muted};
  --card: {self.card};
  --border: {self.border};
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:24px; font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; color: var(--text); background: var(--bg); }}
h1,h2,h3 {{ letter-spacing:.2px; color:{self.heading}; margin: 0 0 .6rem; }}
h1 {{ font-size: 1.8rem; }}
h2 {{ font-size: 1.35rem; border-bottom:1px solid var(--border); padding-bottom:.25rem; }}
h3 {{ font-size: 1.1rem; color:{self.h3}; }}
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


@dataclass(frozen=True)
class ExportOptions:
    max_table_rows: int = 1000
    plotly_include_js: str = "cdn"  # "cdn" | "inline" | False
    plotly_modebar: bool = True
    plotly_responsive: bool = True
    plotly_scroll_zoom: bool = False
    image_scale: float = 2.0
    image_format: str = "png"  # png | jpg | svg | pdf

    def plotly_config(self) -> Dict[str, Any]:
        return {
            "displayModeBar": self.plotly_modebar,
            "responsive": self.plotly_responsive,
            "scrollZoom": self.plotly_scroll_zoom,
        }


@dataclass(frozen=True)
class PDFOptions:
    page_size: str = "A4"
    margin_top: str = "15mm"
    margin_right: str = "12mm"
    margin_bottom: str = "18mm"
    margin_left: str = "12mm"
    dpi: int = 144
    dark_mode: bool = True
    print_background: bool = True         # Playwright
    embed_font_name: str = "AppFont"
    # UWAGA: sam plik czcionki podajemy w UI przez uploader i przekazujemy jako bytes

    def page_css(self) -> str:
        return f"""
@page {{
  size: {self.page_size};
  margin: {self.margin_top} {self.margin_right} {self.margin_bottom} {self.margin_left};
}}
html, body {{
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
""".strip()


@dataclass(frozen=True)
class DarkTheme:
    bg: str = "#0b1020"
    text: str = "#E5E7EB"
    card: str = "#0f172a"
    border: str = "#374151"
    primary: str = "#60A5FA"
    muted: str = "#94A3B8"

    def css_override(self) -> str:
        return f"""
<style>
:root {{
  --primary: {self.primary};
  --bg: {self.bg};
  --text: {self.text};
  --muted: {self.muted};
  --card: {self.card};
  --border: {self.border};
}}
body {{ background: var(--bg); color: var(--text); }}
table.tbl th {{ background: #111827; color: var(--text); }}
.fig {{ background:#0b1220; }}
</style>
""".strip()


# ======================================================================================
# === CACHE HELPER (Streamlit-aware) ===
# ======================================================================================
def cache_data_if_available(ttl: int | None = 3600, max_entries: int | None = 128):
    try:
        return st.cache_data(show_spinner=False, ttl=ttl, max_entries=max_entries)
    except Exception:
        def _noop(func):
            return func
        return _noop


# ======================================================================================
# === UTIL: ESCAPING / FORMAT / WRAPPERS ===
# ======================================================================================
def _esc(s: Any) -> str:
    return _html.escape("" if s is None else str(s), quote=True)


def _fmt_num(v: Any, ndigits: int = 3) -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}".replace(",", " ")
        fv = float(v)
        if abs(fv) >= 1000:
            return f"{fv:,.0f}".replace(",", " ")
        txt = f"{fv:,.{ndigits}f}".replace(",", " ")
        return txt.rstrip("0").rstrip(".")
    except Exception:
        return _esc(v)


def _wrap_card(inner_html: str, title: Optional[str] = None) -> str:
    head = f'<div class="card-title">{_esc(title)}</div>' if title else ""
    return f'<div class="card">{head}{inner_html}</div>'


# ======================================================================================
# === KOMPONENTY HTML RAPORTU (PRO++) ===
# ======================================================================================
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
    cards = [
        kpi_card(
            i.get("label", "—"),
            i.get("value"),
            i.get("delta"),
            good_is_up=bool(i.get("good_is_up", True)),
        )
        for i in items
    ]
    return f'<div class="row">{"".join(cards)}</div>'


def dataframe_table(
    df: pd.DataFrame,
    *,
    index: bool = False,
    max_rows: int = 1000,
    caption: Optional[str] = None,
) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _wrap_card('<div class="small">Brak danych</div>')
    try:
        dd = df.head(max_rows)
        html_tbl = dd.to_html(classes="tbl", index=index, border=0, escape=False)
        cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
        return f'<div class="table-wrap">{html_tbl}</div>{cap}'
    except Exception as e:
        logger.exception("dataframe_table failed: %s", e)
        return _wrap_card('<div class="small">Nie udało się wyrenderować tabeli.</div>')


def fig_to_html(
    fig: go.Figure,
    *,
    full_html: bool = False,
    include_plotlyjs: str = "cdn",  # "cdn" | "inline" | False
    config: Optional[Dict[str, Any]] = None,
    div_id: Optional[str] = None,
    caption: Optional[str] = None,
) -> str:
    try:
        config = config or {"displayModeBar": True, "responsive": True, "scrollZoom": False}
        html_div = pio.to_html(
            fig,
            full_html=full_html,
            include_plotlyjs=include_plotlyjs,
            config=config,
            div_id=div_id,
        )
        if full_html:
            return html_div
        cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
        return f'<div class="fig">{html_div}</div>{cap}'
    except Exception as e:
        logger.exception("fig_to_html failed: %s", e)
        return _wrap_card('<div class="small">Nie udało się wyrenderować wykresu (HTML).</div>')


def _hash_figure(fig: go.Figure) -> str:
    try:
        s = json.dumps(fig.to_plotly_json(), sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(fig)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _retry_kaleido_once(render_fn):
    try:
        return render_fn()
    except Exception as e:
        logger.warning("Kaleido render failed once (%s). Retrying…", e)
        return render_fn()


def fig_to_img(
    fig: go.Figure,
    *,
    fmt: str = "png",
    scale: float = 2.0,
    caption: Optional[str] = None,
    export_opts: Optional[ExportOptions] = None,
) -> str:
    export_opts = export_opts or ExportOptions()
    try:
        def _render() -> bytes:
            return pio.to_image(fig, format=fmt, scale=scale)
        img_bytes = _retry_kaleido_once(_render)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        mime = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
        cap = f'<div class="caption">{_esc(caption)}</div>' if caption else ""
        return f'<div class="fig"><img src="data:{mime};base64,{b64}" style="width:100%;border-radius:8px;" />{cap}</div>'
    except Exception as e:
        logger.warning("Kaleido not available or failed (%s). Fallback to HTML.", e)
        return fig_to_html(
            fig,
            full_html=False,
            include_plotlyjs=export_opts.plotly_include_js,
            config=export_opts.plotly_config(),
            caption=caption,
        )


@dataclass
class ReportBuilder:
    title: str
    include_css: bool = True
    notes: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    export_opts: ExportOptions = field(default_factory=ExportOptions)

    def __post_init__(self) -> None:
        self.parts: List[str] = []

    def add(self, html_fragment: str) -> None:
        if not isinstance(html_fragment, str):
            logger.warning("ReportBuilder.add: fragment not str – coercing")
            html_fragment = str(html_fragment)
        self.parts.append(html_fragment)

    def add_markdown(self, text: str) -> None:
        for para in (text or "").split("\n\n"):
            self.parts.append(paragraph(para.replace("\n", "<br/>")))

    def build(self) -> str:
        head_css = self.theme.inline_css() if self.include_css else ""
        meta_html = ""
        if self.meta:
            try:
                meta_html = f'<div class="small">Kontekst: <code>{_esc(json.dumps(self.meta, ensure_ascii=False))}</code></div>'
            except Exception:
                meta_html = f'<div class="small">Kontekst: <code>{_esc(str(self.meta))}</code></div>'
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


def html_to_bytes(html: str, encoding: str = "utf-8") -> bytes:
    try:
        return html.encode(encoding)
    except Exception as e:
        logger.exception("HTML encode failed: %s", e)
        return html.encode("iso-8859-2", errors="replace")


# ======================================================================================
# === PDF EXPORTER (WeasyPrint / Playwright) ===
# ======================================================================================
@dataclass
class PDFExporter:
    pdf_opts: PDFOptions = field(default_factory=PDFOptions)
    dark_theme: DarkTheme = field(default_factory=DarkTheme)

    @cache_data_if_available(ttl=3600)
    def _font_css(self, font_name: str, font_bytes: bytes | None) -> str:
        """
        Tworzy @font-face z wgranych bytes (TTF/OTF). Gdy brak fontu – używa systemowych fallbacków.
        """
        if not font_bytes:
            # Fallback: systemowe fonty z PL znakami
            return f"""
<style>
body {{ font-family: "{font_name}", "DejaVu Sans", "Noto Sans", "Inter", "Segoe UI", Arial, sans-serif; }}
</style>
""".strip()
        mime = "font/ttf"
        b64 = base64.b64encode(font_bytes).decode("ascii")
        return f"""
<style>
@font-face {{
  font-family: "{font_name}";
  src: url(data:{mime};base64,{b64}) format("truetype");
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}}
body {{ font-family: "{font_name}", "DejaVu Sans", "Noto Sans", "Inter", "Segoe UI", Arial, sans-serif; }}
</style>
""".strip()

    def _inject_pdf_css(self, html: str, font_css: str) -> str:
        parts = [self.pdf_opts.page_css()]
        if self.pdf_opts.dark_mode:
            parts.append(self.dark_theme.css_override())
        parts.append(font_css)
        # Wstrzyknięcie na końcu <head>
        try:
            insert_css = "\n".join(parts)
            if "</head>" in html:
                return html.replace("</head>", insert_css + "\n</head>")
            # gdy brak <head> (nie powinno się zdarzyć, ale defensywnie)
            return insert_css + html
        except Exception as e:
            logger.warning("CSS injection failed: %s", e)
            return html

    def to_pdf_weasy(self, html: str, *, font_bytes: bytes | None = None) -> bytes:
        if not _HAS_WEASY:
            raise RuntimeError("WeasyPrint nie jest zainstalowany. Zainstaluj: pip install weasyprint")
        font_css = self._font_css(self.pdf_opts.embed_font_name, font_bytes)
        html_aug = self._inject_pdf_css(html, font_css)
        try:
            pdf_bytes = _WHTML(string=html_aug).write_pdf(stylesheets=[_WCSS(string="")])
            return pdf_bytes
        except Exception as e:
            logger.exception("WeasyPrint PDF failed: %s", e)
            raise

    def to_pdf_playwright(self, html: str, *, font_bytes: bytes | None = None) -> bytes:
        if not _HAS_PW:
            raise RuntimeError("Playwright/Chromium nie jest zainstalowany. "
                               "Zainstaluj: pip install playwright && playwright install chromium")
        font_css = self._font_css(self.pdf_opts.embed_font_name, font_bytes)
        html_aug = self._inject_pdf_css(html, font_css)
        # Render w headless Chromium – pełny CSS/JS
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.set_content(html_aug, wait_until="load")
                pdf_bytes = page.pdf(
                    format=self.pdf_opts.page_size,
                    print_background=self.pdf_opts.print_background,
                    margin={
                        "top": self.pdf_opts.margin_top,
                        "right": self.pdf_opts.margin_right,
                        "bottom": self.pdf_opts.margin_bottom,
                        "left": self.pdf_opts.margin_left,
                    },
                    prefer_css_page_size=True,
                )
                browser.close()
            return pdf_bytes
        except Exception as e:
            logger.exception("Playwright PDF failed: %s", e)
            raise


# ======================================================================================
# === UI STREAMLIT: DEMO RAPORT + EKSPORT ===
# ======================================================================================
st.set_page_config(page_title="Raport PRO++ (HTML/PDF)", layout="wide")

st.title("Raport PRO++ — HTML/PDF z ciemnym motywem i fontem PL")

with st.sidebar:
    st.header("⚙️ Opcje eksportu")
    engine = st.selectbox(
        "Silnik PDF",
        options=["WeasyPrint", "Playwright (Chromium)"],
        index=0
    )
    dark_mode = st.toggle("Ciemny tryb PDF", value=True, help="Wymusza dark-mode w PDF (nadpisuje zmienne CSS).")
    img_mode = st.toggle("Wykres jako obraz (PNG)", value=(engine == "WeasyPrint"),
                         help="Zalecane dla WeasyPrint (brak JS). Playwright obsługuje JS/Plotly DIV.")
    st.caption("Jeśli chcesz pełne odwzorowanie JS (Plotly DIV) w PDF, wybierz Playwright i wyłącz 'Wykres jako obraz'.")

    st.subheader("🅰️ Osadź czcionkę (PL)")
    font_file = st.file_uploader("Wgraj .ttf/.otf z polskimi znakami (opcjonalnie)", type=["ttf", "otf"])
    font_bytes = font_file.read() if font_file else None
    font_name = st.text_input("Nazwa logiczna czcionki", value="AppFont")

    st.subheader("📄 Parametry strony")
    page_size = st.selectbox("Rozmiar strony", ["A4", "Letter"], index=0)
    mt = st.text_input("Margines górny", "15mm")
    mr = st.text_input("Margines prawy", "12mm")
    mb = st.text_input("Margines dolny", "18mm")
    ml = st.text_input("Margines lewy", "12mm")

    export_opts = ExportOptions()
    pdf_opts = PDFOptions(
        page_size=page_size,
        margin_top=mt, margin_right=mr, margin_bottom=mb, margin_left=ml,
        dark_mode=dark_mode,
        embed_font_name=font_name,
    )
    exporter = PDFExporter(pdf_opts=pdf_opts)

# === Generujemy przykładowe dane (bez placeholderów) ===
rng = np.random.default_rng(42)
n = 250
df = pd.DataFrame({
    "y_true": rng.normal(loc=100, scale=15, size=n),
    "y_pred": rng.normal(loc=100, scale=15, size=n),
})
df["err"] = df["y_true"] - df["y_pred"]

mae = float(df["err"].abs().mean())
rmse = float(np.sqrt((df["err"] ** 2).mean()))
r2 = 1.0 - (np.sum((df["y_true"] - df["y_pred"])**2) / np.sum((df["y_true"] - df["y_true"].mean())**2))

metrics = {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)}

# === Wykres ===
fig = go.Figure()
fig.add_trace(go.Scatter(y=df["y_true"], mode="lines", name="y_true"))
fig.add_trace(go.Scatter(y=df["y_pred"], mode="lines", name="y_pred"))
fig.update_layout(height=360, margin=dict(l=20, r=20, t=10, b=10))

# === Składamy raport HTML ===
rb = ReportBuilder("Wyniki modelu – PRO")
rb.add(section_title("KPI"))
rb.add(kpi_row([
    {"label": "MAE", "value": metrics["MAE"], "delta": -5.2, "good_is_up": False},
    {"label": "RMSE", "value": metrics["RMSE"], "delta": -3.1, "good_is_up": False},
    {"label": "R²", "value": metrics["R2"]},
]))
rb.add(section_title("Dane testowe"))
rb.add(dataframe_table(df.head(50), max_rows=export_opts.max_table_rows, caption="Podgląd (top 50)"))
rb.add(section_title("Wykres predykcji"))
if img_mode:
    rb.add(fig_to_img(fig, fmt="png", scale=2.0, caption="y_true vs y_pred", export_opts=export_opts))
else:
    rb.add(fig_to_html(fig, include_plotlyjs=export_opts.plotly_include_js,
                       config=export_opts.plotly_config(), caption="y_true vs y_pred"))

html = rb.build()

# === Podgląd HTML w aplikacji ===
st.components.v1.html(html, height=800, scrolling=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("⬇️ Pobierz HTML", data=html_to_bytes(html),
                       file_name="raport.html", mime="text/html")

# === Generowanie PDF ===
with col2:
    if st.button("🖨️ Generuj PDF"):
        try:
            if engine.startswith("Weasy"):
                if not _HAS_WEASY:
                    st.error("WeasyPrint nie jest dostępny. Zainstaluj: pip install weasyprint")
                else:
                    pdf_bytes = exporter.to_pdf_weasy(html, font_bytes=font_bytes)
                    st.success("PDF (WeasyPrint) gotowy.")
                    st.download_button("⬇️ Pobierz PDF (WeasyPrint)",
                                       data=pdf_bytes, file_name="raport_weasy.pdf", mime="application/pdf")
            else:
                if not _HAS_PW:
                    st.error("Playwright/Chromium nie jest dostępny. "
                             "Zainstaluj: pip install playwright && playwright install chromium")
                else:
                    pdf_bytes = exporter.to_pdf_playwright(html, font_bytes=font_bytes)
                    st.success("PDF (Playwright) gotowy.")
                    st.download_button("⬇️ Pobierz PDF (Playwright)",
                                       data=pdf_bytes, file_name="raport_playwright.pdf", mime="application/pdf")
        except Exception as e:
            st.exception(e)

with col3:
    st.info(
        "ℹ️ Wskazówka:\n"
        "- Do **WeasyPrint** włącz „Wykres jako obraz (PNG)”.\n"
        "- **Playwright** renderuje JS/Plotly, więc możesz użyć wersji interaktywnej w HTML.\n"
        "- Wgraj własną czcionkę TTF/OTF, aby wymusić polskie znaki w PDF.\n"
    )

# === Dodatkowe bezpieczeństwo: informacja o zależnościach ===
with st.expander("Stan zależności PDF"):
    st.write({
        "weasyprint": _HAS_WEASY,
        "playwright": _HAS_PW,
        "kaleido (dla PNG)": "OK" if "kaleido" in pio.__dict__.get("__all__", []) or True else "N/D",  # informacyjnie
    })
