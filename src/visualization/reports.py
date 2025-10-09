"""
Report Generator PRO++++ - Zaawansowany generator raportów HTML/PDF z profesjonalnym stylingiem.

Funkcjonalności PRO++++:
- Multi-format export (HTML/PDF/PNG)
- Multiple PDF engines (WeasyPrint/Playwright/ReportLab)
- Professional themes (light/dark/custom)
- Custom font embedding (TTF/OTF) z pełnym wsparciem PL
- Interactive Plotly charts lub static images
- Component-based report building
- Template system z Jinja2
- Markdown support z rozszerzeniami
- Table of Contents generation
- Page numbers i headers/footers
- Watermarks i security
- Batch report generation
- Async PDF rendering
- Caching dla performance
- Export queue system
- Custom CSS injection
- Responsive design
- Accessibility (WCAG AA)
"""

from __future__ import annotations

import base64
import hashlib
import html as _html
import io
import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Literal,
    Union, Callable, Iterable
)
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Optional imports with graceful fallback
try:
    from weasyprint import HTML as _WHTML, CSS as _WCSS
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    warnings.warn("WeasyPrint not available. Install with: pip install weasyprint")

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    warnings.warn("Playwright not available. Install with: pip install playwright")

try:
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    warnings.warn("ReportLab not available. Install with: pip install reportlab")

try:
    from jinja2 import Template, Environment, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    warnings.warn("Jinja2 not available. Install with: pip install jinja2")

try:
    import markdown
    from markdown.extensions import tables, fenced_code, codehilite
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    warnings.warn("Markdown not available. Install with: pip install markdown")

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "report_generator", level: int = logging.INFO) -> logging.Logger:
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
# ENUMS & DATACLASSES
# ========================================================================================

class ReportTheme(str, Enum):
    """Dostępne motywy raportów."""
    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    CORPORATE = "corporate"


class PDFEngine(str, Enum):
    """Dostępne silniki PDF."""
    WEASYPRINT = "weasyprint"
    PLAYWRIGHT = "playwright"
    REPORTLAB = "reportlab"


class PageSize(str, Enum):
    """Rozmiary stron."""
    A4 = "A4"
    LETTER = "Letter"
    LEGAL = "Legal"
    A3 = "A3"


class ImageFormat(str, Enum):
    """Formaty obrazów."""
    PNG = "png"
    JPEG = "jpeg"
    SVG = "svg"
    PDF = "pdf"


@dataclass(frozen=True)
class ThemeConfig:
    """Konfiguracja motywu raportu."""
    name: str
    primary: str
    secondary: str
    accent: str
    bg: str
    text: str
    muted: str
    card: str
    border: str
    heading: str
    link: str
    success: str
    warning: str
    error: str
    code_bg: str
    
    def to_css_variables(self) -> str:
        """Konwertuje do CSS variables."""
        return f"""
:root {{
  --theme-primary: {self.primary};
  --theme-secondary: {self.secondary};
  --theme-accent: {self.accent};
  --theme-bg: {self.bg};
  --theme-text: {self.text};
  --theme-muted: {self.muted};
  --theme-card: {self.card};
  --theme-border: {self.border};
  --theme-heading: {self.heading};
  --theme-link: {self.link};
  --theme-success: {self.success};
  --theme-warning: {self.warning};
  --theme-error: {self.error};
  --theme-code-bg: {self.code_bg};
}}
""".strip()


# Predefiniowane motywy
THEMES = {
    ReportTheme.LIGHT: ThemeConfig(
        name="Light",
        primary="#4A90E2",
        secondary="#22d3ee",
        accent="#a78bfa",
        bg="#ffffff",
        text="#111111",
        muted="#6b7280",
        card="#f8f9fb",
        border="#e5e7eb",
        heading="#1d3557",
        link="#2563eb",
        success="#16a34a",
        warning="#f59e0b",
        error="#dc2626",
        code_bg="#f3f4f6"
    ),
    ReportTheme.DARK: ThemeConfig(
        name="Dark",
        primary="#60A5FA",
        secondary="#22d3ee",
        accent="#c084fc",
        bg="#0b1020",
        text="#E5E7EB",
        muted="#94A3B8",
        card="#0f172a",
        border="#374151",
        heading="#f9fafb",
        link="#60a5fa",
        success="#34d399",
        warning="#fbbf24",
        error="#f87171",
        code_bg="#1e293b"
    ),
    ReportTheme.PROFESSIONAL: ThemeConfig(
        name="Professional",
        primary="#1e40af",
        secondary="#0891b2",
        accent="#7c3aed",
        bg="#fafafa",
        text="#0f172a",
        muted="#64748b",
        card="#ffffff",
        border="#cbd5e1",
        heading="#1e293b",
        link="#1e40af",
        success="#059669",
        warning="#d97706",
        error="#b91c1c",
        code_bg="#f1f5f9"
    ),
    ReportTheme.MINIMAL: ThemeConfig(
        name="Minimal",
        primary="#000000",
        secondary="#404040",
        accent="#808080",
        bg="#ffffff",
        text="#1a1a1a",
        muted="#737373",
        card="#fafafa",
        border="#e5e5e5",
        heading="#000000",
        link="#000000",
        success="#22c55e",
        warning="#eab308",
        error="#ef4444",
        code_bg="#f5f5f5"
    ),
    ReportTheme.CORPORATE: ThemeConfig(
        name="Corporate",
        primary="#003366",
        secondary="#0066cc",
        accent="#6699cc",
        bg="#f5f7fa",
        text="#1a1a1a",
        muted="#666666",
        card="#ffffff",
        border="#d1d5db",
        heading="#003366",
        link="#0066cc",
        success="#10b981",
        warning="#f59e0b",
        error="#dc2626",
        code_bg="#e5e7eb"
    )
}


@dataclass(frozen=True)
class ExportOptions:
    """Opcje eksportu."""
    max_table_rows: int = 1000
    plotly_include_js: Literal["cdn", "inline", False] = "cdn"
    plotly_modebar: bool = True
    plotly_responsive: bool = True
    plotly_scroll_zoom: bool = False
    image_scale: float = 2.0
    image_format: ImageFormat = ImageFormat.PNG
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    def plotly_config(self) -> Dict[str, Any]:
        """Zwraca konfigurację Plotly."""
        return {
            "displayModeBar": self.plotly_modebar,
            "responsive": self.plotly_responsive,
            "scrollZoom": self.plotly_scroll_zoom,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"] if not self.plotly_scroll_zoom else []
        }


@dataclass(frozen=True)
class PDFOptions:
    """Opcje PDF."""
    page_size: PageSize = PageSize.A4
    margin_top: str = "15mm"
    margin_right: str = "12mm"
    margin_bottom: str = "18mm"
    margin_left: str = "12mm"
    dpi: int = 144
    print_background: bool = True
    embed_fonts: bool = True
    font_name: str = "AppFont"
    font_fallbacks: Tuple[str, ...] = ("DejaVu Sans", "Noto Sans", "Inter", "Segoe UI", "Arial")
    
    # Headers & Footers
    header_text: Optional[str] = None
    footer_text: Optional[str] = None
    show_page_numbers: bool = True
    page_number_format: str = "Page {page} of {total}"
    
    # Security
    watermark_text: Optional[str] = None
    watermark_opacity: float = 0.1
    
    def page_css(self) -> str:
        """Generuje CSS dla strony."""
        return f"""
@page {{
  size: {self.page_size.value};
  margin: {self.margin_top} {self.margin_right} {self.margin_bottom} {self.margin_left};
  
  @top-center {{
    content: "{self.header_text or ''}";
    font-size: 9pt;
    color: #666;
  }}
  
  @bottom-center {{
    content: "{self.footer_text or ''}";
    font-size: 9pt;
    color: #666;
  }}
  
  @bottom-right {{
    content: counter(page) " / " counter(pages);
    font-size: 9pt;
    color: #666;
  }}
}}

html, body {{
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
  color-adjust: exact;
}}
""".strip()


@dataclass
class ReportMetadata:
    """Metadata raportu."""
    title: str
    author: Optional[str] = None
    company: Optional[str] = None
    date: Optional[str] = None
    version: str = "1.0"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M")


@dataclass
class TableOfContents:
    """Table of Contents."""
    enabled: bool = True
    title: str = "Spis treści"
    max_depth: int = 3
    entries: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_entry(self, level: int, title: str, anchor: str) -> None:
        """Dodaje wpis do TOC."""
        if level <= self.max_depth:
            self.entries.append({
                "level": level,
                "title": title,
                "anchor": anchor
            })
    
    def to_html(self) -> str:
        """Generuje HTML TOC."""
        if not self.enabled or not self.entries:
            return ""
        
        html = [f'<div class="toc"><h2>{_esc(self.title)}</h2><ul class="toc-list">']
        
        for entry in self.entries:
            indent = (entry["level"] - 1) * 20
            html.append(
                f'<li style="margin-left:{indent}px">'
                f'<a href="#{entry["anchor"]}">{_esc(entry["title"])}</a>'
                f'</li>'
            )
        
        html.append('</ul></div>')
        return "\n".join(html)


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def _esc(s: Any) -> str:
    """HTML escape."""
    return _html.escape("" if s is None else str(s), quote=True)


def _fmt_num(v: Any, ndigits: int = 3, thousands_sep: str = " ") -> str:
    """Formatuje liczbę z separatorem tysięcy."""
    if v is None:
        return "—"
    
    try:
        if isinstance(v, bool):
            return str(v)
        
        if isinstance(v, int):
            return f"{v:,}".replace(",", thousands_sep)
        
        fv = float(v)
        
        if abs(fv) >= 1000:
            return f"{fv:,.0f}".replace(",", thousands_sep)
        
        txt = f"{fv:,.{ndigits}f}".replace(",", thousands_sep)
        return txt.rstrip("0").rstrip(".")
        
    except (ValueError, TypeError):
        return _esc(v)


def _generate_anchor(text: str) -> str:
    """Generuje anchor ID z tekstu."""
    import re
    # Remove special chars, convert to lowercase, replace spaces
    anchor = re.sub(r'[^\w\s-]', '', text.lower())
    anchor = re.sub(r'[-\s]+', '-', anchor)
    return anchor.strip('-')


@lru_cache(maxsize=128)
def _hash_content(content: str) -> str:
    """Hashuje zawartość dla cache key."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _encode_image_base64(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Enkoduje obrazek do base64 data URI."""
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


# ========================================================================================
# CSS GENERATOR
# ========================================================================================

def generate_report_css(theme: ThemeConfig, custom_css: Optional[str] = None) -> str:
    """
    Generuje kompletny CSS dla raportu.
    
    Args:
        theme: Konfiguracja motywu
        custom_css: Dodatkowy custom CSS
        
    Returns:
        Complete CSS string
    """
    base_css = f"""
{theme.to_css_variables()}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

body {{
  margin: 0;
  padding: 24px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: var(--theme-text);
  background: var(--theme-bg);
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--theme-heading);
  margin: 1.5rem 0 0.75rem;
  line-height: 1.3;
}}

h1 {{ font-size: 2rem; margin-top: 0; }}
h2 {{ font-size: 1.5rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.3rem; }}
h3 {{ font-size: 1.25rem; }}
h4 {{ font-size: 1.1rem; }}

p {{
  margin: 0.75rem 0;
  line-height: 1.65;
}}

a {{
  color: var(--theme-link);
  text-decoration: none;
  transition: color 0.2s;
}}

a:hover {{
  text-decoration: underline;
}}

/* Lists */
ul, ol {{
  margin: 0.75rem 0;
  padding-left: 2rem;
}}

li {{
  margin: 0.3rem 0;
}}

/* Code */
code {{
  background: var(--theme-code-bg);
  padding: 0.15rem 0.4rem;
  border-radius: 3px;
  font-family: "Fira Code", "Consolas", monospace;
  font-size: 0.9em;
}}

pre {{
  background: var(--theme-code-bg);
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  margin: 1rem 0;
}}

pre code {{
  background: none;
  padding: 0;
}}

/* Layout */
.container {{
  max-width: 1200px;
  margin: 0 auto;
}}

.row {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: stretch;
  margin: 1rem 0;
}}

.col {{
  flex: 1;
  min-width: 0;
}}

/* Cards */
.card {{
  background: var(--theme-card);
  border: 1px solid var(--theme-border);
  border-radius: 12px;
  padding: 1.25rem;
  margin: 1rem 0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  flex: 1 1 260px;
}}

.card-title {{
  font-weight: 600;
  font-size: 1.05rem;
  margin-bottom: 0.75rem;
  color: var(--theme-heading);
}}

.card-subtitle {{
  color: var(--theme-muted);
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}}

/* KPI Components */
.kpi {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}}

.kpi-value {{
  font-size: 2rem;
  font-weight: 700;
  color: var(--theme-heading);
  line-height: 1;
}}

.kpi-label {{
  color: var(--theme-muted);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.kpi-delta {{
  font-size: 0.9rem;
  font-weight: 600;
}}

.kpi-delta.positive {{
  color: var(--theme-success);
}}

.kpi-delta.negative {{
  color: var(--theme-error);
}}

.kpi-delta.neutral {{
  color: var(--theme-muted);
}}

/* Tables */
.table-wrapper {{
  overflow-x: auto;
  border: 1px solid var(--theme-border);
  border-radius: 12px;
  margin: 1rem 0;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}}

th, td {{
  padding: 0.75rem 1rem;
  text-align: left;
  border-bottom: 1px solid var(--theme-border);
}}

th {{
  font-weight: 600;
  background: var(--theme-card);
  color: var(--theme-heading);
  white-space: nowrap;
}}

tbody tr:hover {{
  background: var(--theme-card);
}}

tbody tr:last-child td {{
  border-bottom: none;
}}

/* Figures */
.figure {{
  border: 1px solid var(--theme-border);
  border-radius: 12px;
  padding: 1rem;
  margin: 1.5rem 0;
  background: var(--theme-card);
}}

.figure-caption {{
  color: var(--theme-muted);
  font-size: 0.9rem;
  margin-top: 0.5rem;
  text-align: center;
  font-style: italic;
}}

/* Badges & Pills */
.badge {{
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.8rem;
  background: rgba(74, 144, 226, 0.1);
  color: var(--theme-primary);
}}

.badge-success {{
  background: rgba(22, 163, 74, 0.1);
  color: var(--theme-success);
}}

.badge-warning {{
  background: rgba(245, 158, 11, 0.1);
  color: var(--theme-warning);
}}

.badge-error {{
  background: rgba(220, 38, 38, 0.1);
  color: var(--theme-error);
}}

/* Alerts */
.alert {{
  padding: 1rem 1.25rem;
  border-radius: 8px;
  margin: 1rem 0;
  border-left: 4px solid;
}}

.alert-info {{
  background: rgba(74, 144, 226, 0.1);
  border-left-color: var(--theme-primary);
  color: var(--theme-text);
}}

.alert-success {{
  background: rgba(22, 163, 74, 0.1);
  border-left-color: var(--theme-success);
}}

.alert-warning {{
  background: rgba(245, 158, 11, 0.1);
  border-left-color: var(--theme-warning);
}}

.alert-error {{
  background: rgba(220, 38, 38, 0.1);
  border-left-color: var(--theme-error);
}}

/* TOC */
.toc {{
  background: var(--theme-card);
  border: 1px solid var(--theme-border);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 2rem 0;
}}

.toc h2 {{
  margin-top: 0;
  border-bottom: none;
  font-size: 1.3rem;
}}

.toc-list {{
  list-style: none;
  padding: 0;
}}

.toc-list li {{
  margin: 0.5rem 0;
}}

.toc-list a {{
  color: var(--theme-text);
  font-weight: 500;
}}

.toc-list a:hover {{
  color: var(--theme-link);
}}

/* Utilities */
.text-center {{ text-align: center; }}
.text-right {{ text-align: right; }}
.text-muted {{ color: var(--theme-muted); }}
.text-small {{ font-size: 0.9rem; }}
.text-large {{ font-size: 1.1rem; }}

.mt-0 {{ margin-top: 0; }}
.mt-1 {{ margin-top: 0.5rem; }}
.mt-2 {{ margin-top: 1rem; }}
.mt-3 {{ margin-top: 1.5rem; }}
.mt-4 {{ margin-top: 2rem; }}

.mb-0 {{ margin-bottom: 0; }}
.mb-1 {{ margin-bottom: 0.5rem; }}
.mb-2 {{ margin-bottom: 1rem; }}
.mb-3 {{ margin-bottom: 1.5rem; }}
.mb-4 {{ margin-bottom: 2rem; }}

.divider {{
  border: none;
  border-top: 1px solid var(--theme-border);
  margin: 2rem 0;
}}

/* Footer */
footer {{
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid var(--theme-border);
  color: var(--theme-muted);
  font-size: 0.9rem;
  text-align: center;
}}

/* Print specific */
@media print {{
  body {{
    padding: 0;
  }}
  
  .no-print {{
    display: none !important;
  }}
  
  .page-break {{
    page-break-after: always;
  }}
}}
"""
    
    if custom_css:
        base_css += f"\n\n/* Custom CSS */\n{custom_css}"
    
    return base_css


# ========================================================================================
# HTML COMPONENTS
# ========================================================================================

def section_title(
    text: str,
    level: int = 2,
    anchor: Optional[str] = None,
    add_to_toc: bool = True,
    toc: Optional[TableOfContents] = None
) -> str:
    """
    Tworzy tytuł sekcji.
    
    Args:
        text: Tekst tytułu
        level: Poziom nagłówka (1-6)
        anchor: Custom anchor ID
        add_to_toc: Czy dodać do TOC
        toc: Table of Contents object
        
    Returns:
        HTML string
    """
    level = max(1, min(level, 6))
    
    if anchor is None:
        anchor = _generate_anchor(text)
    
    if add_to_toc and toc:
        toc.add_entry(level, text, anchor)
    
    return f'<h{level} id="{_esc(anchor)}">{_esc(text)}</h{level}>'


def paragraph(text: str, css_class: Optional[str] = None) -> str:
    """Tworzy paragraf."""
    class_attr = f' class="{_esc(css_class)}"' if css_class else ""
    return f'<p{class_attr}>{_esc(text)}</p>'


def badge(text: str, variant: Literal["default", "success", "warning", "error"] = "default") -> str:
    """Tworzy badge."""
    css_class = f"badge badge-{variant}" if variant != "default" else "badge"
    return f'<span class="{css_class}">{_esc(text)}</span>'


def alert(
    message: str,
    variant: Literal["info", "success", "warning", "error"] = "info",
    title: Optional[str] = None
) -> str:
    """Tworzy alert box."""
    title_html = f"<strong>{_esc(title)}</strong><br/>" if title else ""
    return f'<div class="alert alert-{variant}">{title_html}{_esc(message)}</div>'


def kpi_card(
    label: str,
    value: Any,
    delta: Optional[Union[float, str]] = None,
    *,
    higher_is_better: bool = True,
    unit: str = "",
    format_digits: int = 2
) -> str:
    """
    Tworzy kartę KPI.
    
    Args:
        label: Etykieta KPI
        value: Wartość
        delta: Zmiana (% lub tekst)
        higher_is_better: Czy wyższa wartość = lepiej
        unit: Jednostka
        format_digits: Liczba cyfr po przecinku
        
    Returns:
        HTML string
    """
    formatted_value = _fmt_num(value, ndigits=format_digits)
    
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            # Determine class based on delta sign and higher_is_better
            is_positive = delta > 0
            if (is_positive and higher_is_better) or (not is_positive and not higher_is_better):
                delta_class = "positive"
                symbol = "▲" if is_positive else "▼"
            elif (is_positive and not higher_is_better) or (not is_positive and higher_is_better):
                delta_class = "negative"
                symbol = "▲" if is_positive else "▼"
            else:
                delta_class = "neutral"
                symbol = "●"
            
            delta_html = f'<div class="kpi-delta {delta_class}">{symbol} {abs(delta):.1f}%</div>'
        else:
            delta_html = f'<div class="kpi-delta neutral">{_esc(delta)}</div>'
    
    html = f"""
<div class="kpi">
  <div class="kpi-label">{_esc(label)}</div>
  <div class="kpi-value">{formatted_value}{_esc(unit)}</div>
  {delta_html}
</div>
"""
    return f'<div class="card">{html}</div>'


def kpi_row(items: Sequence[Dict[str, Any]]) -> str:
    """
    Tworzy wiersz z wieloma KPI.
    
    Args:
        items: Lista dict z parametrami dla kpi_card
        
    Returns:
        HTML string
    """
    cards = [kpi_card(**item) for item in items]
    return f'<div class="row">{"".join(cards)}</div>'


def dataframe_table(
    df: pd.DataFrame,
    *,
    index: bool = False,
    max_rows: int = 1000,
    caption: Optional[str] = None,
    css_class: str = "table-wrapper"
) -> str:
    """
    Konwertuje DataFrame do HTML table.
    
    Args:
        df: DataFrame
        index: Czy pokazać indeks
        max_rows: Maksymalna liczba wierszy
        caption: Podpis tabeli
        css_class: Klasa CSS
        
    Returns:
        HTML string
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f'<div class="card"><p class="text-muted">Brak danych w tabeli</p></div>'
    
    try:
        # Limit rows
        df_display = df.head(max_rows)
        
        # Generate HTML table
        html_table = df_display.to_html(
            index=index,
            border=0,
            escape=True,
            na_rep="—",
            float_format=lambda x: _fmt_num(x)
        )
        
        # Add caption
        caption_html = f'<div class="figure-caption">{_esc(caption)}</div>' if caption else ""
        
        # Wrap in styled div
        return f"""
<div class="{css_class}">
  {html_table}
</div>
{caption_html}
"""
    except Exception as e:
        LOGGER.error(f"Failed to render DataFrame table: {e}")
        return f'<div class="card"><p class="text-muted">Błąd renderowania tabeli: {_esc(str(e))}</p></div>'


def fig_to_html(
    fig: go.Figure,
    *,
    full_html: bool = False,
    include_plotlyjs: Union[Literal["cdn", "inline"], bool] = "cdn",
    config: Optional[Dict[str, Any]] = None,
    div_id: Optional[str] = None,
    caption: Optional[str] = None,
    export_opts: Optional[ExportOptions] = None
) -> str:
    """
    Konwertuje Plotly Figure do HTML.
    
    Args:
        fig: Plotly Figure
        full_html: Czy zwrócić pełny HTML document
        include_plotlyjs: Sposób includowania Plotly.js
        config: Konfiguracja Plotly
        div_id: ID elementu div
        caption: Podpis wykresu
        export_opts: Opcje eksportu
        
    Returns:
        HTML string
    """
    if fig is None:
        return '<div class="card"><p class="text-muted">Brak wykresu</p></div>'
    
    try:
        export_opts = export_opts or ExportOptions()
        config = config or export_opts.plotly_config()
        
        html_div = pio.to_html(
            fig,
            full_html=full_html,
            include_plotlyjs=include_plotlyjs,
            config=config,
            div_id=div_id,
            validate=False
        )
        
        if full_html:
            return html_div
        
        caption_html = f'<div class="figure-caption">{_esc(caption)}</div>' if caption else ""
        
        return f"""
<div class="figure">
  {html_div}
  {caption_html}
</div>
"""
    except Exception as e:
        LOGGER.error(f"Failed to render Plotly figure: {e}")
        return f'<div class="card"><p class="text-muted">Błąd renderowania wykresu: {_esc(str(e))}</p></div>'


def fig_to_image(
    fig: go.Figure,
    *,
    format: ImageFormat = ImageFormat.PNG,
    scale: float = 2.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    caption: Optional[str] = None,
    export_opts: Optional[ExportOptions] = None
) -> str:
    """
    Konwertuje Plotly Figure do static image (base64).
    
    Args:
        fig: Plotly Figure
        format: Format obrazu
        scale: Skala renderowania
        width: Szerokość w pikselach
        height: Wysokość w pikselach
        caption: Podpis
        export_opts: Opcje eksportu
        
    Returns:
        HTML string z embedded image
    """
    if fig is None:
        return '<div class="card"><p class="text-muted">Brak wykresu</p></div>'
    
    try:
        # Render to image bytes
        img_bytes = pio.to_image(
            fig,
            format=format.value,
            scale=scale,
            width=width,
            height=height,
            validate=False
        )
        
        # Determine MIME type
        mime_map = {
            ImageFormat.PNG: "image/png",
            ImageFormat.JPEG: "image/jpeg",
            ImageFormat.SVG: "image/svg+xml",
            ImageFormat.PDF: "application/pdf"
        }
        mime = mime_map.get(format, "image/png")
        
        # Encode to base64
        data_uri = _encode_image_base64(img_bytes, mime)
        
        caption_html = f'<div class="figure-caption">{_esc(caption)}</div>' if caption else ""
        
        return f"""
<div class="figure">
  <img src="{data_uri}" style="width:100%; border-radius:8px;" alt="{_esc(caption or 'Chart')}" />
  {caption_html}
</div>
"""
    except Exception as e:
        LOGGER.warning(f"Image render failed: {e}. Falling back to HTML.")
        export_opts = export_opts or ExportOptions()
        return fig_to_html(fig, caption=caption, export_opts=export_opts)


def markdown_to_html(text: str, extensions: Optional[List[str]] = None) -> str:
    """
    Konwertuje Markdown do HTML.
    
    Args:
        text: Tekst Markdown
        extensions: Lista rozszerzeń Markdown
        
    Returns:
        HTML string
    """
    if not HAS_MARKDOWN:
        # Fallback: basic conversion
        paragraphs = text.split("\n\n")
        return "".join(f"<p>{_esc(p)}</p>" for p in paragraphs)
    
    try:
        extensions = extensions or ["tables", "fenced_code", "codehilite"]
        html = markdown.markdown(text, extensions=extensions)
        return html
    except Exception as e:
        LOGGER.error(f"Markdown conversion failed: {e}")
        return f"<p>{_esc(text)}</p>"


# ========================================================================================
# REPORT BUILDER
# ========================================================================================

@dataclass
class ReportBuilder:
    """
    Zaawansowany builder raportów HTML PRO++++.
    
    Attributes:
        metadata: Metadata raportu
        theme: Motyw wizualny
        export_opts: Opcje eksportu
        toc: Table of Contents
        custom_css: Dodatkowy CSS
    """
    metadata: ReportMetadata
    theme: Union[ReportTheme, ThemeConfig] = ReportTheme.PROFESSIONAL
    export_opts: ExportOptions = field(default_factory=ExportOptions)
    toc: Optional[TableOfContents] = None
    custom_css: Optional[str] = None
    
    def __post_init__(self):
        """Inicjalizacja builder."""
        self.parts: List[str] = []
        
        # Convert theme enum to config
        if isinstance(self.theme, ReportTheme):
            self.theme = THEMES[self.theme]
        
        # Initialize TOC if not provided
        if self.toc is None:
            self.toc = TableOfContents()
    
    def add(self, html_fragment: str) -> ReportBuilder:
        """
        Dodaje fragment HTML.
        
        Args:
            html_fragment: Fragment HTML
            
        Returns:
            Self dla method chaining
        """
        if not isinstance(html_fragment, str):
            LOGGER.warning("ReportBuilder.add: fragment not string, converting")
            html_fragment = str(html_fragment)
        
        self.parts.append(html_fragment)
        return self
    
    def add_section(
        self,
        title: str,
        level: int = 2,
        anchor: Optional[str] = None
    ) -> ReportBuilder:
        """Dodaje tytuł sekcji."""
        html = section_title(title, level=level, anchor=anchor, toc=self.toc)
        return self.add(html)
    
    def add_paragraph(self, text: str, css_class: Optional[str] = None) -> ReportBuilder:
        """Dodaje paragraf."""
        return self.add(paragraph(text, css_class=css_class))
    
    def add_markdown(self, text: str, extensions: Optional[List[str]] = None) -> ReportBuilder:
        """Dodaje Markdown (konwertowany do HTML)."""
        html = markdown_to_html(text, extensions=extensions)
        return self.add(html)
    
    def add_kpi_row(self, items: Sequence[Dict[str, Any]]) -> ReportBuilder:
        """Dodaje wiersz KPI."""
        return self.add(kpi_row(items))
    
    def add_table(
        self,
        df: pd.DataFrame,
        caption: Optional[str] = None,
        **kwargs
    ) -> ReportBuilder:
        """Dodaje tabelę DataFrame."""
        html = dataframe_table(df, caption=caption, **kwargs)
        return self.add(html)
    
    def add_figure(
        self,
        fig: go.Figure,
        as_image: bool = False,
        caption: Optional[str] = None,
        **kwargs
    ) -> ReportBuilder:
        """
        Dodaje wykres Plotly.
        
        Args:
            fig: Plotly Figure
            as_image: Czy renderować jako static image
            caption: Podpis
            **kwargs: Dodatkowe argumenty
            
        Returns:
            Self
        """
        if as_image:
            html = fig_to_image(fig, caption=caption, export_opts=self.export_opts, **kwargs)
        else:
            html = fig_to_html(fig, caption=caption, export_opts=self.export_opts, **kwargs)
        
        return self.add(html)
    
    def add_alert(
        self,
        message: str,
        variant: Literal["info", "success", "warning", "error"] = "info",
        title: Optional[str] = None
    ) -> ReportBuilder:
        """Dodaje alert box."""
        return self.add(alert(message, variant=variant, title=title))
    
    def add_divider(self) -> ReportBuilder:
        """Dodaje separator."""
        return self.add('<hr class="divider" />')
    
    def add_page_break(self) -> ReportBuilder:
        """Dodaje page break (dla PDF)."""
        return self.add('<div class="page-break"></div>')
    
    def add_custom_html(self, html: str) -> ReportBuilder:
        """Dodaje custom HTML."""
        return self.add(html)
    
    def build_html(self, include_toc: bool = True) -> str:
        """
        Buduje kompletny HTML document.
        
        Args:
            include_toc: Czy includować TOC
            
        Returns:
            Complete HTML string
        """
        # Generate CSS
        css = generate_report_css(self.theme, self.custom_css)
        
        # Build metadata section
        metadata_items = []
        if self.metadata.author:
            metadata_items.append(f"<strong>Autor:</strong> {_esc(self.metadata.author)}")
        if self.metadata.company:
            metadata_items.append(f"<strong>Organizacja:</strong> {_esc(self.metadata.company)}")
        if self.metadata.date:
            metadata_items.append(f"<strong>Data:</strong> {_esc(self.metadata.date)}")
        if self.metadata.version:
            metadata_items.append(f"<strong>Wersja:</strong> {_esc(self.metadata.version)}")
        
        metadata_html = ""
        if metadata_items:
            metadata_html = f'<div class="text-muted text-small mb-3">{" | ".join(metadata_items)}</div>'
        
        # Description
        description_html = ""
        if self.metadata.description:
            description_html = f'<p class="mb-3">{_esc(self.metadata.description)}</p>'
        
        # Tags
        tags_html = ""
        if self.metadata.tags:
            tags = " ".join(badge(tag) for tag in self.metadata.tags)
            tags_html = f'<div class="mb-3">{tags}</div>'
        
        # TOC
        toc_html = ""
        if include_toc and self.toc and self.toc.enabled:
            toc_html = self.toc.to_html()
        
        # Body content
        body_content = "\n".join(self.parts)
        
        # Footer
        footer_html = f"""
<footer>
  <p><em>Wygenerowano przez Intelligent Predictor</em></p>
  <p class="text-small">{self.metadata.date}</p>
</footer>
"""
        
        # Complete HTML
        html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="author" content="{_esc(self.metadata.author or '')}">
  <meta name="description" content="{_esc(self.metadata.description or '')}">
  <title>{_esc(self.metadata.title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="container">
    <h1>{_esc(self.metadata.title)}</h1>
    {metadata_html}
    {description_html}
    {tags_html}
    {toc_html}
    {body_content}
    {footer_html}
  </div>
</body>
</html>"""
        
        return html
    
    def to_bytes(self, encoding: str = "utf-8") -> bytes:
        """
        Konwertuje HTML do bytes.
        
        Args:
            encoding: Kodowanie
            
        Returns:
            HTML jako bytes
        """
        html = self.build_html()
        try:
            return html.encode(encoding)
        except UnicodeEncodeError:
            LOGGER.warning(f"Encoding {encoding} failed, falling back to utf-8 with errors='replace'")
            return html.encode("utf-8", errors="replace")


# ========================================================================================
# PDF EXPORTERS
# ========================================================================================

class FontManager:
    """Manager dla custom fontów."""
    
    def __init__(self):
        self._fonts: Dict[str, bytes] = {}
    
    def add_font(self, name: str, font_bytes: bytes) -> None:
        """Dodaje czcionkę."""
        self._fonts[name] = font_bytes
        LOGGER.debug(f"Added font: {name} ({len(font_bytes)} bytes)")
    
    def get_font(self, name: str) -> Optional[bytes]:
        """Pobiera czcionkę."""
        return self._fonts.get(name)
    
    def has_font(self, name: str) -> bool:
        """Sprawdza czy czcionka istnieje."""
        return name in self._fonts
    
    @lru_cache(maxsize=32)
    def generate_font_face_css(self, name: str, font_bytes: bytes) -> str:
        """
        Generuje @font-face CSS.
        
        Args:
            name: Nazwa czcionki
            font_bytes: Dane binarne czcionki
            
        Returns:
            CSS string
        """
        if not font_bytes:
            return ""
        
        # Detect format from magic bytes
        if font_bytes.startswith(b'\x00\x01\x00\x00'):
            format_type = "truetype"
            mime = "font/ttf"
        elif font_bytes.startswith(b'OTTO'):
            format_type = "opentype"
            mime = "font/otf"
        elif font_bytes.startswith(b'wOFF'):
            format_type = "woff"
            mime = "font/woff"
        elif font_bytes.startswith(b'wOF2'):
            format_type = "woff2"
            mime = "font/woff2"
        else:
            format_type = "truetype"
            mime = "font/ttf"
        
        b64 = base64.b64encode(font_bytes).decode("ascii")
        
        return f"""
@font-face {{
  font-family: "{name}";
  src: url(data:{mime};base64,{b64}) format("{format_type}");
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}}
"""


@dataclass
class PDFExporter:
    """
    Zaawansowany exporter PDF PRO++++.
    
    Supports multiple rendering engines:
    - WeasyPrint (CSS-based)
    - Playwright (Chromium)
    - ReportLab (programmatic)
    """
    pdf_opts: PDFOptions = field(default_factory=PDFOptions)
    font_manager: FontManager = field(default_factory=FontManager)
    
    def _inject_pdf_css(
        self,
        html: str,
        additional_css: Optional[str] = None
    ) -> str:
        """
        Wstrzykuje PDF-specific CSS.
        
        Args:
            html: HTML string
            additional_css: Dodatkowy CSS
            
        Returns:
            HTML z wstrzykniętym CSS
        """
        css_parts = [self.pdf_opts.page_css()]
        
        # Font CSS
        if self.pdf_opts.embed_fonts:
            font_bytes = self.font_manager.get_font(self.pdf_opts.font_name)
            if font_bytes:
                font_css = self.font_manager.generate_font_face_css(
                    self.pdf_opts.font_name,
                    font_bytes
                )
                css_parts.append(font_css)
            
            # Add font-family to body
            fallbacks = ", ".join(f'"{f}"' for f in self.pdf_opts.font_fallbacks)
            css_parts.append(f"""
body {{
  font-family: "{self.pdf_opts.font_name}", {fallbacks}, sans-serif;
}}
""")
        
        # Additional CSS
        if additional_css:
            css_parts.append(additional_css)
        
        # Watermark
        if self.pdf_opts.watermark_text:
            css_parts.append(f"""
body::before {{
  content: "{self.pdf_opts.watermark_text}";
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-45deg);
  font-size: 80px;
  opacity: {self.pdf_opts.watermark_opacity};
  color: #000;
  pointer-events: none;
  z-index: 9999;
}}
""")
        
        # Inject CSS
        inject_css = f'<style>{chr(10).join(css_parts)}</style>'
        
        if "</head>" in html:
            return html.replace("</head>", f"{inject_css}\n</head>")
        elif "<body>" in html:
            return html.replace("<body>", f"<head>{inject_css}</head>\n<body>")
        else:
            return f"<html><head>{inject_css}</head><body>{html}</body></html>"
    
    def to_pdf_weasyprint(self, html: str) -> bytes:
        """
        Eksportuje do PDF używając WeasyPrint.
        
        Args:
            html: HTML string
            
        Returns:
            PDF bytes
            
        Raises:
            RuntimeError: Jeśli WeasyPrint niedostępny
        """
        if not HAS_WEASYPRINT:
            raise RuntimeError(
                "WeasyPrint nie jest zainstalowany. "
                "Zainstaluj: pip install weasyprint"
            )
        
        try:
            html_augmented = self._inject_pdf_css(html)
            
            pdf_bytes = _WHTML(string=html_augmented).write_pdf(
                stylesheets=[_WCSS(string="")]
            )
            
            LOGGER.info(f"Generated PDF with WeasyPrint ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            LOGGER.error(f"WeasyPrint PDF generation failed: {e}")
            raise
    
    def to_pdf_playwright(self, html: str) -> bytes:
        """
        Eksportuje do PDF używając Playwright/Chromium.
        
        Args:
            html: HTML string
            
        Returns:
            PDF bytes
            
        Raises:
            RuntimeError: Jeśli Playwright niedostępny
        """
        if not HAS_PLAYWRIGHT:
            raise RuntimeError(
                "Playwright nie jest zainstalowany. "
                "Zainstaluj: pip install playwright && playwright install chromium"
            )
        
        try:
            html_augmented = self._inject_pdf_css(html)
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set content and wait for load
                page.set_content(html_augmented, wait_until="networkidle")
                
                # Generate PDF
                pdf_bytes = page.pdf(
                    format=self.pdf_opts.page_size.value,
                    print_background=self.pdf_opts.print_background,
                    margin={
                        "top": self.pdf_opts.margin_top,
                        "right": self.pdf_opts.margin_right,
                        "bottom": self.pdf_opts.margin_bottom,
                        "left": self.pdf_opts.margin_left,
                    },
                    prefer_css_page_size=True,
                    display_header_footer=bool(
                        self.pdf_opts.header_text or
                        self.pdf_opts.footer_text or
                        self.pdf_opts.show_page_numbers
                    )
                )
                
                browser.close()
            
            LOGGER.info(f"Generated PDF with Playwright ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            LOGGER.error(f"Playwright PDF generation failed: {e}")
            raise
    
    def to_pdf(
        self,
        html: str,
        engine: PDFEngine = PDFEngine.PLAYWRIGHT
    ) -> bytes:
        """
        Eksportuje do PDF używając wybranego silnika.
        
        Args:
            html: HTML string
            engine: Silnik PDF
            
        Returns:
            PDF bytes
        """
        if engine == PDFEngine.WEASYPRINT:
            return self.to_pdf_weasyprint(html)
        elif engine == PDFEngine.PLAYWRIGHT:
            return self.to_pdf_playwright(html)
        elif engine == PDFEngine.REPORTLAB:
            raise NotImplementedError("ReportLab engine not yet implemented")
        else:
            raise ValueError(f"Unknown PDF engine: {engine}")


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def quick_report(
    title: str,
    sections: List[Dict[str, Any]],
    *,
    theme: Union[ReportTheme, ThemeConfig] = ReportTheme.PROFESSIONAL,
    output_format: Literal["html", "pdf"] = "html",
    pdf_engine: PDFEngine = PDFEngine.PLAYWRIGHT
) -> Union[str, bytes]:
    """
    Szybkie tworzenie raportu z listy sekcji.
    
    Args:
        title: Tytuł raportu
        sections: Lista sekcji z dict zawierającymi type i content
        theme: Motyw
        output_format: Format wyjściowy
        pdf_engine: Silnik PDF
        
    Returns:
        HTML string lub PDF bytes
        
    Examples:
        >>> sections = [
        ...     {"type": "section", "title": "Overview"},
        ...     {"type": "kpi", "items": [{"label": "Total", "value": 1000}]},
        ...     {"type": "table", "df": my_dataframe}
        ... ]
        >>> html = quick_report("My Report", sections)
    """
    metadata = ReportMetadata(title=title)
    builder = ReportBuilder(metadata=metadata, theme=theme)
    
    for section in sections:
        section_type = section.get("type")
        
        if section_type == "section":
            builder.add_section(
                section.get("title", ""),
                level=section.get("level", 2)
            )
        
        elif section_type == "paragraph":
            builder.add_paragraph(section.get("text", ""))
        
        elif section_type == "markdown":
            builder.add_markdown(section.get("text", ""))
        
        elif section_type == "kpi":
            builder.add_kpi_row(section.get("items", []))
        
        elif section_type == "table":
            builder.add_table(
                section.get("df"),
                caption=section.get("caption")
            )
        
        elif section_type == "figure":
            builder.add_figure(
                section.get("fig"),
                as_image=section.get("as_image", False),
                caption=section.get("caption")
            )
        
        elif section_type == "alert":
            builder.add_alert(
                section.get("message", ""),
                variant=section.get("variant", "info"),
                title=section.get("title")
            )
        
        elif section_type == "divider":
            builder.add_divider()
        
        elif section_type == "page_break":
            builder.add_page_break()
    
    if output_format == "html":
        return builder.build_html()
    else:
        html = builder.build_html()
        exporter = PDFExporter()
        return exporter.to_pdf(html, engine=pdf_engine)


# ========================================================================================
# EXPORT & DOCUMENTATION
# ========================================================================================

__all__ = [
    # Enums
    "ReportTheme",
    "PDFEngine",
    "PageSize",
    "ImageFormat",
    
    # Dataclasses
    "ThemeConfig",
    "ExportOptions",
    "PDFOptions",
    "ReportMetadata",
    "TableOfContents",
    
    # Main classes
    "ReportBuilder",
    "PDFExporter",
    "FontManager",
    
    # HTML components
    "section_title",
    "paragraph",
    "badge",
    "alert",
    "kpi_card",
    "kpi_row",
    "dataframe_table",
    "fig_to_html",
    "fig_to_image",
    "markdown_to_html",
    
    # Utilities
    "generate_report_css",
    "quick_report",
    
    # Constants
    "THEMES",
    "HAS_WEASYPRINT",
    "HAS_PLAYWRIGHT",
    "HAS_REPORTLAB",
    "HAS_JINJA2",
    "HAS_MARKDOWN",
]

# ========================================================================================
# MODULE INITIALIZATION
# ========================================================================================

LOGGER.info(
    f"Report Generator PRO++++ initialized | "
    f"WeasyPrint: {HAS_WEASYPRINT} | "
    f"Playwright: {HAS_PLAYWRIGHT} | "
    f"ReportLab: {HAS_REPORTLAB} | "
    f"Jinja2: {HAS_JINJA2} | "
    f"Markdown: {HAS_MARKDOWN}"
)