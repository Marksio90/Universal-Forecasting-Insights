# profiling_report.py — PRO++
from __future__ import annotations

import html
import traceback
from dataclasses import dataclass
from typing import Optional, Literal

import pandas as pd

try:
    # ydata-profiling (dawniej pandas-profiling)
    from ydata_profiling import ProfileReport  # type: ignore
except Exception:
    ProfileReport = None  # type: ignore


# =========================
# Konfiguracja / typy
# =========================
Mode = Literal["minimal", "explorative", "full"]

@dataclass(frozen=True)
class ProfileOptions:
    mode: Mode = "minimal"
    sample_max: int = 5_000
    use_random_sample: bool = True
    random_state: int = 42
    correlations: bool = False
    max_cols: int = 1_000            # safety: twardy limit kolumn do profilowania
    html_theme: str = "flatly"       # motyw ydata_profiling
    full_width: bool = True


# =========================
# Public API (kompatybilne)
# =========================
def make_profile_html(
    df: pd.DataFrame,
    title: str = "Data Profile",
    mode: str = "minimal",
    sample_max: int = 5000,
    correlations: bool = False,
) -> str:
    """
    Tworzy bezpieczny i szybki raport profilujący (HTML) dla danych.
    (Back-compat: zachowuje ten sam podpis co wcześniej)
    """
    opts = ProfileOptions(
        mode=_coerce_mode(mode),
        sample_max=sample_max,
        correlations=bool(correlations),
    )
    return _make_profile_html_pro(df, title=title, opts=opts)


# =========================
# Wersja PRO z opcjami (do użycia w nowych miejscach)
# =========================
def make_profile_html_pro(
    df: pd.DataFrame,
    title: str = "Data Profile",
    opts: ProfileOptions = ProfileOptions(),
) -> str:
    """Nowe, bogatsze API (alias)."""
    return _make_profile_html_pro(df, title=title, opts=opts)


# =========================
# Implementacja
# =========================
def _coerce_mode(m: str) -> Mode:
    m = (m or "minimal").strip().lower()
    return "explorative" if m == "explorative" else ("full" if m == "full" else "minimal")

def _html_note(text: str) -> str:
    return f"<p style='color:#888;font-size:0.9em'><em>{html.escape(text)}</em></p>"

def _fallback_kpi_html(df: pd.DataFrame, title: str, note: str, err: Optional[str] = None) -> str:
    # bardzo lekki raport HTML na wypadek braku ydata_profiling lub błędu
    rows, cols = df.shape
    missing_pct = float((df.isna().sum().sum()) / (df.size or 1)) * 100.0 if rows and cols else 0.0
    dupes = int(df.duplicated().sum()) if rows else 0
    head_html = df.head(20).to_html(index=False, border=0)
    err_block = f"<pre style='background:#f8f8f8;padding:10px;border-radius:8px'>{html.escape(err)}</pre>" if err else ""
    return f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui;margin:20px;line-height:1.45}}
.card{{border:1px solid #eee;border-radius:12px;padding:16px;margin:8px 0;background:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.kpi{{text-align:center;padding:10px;background:#fafafa;border-radius:10px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #eee;padding:6px;text-align:left}}
thead th{{background:#fafafa}}
small{{color:#666}}
</style>
</head><body>
<h1>{html.escape(title)}</h1>
{_html_note(note)}
<div class="grid">
  <div class="kpi"><div><strong>Wiersze</strong></div><div>{rows:,}</div></div>
  <div class="kpi"><div><strong>Kolumny</strong></div><div>{cols:,}</div></div>
  <div class="kpi"><div><strong>Braki</strong></div><div>{missing_pct:.2f}%</div></div>
  <div class="kpi"><div><strong>Duplikaty</strong></div><div>{dupes:,}</div></div>
</div>
<div class="card">
  <h3>Podgląd (pierwsze 20 wierszy)</h3>
  {head_html}
</div>
{('<div class="card"><h3>Błąd generatora profilu</h3>'+err_block+'</div>') if err else ''}
<p><small>Wersja lite — ydata_profiling niedostępny lub wystąpił błąd.</small></p>
</body></html>"""

def _make_profile_html_pro(df: pd.DataFrame, title: str, opts: ProfileOptions) -> str:
    # Walidacja wejścia
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "<p><strong>Brak danych do profilowania.</strong></p>"

    original_rows = len(df)
    original_cols = df.shape[1]

    # Safety: ogranicz liczbę kolumn
    if original_cols > opts.max_cols:
        df = df.iloc[:, : opts.max_cols]
        col_note = f"Ucięto kolumny do pierwszych {opts.max_cols} (z {original_cols}). "
    else:
        col_note = ""

    # Sampling dla wydajności
    note = ""
    if original_rows > opts.sample_max:
        if opts.use_random_sample:
            df = df.sample(opts.sample_max, random_state=opts.random_state)
            note = f"Użyto próbki {opts.sample_max:,} wierszy z {original_rows:,}. "
        else:
            df = df.head(opts.sample_max)
            note = f"Użyto pierwszych {opts.sample_max:,} wierszy z {original_rows:,}. "
    else:
        note = f"Profil pełnych danych ({original_rows:,} wierszy). "
    note += col_note

    # Wyłącz ciężkie części, jeśli trzeba
    minimal = opts.mode == "minimal"
    explorative = opts.mode == "explorative"
    # korelacje mogą być bardzo wolne — wybieramy jawnie
    corr_cfg = None
    if opts.correlations:
        corr_cfg = {
            "pearson": True,
            "spearman": True,
            "kendall": False,
            "phi_k": False,
            "cramers": False,
        }

    # Gdy brak ydata_profiling — fallback
    if ProfileReport is None:
        return _fallback_kpi_html(df, title, note, err="Pakiet ydata_profiling nie jest zainstalowany.")

    # Spróbuj pełnego profilu
    try:
        profile = ProfileReport(
            df,
            title=title,
            minimal=minimal,
            explorative=explorative,
            correlations=corr_cfg,
            progress_bar=False,
            pool_size=0,           # bez multiprocesów – stabilniej w środowiskach serwerowych
            infer_dtypes=True,
            html={"style": {"full_width": opts.full_width, "theme": opts.html_theme}},
        )
        html_out = profile.to_html()
        return _html_note(note) + html_out
    except Exception as e:
        # Fallback: lekkie HTML + traceback
        return _fallback_kpi_html(df, title, note, err=traceback.format_exc())
