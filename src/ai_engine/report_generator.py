# src/ai_engine/report_generator.py — PRO++
from __future__ import annotations

import os
import json
import time
import pathlib
import traceback
import threading
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
)

# -----------------------------
# Domyślne ścieżki
# -----------------------------
THIS_FILE = pathlib.Path(__file__).resolve()
# standardowe miejsce assets/templates względem repo
DEFAULT_ASSETS_DIR = THIS_FILE.parents[2] / "assets" / "templates"
DEFAULT_TEMPLATE = DEFAULT_ASSETS_DIR / "report_template.html"

# Środowiskowa ścieżka alternatywna (np. dla deploymentu)
ENV_TEMPLATES_DIR = os.getenv("REPORT_TEMPLATES_DIR")

# -----------------------------
# Cache środowisk per katalog
# -----------------------------
_env_cache: Dict[str, Environment] = {}
_env_lock = threading.Lock()


def _make_env(templates_dir: pathlib.Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # --- filtry ---
    env.filters["tojson_pretty"] = lambda obj: json.dumps(obj, ensure_ascii=False, indent=2)
    env.filters["safe_json"] = lambda obj: json.dumps(obj, ensure_ascii=False)
    env.filters["nl2br"] = lambda s: (str(s) or "").replace("\n", "<br/>")
    env.filters["truncate_chars"] = lambda s, n=400: (s if len(str(s)) <= n else str(s)[: n - 1] + "…")
    env.filters["fmt_num"] = lambda x: f"{x:,.0f}".replace(",", " ").replace(".0", "")

    # --- globalne pomocniki ---
    env.globals["now"] = lambda fmt="%Y-%m-%d %H:%M": time.strftime(fmt)
    return env


def _get_env_for_dir(templates_dir: pathlib.Path) -> Environment:
    key = str(templates_dir.resolve())
    with _env_lock:
        if key not in _env_cache:
            _env_cache[key] = _make_env(templates_dir)
        return _env_cache[key]


# -----------------------------
# Lokalizacja szablonu
# -----------------------------
def _candidate_template_dirs(template_path: Optional[pathlib.Path]) -> List[pathlib.Path]:
    """
    Zwraca listę katalogów, w których szukamy szablonu i ładujemy Jinja2.
    Priorytety:
      1) katalog wskazany przez template_path (jeśli podano),
      2) REPORT_TEMPLATES_DIR (env),
      3) bieżący katalog roboczy ./assets/templates,
      4) domyślny DEFAULT_ASSETS_DIR.
    """
    dirs: List[pathlib.Path] = []
    if template_path:
        dirs.append(template_path.parent.resolve())
    if ENV_TEMPLATES_DIR:
        dirs.append(pathlib.Path(ENV_TEMPLATES_DIR).resolve())
    # ./assets/templates w cwd (np. przy uruchomieniu z repo root)
    cwd_assets = pathlib.Path.cwd() / "assets" / "templates"
    dirs.append(cwd_assets.resolve())
    # domyślny
    dirs.append(DEFAULT_ASSETS_DIR.resolve())

    # deduplikacja przy zachowaniu kolejności
    seen = set()
    uniq: List[pathlib.Path] = []
    for d in dirs:
        if str(d) not in seen:
            uniq.append(d)
            seen.add(str(d))
    return uniq


def _locate_template(template_path: Optional[pathlib.Path]) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Zwraca krotkę: (katalog_loadera, finalna_ścieżka_szablonu)
    Jeśli nie znajdzie wskazanego pliku, zwraca (ostatni_katalog, ścieżkę_domyslnego_raportu_mogącą_nie_istnieć).
    """
    if template_path and template_path.exists():
        return template_path.parent, template_path

    # Jeśli nie podano pliku, użyj domyślnego nazwy 'report_template.html'
    filename = (template_path.name if template_path else DEFAULT_TEMPLATE.name)
    for d in _candidate_template_dirs(template_path):
        candidate = d / filename
        if candidate.exists():
            return d, candidate

    # Nie znaleziono — wskaż ostatni katalog i ścieżkę domyślnego pliku (może nie istnieć)
    dirs = _candidate_template_dirs(template_path)
    return dirs[-1], (dirs[-1] / filename)


# -----------------------------
# Domyślne wartości kontekstu
# -----------------------------
_DEFAULTS: Dict[str, Any] = {
    "title": "Raport",
    "subtitle": "",
    "run_meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M")},
    "metrics": {},
    "notes": "",
    "kpis": None,
    "tags": None,
    "data_dictionary": None,
    "forecast": None,
    "anomalies": None,
    "recommendations": None,
    "insights": None,
    "model_card": None,
    "df_preview": None,
}


# -----------------------------
# Public API (zachowuje podpis)
# -----------------------------
def build_report_html(context: dict[str, Any], template_path: str | pathlib.Path | None = None) -> str:
    """
    Renderuje raport HTML na bazie szablonu Jinja2 i kontekstu.
    - context: dict z danymi (może być niepełny)
    - template_path: opcjonalna ścieżka do pliku szablonu (możesz wskazać dowolny .html w katalogu loadera)
    Zwraca zawsze **HTML** (nawet w przypadku błędu — wtedy ładny fallback z diagnostyką).
    """
    tpl_path = pathlib.Path(template_path) if template_path else None
    loader_dir, final_path = _locate_template(tpl_path)

    # Gdy plik nie istnieje — użyj fallbacku HTML z diagnostyką
    if not final_path.exists():
        return _render_missing_template_fallback(final_path, loader_dir, context)

    try:
        env = _get_env_for_dir(loader_dir)
        tpl = env.get_template(final_path.name)
        merged = {**_DEFAULTS, **(context or {})}
        html = tpl.render(**merged)
        return html
    except Exception as e:
        return _render_exception_fallback(e, context, loader_dir, final_path)


# -----------------------------
# Dodatkowe API (opcjonalnie)
# -----------------------------
@dataclass(frozen=True)
class RenderDiagnostics:
    ok: bool
    message: str
    loader_dir: str
    template_file: str
    candidates: List[str]
    traceback: Optional[str] = None

@dataclass(frozen=True)
class RenderResult:
    html: str
    diagnostics: RenderDiagnostics


def render_report_html_pro(context: dict[str, Any], template_path: str | pathlib.Path | None = None) -> RenderResult:
    """Wersja z diagnostyką: przydaje się w testach i CI."""
    tpl_path = pathlib.Path(template_path) if template_path else None
    loader_dir, final_path = _locate_template(tpl_path)

    candidates = [str(p) for p in _candidate_template_dirs(tpl_path)]
    if not final_path.exists():
        html = _render_missing_template_fallback(final_path, loader_dir, context)
        di = RenderDiagnostics(
            ok=False,
            message="Template file not found",
            loader_dir=str(loader_dir),
            template_file=str(final_path),
            candidates=candidates,
            traceback=None,
        )
        return RenderResult(html=html, diagnostics=di)

    try:
        env = _get_env_for_dir(loader_dir)
        tpl = env.get_template(final_path.name)
        merged = {**_DEFAULTS, **(context or {})}
        html = tpl.render(**merged)
        di = RenderDiagnostics(
            ok=True,
            message="OK",
            loader_dir=str(loader_dir),
            template_file=str(final_path),
            candidates=candidates,
        )
        return RenderResult(html=html, diagnostics=di)
    except Exception as e:
        tb = traceback.format_exc()
        html = _render_exception_fallback(e, context, loader_dir, final_path)
        di = RenderDiagnostics(
            ok=False,
            message=str(e),
            loader_dir=str(loader_dir),
            template_file=str(final_path),
            candidates=candidates,
            traceback=tb,
        )
        return RenderResult(html=html, diagnostics=di)


# -----------------------------
# Fallbacki HTML
# -----------------------------
def _render_missing_template_fallback(final_path: pathlib.Path, loader_dir: pathlib.Path, ctx: dict[str, Any]) -> str:
    merged = {**_DEFAULTS, **(ctx or {})}
    candidates = "\n".join([f"<li><code>{p}</code></li>" for p in _candidate_template_dirs(final_path)])
    return f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<title>Raport — brak szablonu</title>
<style>
body{{font-family:system-ui;margin:24px;line-height:1.45}}
code{{background:#f5f5f5;padding:2px 4px;border-radius:4px}}
pre{{background:#f5f5f5;padding:12px;border-radius:8px;overflow:auto}}
small{{color:#666}}
</style></head>
<body>
  <h1>❌ Brak szablonu raportu</h1>
  <p>Nie znaleziono pliku: <code>{final_path}</code></p>
  <h3>Gdzie szukałem?</h3>
  <ul>{candidates}</ul>
  <h3>Minimalny podgląd danych</h3>
  <p><strong>{merged.get('title','Raport')}</strong> — <small>{merged.get('subtitle','')}</small></p>
  <pre>{_safe_json(merged)}</pre>
  <hr/>
  <small>Loader dir: <code>{loader_dir}</code></small>
</body></html>"""

def _render_exception_fallback(e: Exception, ctx: dict[str, Any], loader_dir: pathlib.Path, final_path: pathlib.Path) -> str:
    tb = traceback.format_exc()
    merged = {**_DEFAULTS, **(ctx or {})}
    return f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<title>Raport — błąd renderowania</title>
<style>
body{{font-family:system-ui;margin:24px;line-height:1.45}}
code{{background:#f5f5f5;padding:2px 4px;border-radius:4px}}
pre{{background:#f5f5f5;padding:12px;border-radius:8px;overflow:auto}}
small{{color:#666}}
</style></head>
<body>
  <h1>❌ Błąd renderowania raportu</h1>
  <p>{_escape_html(str(e))}</p>
  <h3>Diagnostyka</h3>
  <ul>
    <li>Loader dir: <code>{loader_dir}</code></li>
    <li>Template file: <code>{final_path}</code></li>
  </ul>
  <h3>Traceback</h3>
  <pre>{_escape_html(tb)}</pre>
  <h3>Minimalny kontekst</h3>
  <pre>{_safe_json(merged)}</pre>
</body></html>"""

# -----------------------------
# Helpers
# -----------------------------
def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
