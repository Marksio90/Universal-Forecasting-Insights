from __future__ import annotations
import json
import pathlib
import traceback
from typing import Any, Optional
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape

# -----------------------------
# Domyślny template path
# -----------------------------
ASSETS_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "templates"
DEFAULT_TEMPLATE = ASSETS_DIR / "report_template.html"

# Cache środowiska
_env_cache: Optional[Environment] = None

def _get_env() -> Environment:
    """Zwraca skonfigurowane środowisko Jinja2 z cache."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    env = Environment(
        loader=FileSystemLoader(str(ASSETS_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # niestandardowe filtry
    env.filters["tojson_pretty"] = lambda obj: json.dumps(obj, ensure_ascii=False, indent=2)
    env.filters["safe_json"] = lambda obj: json.dumps(obj, ensure_ascii=False)
    _env_cache = env
    return env

# -----------------------------
# Budowa raportu
# -----------------------------
def build_report_html(context: dict[str, Any], template_path: str | pathlib.Path | None = None) -> str:
    """
    Renderuje raport HTML na bazie szablonu Jinja2 i kontekstu.
    - context: dict z danymi (może być niepełny)
    - template_path: opcjonalna ścieżka alternatywnego szablonu
    """
    tpl_path = pathlib.Path(template_path) if template_path else DEFAULT_TEMPLATE

    if not tpl_path.exists():
        # fallback: prosty HTML z komunikatem
        return f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Raport</title></head>
<body><h1>❌ Brak szablonu raportu</h1>
<p>Nie znaleziono pliku: <code>{tpl_path}</code></p></body></html>"""

    try:
        env = _get_env()
        tpl = env.get_template(tpl_path.name)

        # Domyślne wartości, by raport zawsze się wygenerował
        defaults = {
            "title": "Raport",
            "subtitle": "",
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
        }
        merged = {**defaults, **(context or {})}

        html = tpl.render(**merged)
        return html

    except Exception as e:
        tb = traceback.format_exc()
        return f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Błąd raportu</title></head>
<body style="font-family:system-ui">
<h1>❌ Błąd renderowania raportu</h1>
<p>{e}</p>
<pre>{tb}</pre>
</body></html>"""
