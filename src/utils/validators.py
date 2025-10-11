# src/frontend/css_injector.py
# === CSS INJECTOR PRO+++ ===
from __future__ import annotations
from typing import Iterable, Union, Optional, Dict, List
from pathlib import Path
import os, json, hashlib
import streamlit as st

# — minimalny fallback, kiedy brak własnego CSS —
_FALLBACK_CSS = """
:root{ --radius:14px; --muted:#8b95a7; }
.main .block-container{ max-width:1200px; }
.section{ padding:1rem 1.2rem; border-radius:var(--radius); border:1px solid rgba(255,255,255,.08); margin:.75rem 0; }
.kpi-card{ border-radius:var(--radius); padding:1rem; border:1px solid rgba(108,92,231,.25); }
.small{ font-size:.9rem; color:var(--muted); }
"""

# === PLIK I CACHE ===
def _file_mtime(path: str) -> float | None:
    try: return os.path.getmtime(path)
    except Exception: return None

@st.cache_data(show_spinner=False)
def _read_css_text(path: str, mtime: float | None) -> str:
    """Czyta plik CSS; 'mtime' wymusza invalidację cache przy zmianie pliku."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def _root_vars_block(vars_map: Dict[str, str] | None) -> str:
    if not vars_map: return ""
    pairs = [f"--{str(k).strip().lstrip('-').replace(' ','-')}:{v}" for k,v in vars_map.items()]
    return ":root{" + ";".join(pairs) + "}"

def _digest_blob(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        if p: h.update(p.encode("utf-8"))
    return h.hexdigest()

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1","true","yes","on"}: return True
    if raw in {"0","false","no","off"}: return False
    return default

def _paths_from_env(default: List[str]) -> List[str]:
    raw = os.getenv("CSS_GLOBAL_PATHS", "").strip()
    if not raw: return default
    try:
        if raw.startswith("["):  # JSON list
            return [str(x) for x in json.loads(raw)]
    except Exception:
        pass
    return [p.strip() for p in raw.split(",") if p.strip()]

def _vars_from_env() -> Dict[str, str] | None:
    raw = os.getenv("CSS_VARS_JSON", "").strip()
    if not raw: return None
    try:
        data = json.loads(raw)
        return {str(k): str(v)} if isinstance(data, dict) else None
    except Exception:
        return None

def _inject_css_once(css_all: str, *, key: str) -> bool:
    """Idempotentne wstrzyknięcie CSS – ponawia tylko przy zmianie treści."""
    digest = _digest_blob(css_all)
    ss_key = f"_css_digest_{key}"
    if st.session_state.get(ss_key) == digest:
        return False
    st.session_state[ss_key] = digest
    st.markdown(f"<style>{css_all}</style>", unsafe_allow_html=True)
    return True

def clear_css_cache() -> None:
    """Czyści cache plików + wymusza ponowne wstrzyknięcie CSS przy następnym wywołaniu."""
    try: st.cache_data.clear()
    except Exception: ...
    for k in list(st.session_state.keys()):
        if k.startswith("_css_digest_"): del st.session_state[k]

# === PUBLIC API ===
def apply_css(
    css_paths: Union[str, Iterable[str]] = "assets/styles/global.css",
    *,
    css_vars: Optional[Dict[str, str]] = None,
    extra_css: str = "",
    key: str = "global",
    dev_reload: Optional[bool] = None,
    include_fallback: bool = True,
) -> bool:
    """
    Wstrzykuje CSS (multi-file + :root vars + extra). Zwraca True jeśli nastąpiła aktualizacja.
    ENV:
      - CSS_GLOBAL_PATHS='a.css,b.css' lub JSON listą
      - CSS_VARS_JSON='{"accent":"#6C5CE7"}'
      - DEV_RELOAD_CSS=1   # invalidacja cache po zmianie pliku (mtime)
    """
    paths = [css_paths] if isinstance(css_paths, str) else list(css_paths)
    if dev_reload is None:
        dev_reload = _bool_env("DEV_RELOAD_CSS", False)

    blobs: List[str] = []
    if include_fallback:
        blobs.append(_FALLBACK_CSS)

    for p in _paths_from_env(paths):
        mtime = _file_mtime(p) if dev_reload else None
        blobs.append(_read_css_text(p, mtime))

    env_vars = _vars_from_env()
    merged_vars = {**(env_vars or {}), **(css_vars or {})} if (env_vars or css_vars) else None
    if merged_vars:
        blobs.append(_root_vars_block(merged_vars))
    if extra_css:
        blobs.append(extra_css)

    css_all = "\n".join(x for x in blobs if x)
    return _inject_css_once(css_all, key=key)

# === WSTECZNA KOMPATYBILNOŚĆ ===
def inject_global_css() -> None:
    """
    Kompatybilna wersja uproszczona: wstrzyknij 'assets/styles/global.css'
    (plus fallback i zmienne z ENV).
    """
    apply_css("assets/styles/global.css", key="global")
