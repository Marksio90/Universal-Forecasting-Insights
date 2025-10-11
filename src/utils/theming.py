# src/frontend/css_injector.py
# === STYLING / CSS INJECTOR (PRO+++) ===
from __future__ import annotations
from typing import Iterable, Union, Optional, Dict, List
from pathlib import Path
import os, json, hashlib
import streamlit as st

# --- Fallback minimalny wygląd (jeśli nie dostarczysz własnego CSS) ---
_FALLBACK_CSS = """
:root{ --radius:14px; --muted:#8b95a7; }
.main .block-container{ max-width:1200px; }
.section{ padding:1rem 1.2rem; border-radius:var(--radius); border:1px solid rgba(255,255,255,.08); margin:.75rem 0; }
.kpi-card{ border-radius:var(--radius); padding:1rem; border:1px solid rgba(108,92,231,.25); }
.small{ font-size:.9rem; color:var(--muted); }
"""

# --- Cache odczytu pliku | invalidacja przez mtime (przekazywany jako argument) ---
@st.cache_data(show_spinner=False)
def _read_css_text(path: str, mtime: float | None) -> str:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def _file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def _root_vars_block(vars_map: Dict[str, str] | None) -> str:
    if not vars_map:
        return ""
    pairs = []
    for k, v in vars_map.items():
        k = str(k).strip().lstrip("-").replace(" ", "-")
        pairs.append(f"--{k}:{v}")
    return ":root{" + ";".join(pairs) + "}"

def _digest_blob(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        if p:
            h.update(p.encode("utf-8"))
    return h.hexdigest()

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1","true","yes","on"}: return True
    if raw in {"0","false","no","off"}: return False
    return default

def _paths_from_env(default: List[str]) -> List[str]:
    raw = os.getenv("CSS_GLOBAL_PATHS", "").strip()
    if not raw:
        return default
    # Akceptujemy CSV lub JSON list
    try:
        if raw.startswith("["):
            xs = json.loads(raw)
            return [str(x) for x in xs]
    except Exception:
        pass
    return [p.strip() for p in raw.split(",") if p.strip()]

def _vars_from_env() -> Dict[str, str] | None:
    raw = os.getenv("CSS_VARS_JSON", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return None
    return None

def _inject_css_once(css_all: str, *, key: str) -> bool:
    """
    Idempotentne wstrzyknięcie CSS: jeśli digest identyczny – nic nie robi.
    Zapamiętuje ostatni digest w session_state pod kluczem zależnym od 'key'.
    """
    digest = _digest_blob(css_all)
    ss_key = f"_css_digest_{key}"
    if st.session_state.get(ss_key) == digest:
        return False
    # wyczyść poprzednie wpisy (utrzymujemy pojedynczy aktywny wpis na dany 'key')
    st.session_state[ss_key] = digest
    st.markdown(f"<style>{css_all}</style>", unsafe_allow_html=True)
    return True

def clear_css_cache() -> None:
    """Czyści cache odczytu plików i wymusza ponowne wstrzyknięcie CSS przy następnym wywołaniu."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # nie czyścimy session_state digestów – możesz to zrobić selektywnie:
    for k in list(st.session_state.keys()):
        if k.startswith("_css_digest_"):
            del st.session_state[k]

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
    Wstrzykuje CSS do aplikacji Streamlit, łącząc wiele plików + :root zmienne + dodatkowy CSS.
    Idempotentne dzięki digestowi treści. Zwraca True, jeśli wystąpiła aktualizacja CSS.

    Args:
      css_paths: ścieżka lub lista ścieżek do plików CSS.
      css_vars: dict zmiennych CSS → do :root{--k:v}.
      extra_css: dodatkowy CSS inline (np. hotfix).
      key: nazwa przestrzeni – niezależne digesty (np. 'global', 'page-eda'…).
      dev_reload: jeśli True (lub ENV DEV_RELOAD_CSS=1), uwzględnia mtime w kluczu cache (auto-przeładowanie).
      include_fallback: czy dodać minimalny _FALLBACK_CSS na początku.
    """
    if isinstance(css_paths, str):
        paths = [css_paths]
    else:
        paths = list(css_paths)

    # DEV reload (auto: ENV DEV_RELOAD_CSS)
    if dev_reload is None:
        dev_reload = _bool_env("DEV_RELOAD_CSS", False)

    blobs: List[str] = []
    if include_fallback:
        blobs.append(_FALLBACK_CSS)

    # Odczyt plików (z invalidacją cache po mtime w trybie dev)
    for p in _paths_from_env(paths):
        mtime = _file_mtime(p) if dev_reload else None
        blobs.append(_read_css_text(p, mtime))

    # Zmienne CSS + extra
    env_vars = _vars_from_env()
    root_vars = {**(env_vars or {}), **(css_vars or {})} if css_vars or env_vars else None
    if root_vars:
        blobs.append(_root_vars_block(root_vars))
    if extra_css:
        blobs.append(extra_css)

    css_all = "\n".join(x for x in blobs if x)
    return _inject_css_once(css_all, key=key)

# === WSTECZNA KOMPATYBILNOŚĆ ===
def inject_global_css() -> None:
    """
    Zgodność z poprzednim API: wstrzyknij 'assets/styles/global.css'.
    Obsługuje ENV:
      - CSS_GLOBAL_PATHS: CSV lub JSON list ścieżek
      - CSS_VARS_JSON: JSON map zmiennych (np. {"accent":"#6C5CE7"})
      - DEV_RELOAD_CSS=1: włącz hot reload w dev (inwalidacja po mtime)
    """
    apply_css("assets/styles/global.css", key="global")
