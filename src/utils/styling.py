# === STYLING_UTILS (PRO+++) ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union, Any
from pathlib import Path
import hashlib
import numbers
import streamlit as st

# ---- Cache I/O ----
@st.cache_data(show_spinner=False)
def _load_css_text(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")

@st.cache_data(show_spinner=False)
def _min_css_inline_vars(vars_map: Dict[str, str]) -> str:
    """Zamienia dict zmiennych na minimalny blok :root{--key:value;}"""
    if not vars_map:
        return ""
    # prosty, bezpieczny sanitizing kluczy
    pairs = []
    for k, v in vars_map.items():
        k = str(k).strip().lstrip("-").replace(" ", "-")
        pairs.append(f"--{k}:{v}")
    return ":root{" + ";".join(pairs) + "}"

def _hash_blob(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()

# ---- Public API: CSS injection ----
def apply_theme_css(
    css_path: Union[str, Iterable[str]] = "frontend/theme.css",
    *,
    css_vars: Optional[Dict[str, str]] = None,
    extra_css: str = "",
) -> None:
    """
    Ładuje i wstrzykuje CSS (jeden lub wiele plików) oraz opcjonalne zmienne CSS.
    Idempotentne: nie wstrzyknie ponownie, jeśli zawartość + zmienne się nie zmieniły.
    """
    paths = [css_path] if isinstance(css_path, str) else list(css_path)
    blobs = []
    for p in paths:
        if p:
            blobs.append(_load_css_text(p))
    if css_vars:
        blobs.append(_min_css_inline_vars(css_vars))
    if extra_css:
        blobs.append(extra_css)

    css_all = "\n".join([b for b in blobs if b])
    # Dodaj bazowy, lekki reset + klasy, jeśli theme.css ich nie definiuje
    _fallback_css = """
:root{--radius:14px; --muted:#8b95a7}
.main .block-container{max-width:1200px}
.section{padding:1rem 1.2rem; border-radius:var(--radius); border:1px solid rgba(255,255,255,.08); margin:.75rem 0}
.kpi-card{border-radius:var(--radius); padding:1rem; border:1px solid rgba(108,92,231,.25)}
.small{font-size:.9rem; color:var(--muted)}
"""
    css_all = _fallback_css + "\n" + css_all if css_all else _fallback_css

    digest = _hash_blob(css_all)
    key = f"_styling_css_{digest}"
    if st.session_state.get(key):
        return  # już wstrzyknięte identyczne CSS
    # wyczyść poprzedni wpis (utrzymujemy jeden aktywny)
    for k in list(st.session_state.keys()):
        if k.startswith("_styling_css_"):
            del st.session_state[k]
    st.markdown(f"<style>{css_all}</style>", unsafe_allow_html=True)
    st.session_state[key] = True

# ---- KPI model & helpers ----
@dataclass
class KPI:
    label: str
    value: Union[str, int, float]
    delta: Optional[Union[str, int, float]] = None   # np. +0.6pp | -3.1% | 12
    fmt: Optional[str] = None                        # np. ".2f", ",.0f", ".2%", None
    help: Optional[str] = None

def _format_value(v: Any, fmt: Optional[str]) -> str:
    # string → zwróć jak jest
    if isinstance(v, str):
        return v
    # liczby – formaty: ".2f", ",.0f", ".2%", itp.
    if isinstance(v, numbers.Number):
        if fmt:
            if fmt.endswith("%"):
                # np. ".2%" -> procent z mnożeniem 100
                digits = fmt[:-1] or ".2"
                return f"{float(v):{digits}f}%"
            return f"{float(v):{fmt}}"
        # domyślne formatowanie
        return f"{v}"
    # inne typy
    return str(v)

def _delta_class(delta: Optional[Union[str, int, float]]) -> str:
    if delta is None:
        return ""
    try:
        # akceptuj: "-2.0%", "+0.3pp", "-1", 1.2
        s = str(delta).strip().replace("pp", "").replace("%", "")
        val = float(s)
        return "up" if val >= 0 else "down"
    except Exception:
        # jeśli nie jesteśmy w stanie sparsować → uznaj jako neutral
        return "up"

# ---- Public UI: KPI row ----
def kpi_row_pro(items: Union[Dict[str, Union[str, int, float]], Iterable[KPI]], *, default_fmt: Optional[str] = None) -> None:
    """
    Renderuje KPI w jednej linii.
    - Wstecznie: podaj dict {"AUC":0.964, "F1":0.91}
    - Nowe: lista KPI(label, value, delta=?, fmt=?, help=?)
    """
    # zainicjuj styl (jednorazowo)
    apply_theme_css()  # nic nie zrobi, jeśli już załadowane

    # normalizacja wejścia
    if isinstance(items, dict):
        data: List[KPI] = [KPI(k, v, None, default_fmt, None) for k, v in items.items()]
    else:
        data = [k if isinstance(k, KPI) else KPI(str(k), k) for k in items]  # type: ignore

    n = max(1, len(data))
    cols = st.columns(n, gap="small")
    # render
    for kpi, c in zip(data, cols):
        with c:
            value_txt = _format_value(kpi.value, kpi.fmt or default_fmt)
            delta_html = ""
            if kpi.delta is not None:
                cls = _delta_class(kpi.delta)
                delta_val = _format_value(kpi.delta, None)
                delta_html = f'<div class="small delta {cls}">{delta_val}</div>'
            title_attr = f'title="{kpi.help}"' if kpi.help else ""
            st.markdown(
                f"""
                <div class="kpi-card" {title_attr}>
                  <div class="small">{kpi.label}</div>
                  <div style="font-size:1.6rem; font-weight:800; line-height:1.2">{value_txt}</div>
                  {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
