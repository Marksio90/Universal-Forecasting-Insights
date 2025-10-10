# === STYLING_UTILS ===
from __future__ import annotations
import streamlit as st
from pathlib import Path

@st.cache_data(show_spinner=False)
def _load_css_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")

def apply_theme_css(css_path: str = "frontend/theme.css") -> None:
    """Ładuje i wstrzykuje CSS do aplikacji (z cache). Bezpieczne, idempotentne."""
    css = _load_css_text(css_path)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def kpi_row_pro(items: dict[str, str|int|float]) -> None:
    """Karty KPI w jednej linii, zgodne ze stylem .kpi-card"""
    cols = st.columns(len(items))
    for (k, v), c in zip(items.items(), cols):
        with c:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="small">{k}</div>
                  <div style="font-size:1.6rem; font-weight:800; line-height:1.2">{v}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
