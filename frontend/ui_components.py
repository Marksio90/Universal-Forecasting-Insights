# frontend/ui_components.py
# === KONTEKST ===
# Komponenty UI PRO+++ dla Streamlit:
# - Globalny CSS (dark/light) + jednorazowa injekcja
# - Sekcje z podtytułem i ikoną
# - KPI grid (obsługa delta/tooltip)
# - Karta z podglądem DataFrame (+ rozmiar/mem) i strefa pobrań (CSV/Parquet/JSON)
# - Badge, alert/toast, code block, spinner helper
# - Całość defensywna: type hints, cache, fallback gdy brak pyarrow, brak st.toast itd.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Tuple
import io
import sys

import pandas as pd
import streamlit as st

# === STYL / CSS (dopasowany do Twojego motywu) ===
_BASE_CSS = """
:root{
  --bg:#0b0f19; --card:#121826; --muted:#8b95a7; --accent:#6C5CE7;
  --radius:14px; --border:rgba(255,255,255,.10); --text:#fff;
}
.main .block-container{padding-top:1.2rem; max-width:1200px;}
.stButton>button{border-radius:var(--radius); padding:.7rem 1.1rem; font-weight:600}
.kpi-card{border-radius:var(--radius); padding:1rem; border:1px solid rgba(108,92,231,.25); background:linear-gradient(180deg, rgba(108,92,231,.10), rgba(108,92,231,.03))}
.section{padding:.8rem 1rem; border-radius:var(--radius); border:1px solid var(--border); margin-bottom:1rem; background:var(--card)}
.small{font-size:.9rem; color:var(--muted)}
.badge{display:inline-block; padding:.2rem .5rem; border:1px solid var(--border); border-radius:999px; font-size:.75rem; color:var(--muted);}
.hr{height:1px; background:var(--border); margin:.6rem 0}
.table-wrap{border:1px solid var(--border); border-radius:12px; padding:.4rem; background:var(--card)}
"""

def apply_global_css(extra_css: Optional[str] = None) -> None:
    """Wstrzyknij globalny CSS tylko raz na sesję."""
    key = "_dg_css_applied"
    if not st.session_state.get(key):
        st.markdown(f"<style>{_BASE_CSS}{extra_css or ''}</style>", unsafe_allow_html=True)
        st.session_state[key] = True


# === DANE KPI ===
@dataclass
class KPI:
    label: str
    value: Any
    delta: Optional[str] = None   # np. "+4.3%" / "-2"
    help: Optional[str] = None    # tooltip

def kpi_row(items: Iterable[KPI] | Dict[str, Any], *, cols: Optional[int] = None) -> None:
    """
    Renderuj rząd KPI. Możesz podać:
      - listę KPI(label, value, delta?, help?), albo
      - dict {"AUC":0.96, "F1":0.91, ...} (bez delta/help).
    """
    # Normalizacja wejścia
    if isinstance(items, dict):
        data = [KPI(k, v) for k, v in items.items()]
    else:
        data = list(items)

    n = len(data)
    if n == 0:
        return
    ncols = cols or min(max(n, 1), 6)
    col_elems = st.columns(ncols)
    for i, k in enumerate(data):
        with col_elems[i % ncols]:
            st.container()
            st.metric(label=k.label, value=k.value, delta=k.delta, help=k.help)


# === SEKCJE / NAGŁÓWKI ===
def section(title: str, subtitle: Optional[str] = None, *, icon: Optional[str] = None, anchor: Optional[str] = None) -> None:
    """
    Otwórz sekcję z tytułem i opcjonalnym podtytułem.
    Użyj przed grupą widgetów w głównej kolumnie.
    """
    apply_global_css()
    ico = f"{icon} " if icon else ""
    anc = f'<a id="{anchor}"></a>' if anchor else ""
    st.markdown(
        f"""{anc}
<div class="section">
  <div style="display:flex; align-items:center; gap:.5rem">
    <div style="font-weight:800; font-size:1.05rem">{ico}{title}</div>
    {'<span class="badge">section</span>' if anchor else ''}
  </div>
  {f'<div class="small" style="margin-top:.25rem">{subtitle}</div>' if subtitle else ''}
</div>
""",
        unsafe_allow_html=True,
    )


# === BADGE / ALERT / TOAST ===
def badge(text: str) -> None:
    st.markdown(f'<span class="badge">{text}</span>', unsafe_allow_html=True)

def notify(text: str, kind: str = "info") -> None:
    """
    Prywatny wrapper na powiadomienia: używa st.toast jeśli jest, inaczej info/warn/error.
    kind: info|success|warning|error
    """
    try:
        st.toast(text, icon="✅" if kind == "success" else "ℹ️")
    except Exception:
        if kind == "success":
            st.success(text)
        elif kind == "warning":
            st.warning(text)
        elif kind == "error":
            st.error(text)
        else:
            st.info(text)


# === DANE: PREVIEW + POBRANIA ===
@st.cache_data(show_spinner=False)
def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def _to_json(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", lines=False, force_ascii=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def _to_parquet(df: pd.DataFrame) -> Tuple[bytes, bool]:
    """
    Zwraca (bytes, used_pyarrow). Gdy brak pyarrow – zwraca (b"", False).
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue(), True
    except Exception:
        return b"", False

def df_preview(
    df: pd.DataFrame,
    *,
    height: int = 380,
    use_container_width: bool = True,
    downloads_base_name: str = "data",
    show_summary: bool = True,
) -> None:
    """
    Karta z podglądem DataFrame i strefą pobrań (CSV/Parquet/JSON).
    """
    apply_global_css()
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    with st.container():
        if show_summary:
            st.markdown(
                f"""
<div class="section">
  <div class="small">Rows: <b>{rows}</b> &nbsp;|&nbsp; Cols: <b>{cols}</b> &nbsp;|&nbsp; Memory: <b>{mem_mb:.2f} MB</b></div>
</div>""",
                unsafe_allow_html=True,
            )

        with st.container():
            st.dataframe(df, height=height, use_container_width=use_container_width)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        csv_bytes = _to_csv(df)
        json_bytes = _to_json(df)
        pq_bytes, has_pa = _to_parquet(df)

        with c1:
            st.download_button(
                "⬇️ CSV",
                data=csv_bytes,
                file_name=f"{downloads_base_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ JSON",
                data=json_bytes,
                file_name=f"{downloads_base_name}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c3:
            if has_pa:
                st.download_button(
                    "⬇️ Parquet",
                    data=pq_bytes,
                    file_name=f"{downloads_base_name}.parquet",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.button("Parquet (pyarrow niedostępny)", disabled=True, use_container_width=True)


# === NARZĘDZIA DODATKOWE ===
def code_block(code: str, *, language: str = "python") -> None:
    st.code(code, language=language)

def spinner(text: str):
    """Context manager: with spinner('...'): ..."""
    return st.spinner(text)


# === PRZYKŁAD UŻYCIA (usuń lub zakomentuj w produkcji) ===
if __name__ == "__main__":
    # Streamlit uruchamiany zwykle przez: streamlit run app.py
    # Ten blok służy jedynie do szybkiego testu komponentów.
    apply_global_css()
    section("KPI demo", "Przykładowe metryki", icon="📊", anchor="kpi")
    kpi_row({
        "AUC": "0.964",
        "F1": "0.912",
        "RMSE": "1.23",
    })
    section("Data preview", "Podgląd danych i pobrania", icon="🧪")
    import numpy as np
    df_demo = pd.DataFrame({"a": np.arange(10), "b": np.random.randn(10)})
    df_preview(df_demo, downloads_base_name="demo_data")
    section("Kod", "Zrzut konfiguracji", icon="🧩")
    code_block("print('hello')\nconfig = {...}")
    notify("To jest przykładowy toast", "info")
