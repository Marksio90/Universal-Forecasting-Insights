# app.py (Variant A: only built-in page navigation)
from __future__ import annotations
import os
import json
import pathlib
import importlib.util as _ilus
from typing import Optional, Dict, Any

import yaml
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- Paths
APP_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
STYLES = ASSETS / "styles" / "custom.css"
CONFIG = APP_DIR / "config.yaml"
DATA_DIR = APP_DIR / "data"
EXPORTS_DIR = DATA_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Env
load_dotenv()

# --- Config
def _load_cfg() -> Dict[str, Any]:
    base = {"app": {"title": "Intelligent Predictor"}, "logging": {"level": "INFO"}}
    if CONFIG.exists():
        try:
            with open(CONFIG, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            base["app"].update(raw.get("app") or {})
            base["logging"].update(raw.get("logging") or {})
        except Exception:
            pass
    return base

CFG = _load_cfg()

# --- Logging
try:
    from src.utils.logger import configure_logger, get_logger, get_memory_logs, set_level
    configure_logger()
    log = get_logger(__name__)
    log.info("App boot")
except Exception:
    import sys
    from loguru import logger as log
    log.remove()
    log.add(sys.stderr, level="INFO")
    def get_memory_logs(n: Optional[int] = None): return []
    def set_level(level: str): pass

# --- Streamlit page
st.set_page_config(
    page_title=CFG["app"].get("title", "Intelligent Predictor"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS
if STYLES.exists():
    st.markdown(f"<style>{STYLES.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ======================================================================================
# Utils
# ======================================================================================
def _mod_available(name: str) -> bool:
    return _ilus.find_spec(name) is not None

def _import_status(mod: str):
    import importlib, importlib.metadata
    try:
        importlib.import_module(mod)
        ver = importlib.metadata.version(mod.split(".")[0])
        return True, ver, ""
    except Exception as e:
        return False, "", str(e)

def _put_session(key: str, val: Any) -> None:
    st.session_state[key] = val

def _get_df() -> Optional[pd.DataFrame]:
    return st.session_state.get("df") or st.session_state.get("df_raw")

# Demo datasets
def _demo_timeseries(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    rng = np.random.default_rng(42)
    y = 100 + 0.2 * t + 10 * np.sin(2 * np.pi * (t / 7.0)) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"date": idx, "sales": y.round(2)})

def _demo_classification(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    seg = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    p = 1.0 / (1.0 + np.exp(-(0.7 * x1 - 1.1 * x2 + (seg == "B") * 0.6 + rng.normal(0, 0.4, n))))
    y = (p > 0.5).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "segment": seg, "target": y})

# Health checks
def _health_checks() -> Dict[str, Any]:
    yp_ok, _, yp_err = _import_status("ydata_profiling")
    ph_ok, _, ph_err = _import_status("prophet")
    return {
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "prophet": ph_ok,
        "prophet_err": ph_err,
        "ydata_profiling": yp_ok,
        "ydata_profiling_err": yp_err,
        "xgboost": _mod_available("xgboost"),
        "lightgbm": _mod_available("lightgbm"),
        "sqlalchemy": _mod_available("sqlalchemy"),
        "redis": _mod_available("redis"),
    }

# ======================================================================================
# Sidebar — only settings & utilities (no custom nav)
# ======================================================================================
with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    lvl = st.selectbox("Poziom logowania", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    set_level(lvl)

    st.markdown("### 🚀 Nawigacja")
    st.caption("Użyj listy stron powyżej (Upload → Reports).")

    st.markdown("---")
    st.markdown("### 🧪 Dane demo")
    demo_choice = st.selectbox("Załaduj zestaw", ["—", "Timeseries: Sales (daily)", "Classification: Toy"])
    if st.button("Wczytaj demo"):
        if demo_choice.endswith("Sales (daily)"):
            df_demo = _demo_timeseries()
            _put_session("df_raw", df_demo.copy())
            _put_session("df", df_demo.copy())
            _put_session("goal", "Prognoza sprzedaży na kolejny miesiąc")
            st.success("Załadowano timeseries demo (sales). Przejdź do Forecasting.")
        elif demo_choice.startswith("Classification"):
            df_demo = _demo_classification()
            _put_session("df_raw", df_demo.copy())
            _put_session("df", df_demo.copy())
            _put_session("goal", "Klasyfikacja szansy zakupu")
            st.success("Załadowano classification demo. Przejdź do Predictions.")
        else:
            st.info("Wybierz zestaw z listy.")

    if st.button("🧹 Reset sesji"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# ======================================================================================
# Header
# ======================================================================================
col1, col2 = st.columns([1, 7])
with col1:
    logo = (ASSETS / "images" / "logo.png")
    if logo.exists():
        st.image(str(logo), width=64)
with col2:
    st.markdown(f"## {CFG['app'].get('title', 'Intelligent Predictor')} — Intelligent Analytics & Forecasting Suite")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
st.caption(f"OpenAI key status: {'🟢' if OPENAI_KEY else '🔴 (ustaw `OPENAI_API_KEY` w .env)'}")

with st.expander("🧪 Test OpenAI (opcjonalnie)"):
    from src.ai_engine.openai_integrator import chat_completion
    if st.button("Wyślij testowy prompt"):
        out = chat_completion(system="You are a test.", user="Odpowiedz: OK.", model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        st.write(out)
    st.caption("Jeśli nie ma odpowiedzi: sprawdź `.env` i połączenie sieciowe.")

# ======================================================================================
# Intro
# ======================================================================================
st.write(
    """
**Witaj!** To panel główny. Skorzystaj z zakładek (po lewej):

1) **Upload Data** — wczytaj pliki (CSV/XLSX/JSON/DOCX/PDF).  
2) **EDA Analysis** — szybka eksploracja i jakość danych.  
3) **AI Insights** — insighty i rekomendacje oparte na GPT.  
4) **Predictions** — AutoML (regresja/klasyfikacja).  
5) **Forecasting** — modele szeregów (Prophet).  
6) **Reports** — biznesowy raport HTML i eksport ZIP.
"""
)

# ======================================================================================
# Live preview (if df in session)
# ======================================================================================
df = _get_df()
try:
    from src.visualization.dashboards import kpi_board, eda_overview
except Exception:
    kpi_board = eda_overview = None  # type: ignore

if df is not None and isinstance(df, pd.DataFrame) and not df.empty and kpi_board and eda_overview:
    st.markdown("### 🔎 Podgląd bieżącej sesji")
    st.dataframe(df.head(), use_container_width=True)
    st.plotly_chart(kpi_board(df), use_container_width=True)
    st.plotly_chart(eda_overview(df, top_numeric=4), use_container_width=True)
else:
    st.info("Brak aktywnych danych w sesji. Przejdź do **Upload Data** lub załaduj **Dane demo** w pasku bocznym.")

# ======================================================================================
# Health & Integrations
# ======================================================================================
hc = _health_checks()
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("🩺 Health")
    st.markdown("\n".join([
        f"{'🟢' if hc['openai_key'] else '🔴'} OpenAI API key",
        f"{'🟢' if hc['prophet'] else '🔴'} Prophet",
        f"{'🟢' if hc['ydata_profiling'] else '🔴'} ydata-profiling",
    ]))
    if not hc["ydata_profiling"] and hc.get("ydata_profiling_err"):
        st.caption(f"ydata-profiling error: {hc['ydata_profiling_err'][:180]}")
with c2:
    st.subheader("🧠 ML libs")
    st.markdown("\n".join([
        f"{'🟢' if hc['xgboost'] else '🔴'} XGBoost",
        f"{'🟢' if hc['lightgbm'] else '🔴'} LightGBM",
    ]))
with c3:
    st.subheader("🗄️ Storage / DB")
    st.markdown("\n".join([
        f"{'🟢' if hc['sqlalchemy'] else '🔴'} SQLAlchemy / SQLite",
        f"{'🟢' if hc['redis'] else '🔴'} Redis cache",
        f"🟢 Exports dir: `{EXPORTS_DIR}`",
    ]))

# ======================================================================================
# Model registry (summary)
# ======================================================================================
with st.expander("🧾 Rejestr modeli (skrót)"):
    try:
        from src.ml_models.model_registry import list_models as registry_list
        items = registry_list()
        if items:
            st.dataframe(pd.DataFrame(items), use_container_width=True, height=240)
        else:
            st.caption("Brak zarejestrowanych modeli (jeszcze). Przejdź do **Predictions**.")
    except Exception as e:
        st.caption(f"Rejestr modeli niedostępny: {e}")

# ======================================================================================
# Session state / Debug
# ======================================================================================
with st.expander("🛠️ Stan sesji i logi (debug)"):
    st.json({
        "data_loaded": bool(df is not None and not getattr(df, "empty", True)),
        "columns": list(df.columns)[:25] if isinstance(df, pd.DataFrame) else None,
        "target": st.session_state.get("target"),
        "problem_type": st.session_state.get("problem_type"),
        "trained_model": "model" in st.session_state,
        "goal": st.session_state.get("goal"),
    })
    st.text("Ostatnie logi:")
    try:
        logs_txt = "".join(get_memory_logs(300)) or "(pusto)"
        st.code(logs_txt, language="text")
    except Exception:
        st.write("(bufor logów niedostępny)")

# ======================================================================================
# Quick export (shortcut; full in Reports page)
# ======================================================================================
with st.expander("📦 Szybki eksport (demo)"):
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Pobierz dane (CSV)", data=csv_bytes, file_name="data.csv", mime="text/csv")
        ctx = {
            "title": "Raport (quick)",
            "metrics": {
                "rows": len(df),
                "cols": df.shape[1],
                "problem_type": st.session_state.get("problem_type"),
                "target": st.session_state.get("target"),
            },
            "notes": st.session_state.get("goal") or "",
        }
        try:
            from src.ai_engine.report_generator import build_report_html
            html = build_report_html(ctx)
        except Exception:
            html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{ctx['title']}</title></head>
<body><h1>{ctx['title']}</h1><pre>{json.dumps(ctx, ensure_ascii=False, indent=2)}</pre></body></html>"""
        st.download_button("🧾 Pobierz szybki raport (HTML)", data=html, file_name="report.html", mime="text/html")
    else:
        st.caption("Załaduj dane, aby udostępnić szybki eksport.")

# ======================================================================================
# Footer
# ======================================================================================
st.markdown("---")
st.caption("© Intelligent Predictor — Streamlit suite. Przejdź do **Upload Data** aby zacząć pracę z własnymi plikami.")
