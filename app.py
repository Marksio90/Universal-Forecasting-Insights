from __future__ import annotations
import os
import pathlib
import yaml
import streamlit as st
from dotenv import load_dotenv

APP_DIR = pathlib.Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"
STYLES = ASSETS / "styles" / "custom.css"
CONFIG = APP_DIR / "config.yaml"

load_dotenv()

with open(CONFIG, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

st.set_page_config(
    page_title=CFG["app"]["title"],
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if STYLES.exists():
    st.markdown(f"<style>{STYLES.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

col1, col2 = st.columns([1,6])
with col1:
    logo = (ASSETS / "images" / "logo.png")
    if logo.exists():
        st.image(str(logo), width=64)
with col2:
    st.markdown(f"## {CFG['app']['title']} — Intelligent Analytics & Forecasting Suite")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
status_ok = "🟢" if OPENAI_KEY else "🔴"
st.caption(f"OpenAI key status: {status_ok}")

st.write(
    '''
**Witaj!** To panel główny. Skorzystaj z zakładek (po lewej):
1) **Upload Data** – wczytaj pliki (CSV/XLSX/JSON/DOCX/PDF).  
2) **EDA Analysis** – szybka eksploracja i jakość danych.  
3) **AI Insights** – insighty i rekomendacje oparte na GPT.  
4) **Predictions** – AutoML (regresja/klasyfikacja) + SHAP.  
5) **Forecasting** – modele szeregów (Prophet) + sMAPE/MASE.  
6) **Reports** – biznesowy raport HTML i eksport ZIP.
'''
)

with st.expander("📌 Stan sesji (debug)"):
    st.json({
        "data_loaded": "df" in st.session_state,
        "target": st.session_state.get("target"),
        "problem_type": st.session_state.get("problem_type"),
        "trained_model": "model" in st.session_state
    })

st.info("Przejdź do **1_📤_Upload_Data** aby zacząć.")
