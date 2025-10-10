from __future__ import annotations
import streamlit as st

def inject_global_css():
    try:
        with open("assets/styles/global.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass
