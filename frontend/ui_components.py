from __future__ import annotations
import streamlit as st

def kpi_row(items: dict):
    cols = st.columns(len(items))
    for (k,v), c in zip(items.items(), cols):
        with c:
            st.metric(k, v)

def section(title: str):
    st.markdown(f"### {title}")
