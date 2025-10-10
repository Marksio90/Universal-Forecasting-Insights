from __future__ import annotations
import streamlit as st
from functools import lru_cache

def cache_data(func):
    return st.cache_data(show_spinner=False)(func)

def cache_resource(func):
    return st.cache_resource(show_spinner=False)(func)

@lru_cache(maxsize=128)
def memoize(key: str) -> str:
    return key
