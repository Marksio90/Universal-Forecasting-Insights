from __future__ import annotations
from typing import Dict, Any, Optional
import os, json
import pandas as pd
from ..utils.rate_limiter import rate_limited
from .prompt_templates import SYSTEM_INSIGHTS

def _get_openai_key() -> Optional[str]:
    # st.secrets-like pattern without importing streamlit here
    key = os.getenv("OPENAI_API_KEY", None)
    return key

@rate_limited(0.6)
def generate_insights(df: pd.DataFrame, *, target: Optional[str]=None) -> Dict[str, Any]:
    key = _get_openai_key()
    # Offline fallback – deterministic summary
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "suggested_target": target or (df.columns[-1] if df.shape[1] > 0 else None),
        "notes": [
            "Sprawdź brakujące wartości oraz duplikaty.",
            "Rozważ normalizację/standaryzację dla modeli liniowych.",
            "Wypróbuj XGBoost/LightGBM dla nieliniowych zależności."
        ]
    }
    # If key present, we would call OpenAI for richer narrative (kept simple to avoid runtime deps here)
    # Returning consistent structure either way
    return {"mode": "online" if key else "offline", "system": SYSTEM_INSIGHTS, "summary": summary}
