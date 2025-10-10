from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, pandas as pd
SYSTEM_MODES = {
    "Data Assistant": "You are a helpful data analyst focusing on data quality and EDA.",
    "Modeling Coach": "You are a senior ML engineer optimizing models and thresholds.",
    "Forecasting Guru": "You are a time-series expert focused on trends and seasonality."
}
def chat_reply(messages: List[Dict[str,str]], mode: str, df: Optional[pd.DataFrame]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    last = messages[-1]["content"] if messages else ""
    if not key:
        if "schema" in last.lower() and df is not None:
            return f"Schema: {df.shape}, cols={len(df.columns)}"
        return f"[{mode}] Offline. Dostępne: schema/train/predict (z UI)."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys = SYSTEM_MODES.get(mode, SYSTEM_MODES["Data Assistant"])
        chat = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys}]+messages, temperature=0.2, max_tokens=300)
        return chat.choices[0].message.content
    except Exception as e:
        return f"[offline] {last}"
