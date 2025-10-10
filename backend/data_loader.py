from __future__ import annotations
from typing import Optional
import pandas as pd
from src.utils.helpers import smart_read

def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        df = smart_read(uploaded_file)
        # try parse dates
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], errors="raise", utc=False)
                    # Heuristic: keep only if many parses succeed
                    if parsed.notna().mean() > 0.8:
                        df[c] = parsed
                except Exception:
                    pass
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")
