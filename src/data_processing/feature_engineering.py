from __future__ import annotations
import pandas as pd
import numpy as np

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # simple date expansions
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[f"{c}_year"] = df[c].dt.year
            df[f"{c}_month"] = df[c].dt.month
            df[f"{c}_dow"] = df[c].dt.dayofweek
    return df
