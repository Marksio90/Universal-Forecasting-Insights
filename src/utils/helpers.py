import pandas as pd
import numpy as np

def smart_cast_numeric(df: pd.DataFrame, max_unique_frac: float = 0.95) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            try:
                conv = pd.to_numeric(out[c].str.replace(",", ".").str.replace(" ", ""), errors="raise")
                if conv.nunique(dropna=True) / max(1, len(conv)) <= max_unique_frac:
                    out[c] = conv
            except Exception:
                pass
    return out

def infer_problem_type(df: pd.DataFrame, target: str | None) -> str | None:
    if target is None or target not in df.columns:
        return None
    y = df[target]
    if pd.api.types.is_datetime64_any_dtype(df.index) or any(
        pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns
    ):
        return "timeseries"
    if y.nunique() <= max(20, int(0.05 * len(y))) and (y.dtype == "object" or pd.api.types.is_integer_dtype(y)):
        return "classification"
    return "regression"

def ensure_datetime_index(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return df.set_index(c).sort_index()
        try:
            dt = pd.to_datetime(df[c], errors="raise", infer_datetime_format=True)
            if dt.notna().sum() > 0.9 * len(dt):
                return df.assign(_dt_=dt).set_index("_dt_").sort_index().drop(columns=[c], errors="ignore")
        except Exception:
            pass
    return df
