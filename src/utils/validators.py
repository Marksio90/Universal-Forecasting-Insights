import pandas as pd

def basic_quality_checks(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing_pct": float((df.isna().sum().sum()) / (df.size or 1)),
        "dupes": int(df.duplicated().sum())
    }
