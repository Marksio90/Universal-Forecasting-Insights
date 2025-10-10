from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class DataHealth:
    rows: int
    cols: int
    missing_pct: float
    duplicated_rows: int
    memory_mb: float
    numeric_cols: int
    categorical_cols: int
    datetime_cols: int

def basic_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    rows, cols = df.shape
    missing_pct = float(df.isna().mean().mean()) if rows*cols > 0 else 0.0
    duplicated_rows = int(df.duplicated().sum())
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024**2)
    num = int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
    cat = int(sum(pd.api.types.is_string_dtype(df[c]) for c in df.columns))
    dt  = int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns))
    return {
        "health": DataHealth(rows, cols, missing_pct, duplicated_rows, memory_mb, num, cat, dt),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "na_count": df.isna().sum().to_dict(),
    }

def infer_problem_type(df: pd.DataFrame, target: str) -> str:
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        # if few unique values it's likely classification
        return "classification" if y.nunique() <= 10 else "regression"
    return "classification"
