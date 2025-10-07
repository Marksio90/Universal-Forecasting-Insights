from __future__ import annotations
import pandas as pd
from .time_series import fit_prophet

def forecast(df: pd.DataFrame, target: str, horizon: int = 12):
    model, fcst = fit_prophet(df, target, horizon=horizon)
    return model, fcst
