from __future__ import annotations
from typing import Tuple
import pandas as pd
from prophet import Prophet

def prepare_ts(df: pd.DataFrame, date_col: str, target: str) -> pd.DataFrame:
    ts = df[[date_col, target]].dropna().copy()
    ts.columns = ["ds", "y"]
    return ts

def fit_forecast(ts: pd.DataFrame, periods: int = 30) -> Tuple[Prophet, pd.DataFrame]:
    m = Prophet()
    m.fit(ts)
    fut = m.make_future_dataframe(periods=periods)
    fcst = m.predict(fut)
    return m, fcst
