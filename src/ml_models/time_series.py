from __future__ import annotations
import pandas as pd
from prophet import Prophet
from ..utils.helpers import ensure_datetime_index

def fit_prophet(df: pd.DataFrame, target: str, horizon: int = 12):
    df2 = df.copy()
    if target not in df2.columns:
        raise ValueError("Brak kolumny celu w danych.")
    df2 = ensure_datetime_index(df2)
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("Nie znaleziono kolumny czasu – nie można trenować Prophet.")
    ds = df2.index.to_series().reset_index(drop=True).rename("ds")
    y = df2[target].reset_index(drop=True).rename("y")
    train = pd.concat([ds, y], axis=1)
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=horizon, include_history=False)
    fcst = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
    return m, fcst
